"""The concealed-damage rule engine.

The contract asks for concealed-damage flags *with the rule that fired*, so this
is an explicit engine over a YAML rule file rather than a model. Two consequences
shape the design:

* **No ``eval``.** Predicates are structured data -- ``all_of`` / ``any_of`` /
  ``none_of`` over ``{field, op, value}`` leaves. A rule file is a data file, and
  a data file that can execute arbitrary Python is not one.
* **Every leaf records what it saw.** A flag carries the actual value of each
  condition that led to it, so "why did this fire" is answered by the output
  rather than by re-running anything.

A rule that references an unknown field is a loading error, not a silent
never-fires: the failure mode of a rule engine is a rule that quietly does
nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import yaml

log = logging.getLogger("cozmo.semantics.rules")

DEFAULT_RULES_PATH = Path(__file__).with_name("rules.yaml")

# Fields a predicate may test. Anything else is a typo, and typos in a rule file
# are silent failures unless they are rejected at load time.
RULE_FIELDS = {
    "damage_class",             # water | mold | fire_smoke | impact | crack | stain | missing_material
    "surface_kind",             # wall | floor | ceiling
    "area_m2",
    "max_extent_m",
    "min_height_above_floor_m",
    "max_height_above_floor_m",
    "severity",
    "confidence",
    "distance_to_exterior_corner_m",
    "distance_to_opening_m",
    "wall_has_opening",
    "room_id",
    "surface_id",
    "is_exterior_surface",
}

OPERATORS: Dict[str, Callable[[Any, Any], bool]] = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "lt": lambda a, b: a is not None and a < b,
    "lte": lambda a, b: a is not None and a <= b,
    "gt": lambda a, b: a is not None and a > b,
    "gte": lambda a, b: a is not None and a >= b,
    "in": lambda a, b: a in b,
    "not_in": lambda a, b: a not in b,
    "contains": lambda a, b: b in (a or ()),
}


class RuleError(ValueError):
    """A rule file that cannot be trusted to mean what it says."""


@dataclass
class Evaluation:
    """One leaf condition and the value it actually saw."""

    field: str
    op: str
    expected: Any
    actual: Any
    passed: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field, "op": self.op,
            "expected": self.expected, "actual": self.actual, "passed": self.passed,
        }


@dataclass
class ConcealedRule:
    id: str
    text: str
    predicate: Mapping[str, Any]
    severity: str = "medium"
    probability: float = 0.5
    recommended_action: str = ""
    inspection_priority: int = 3

    def evaluate(self, context: Mapping[str, Any]) -> tuple:
        """Returns ``(fired, evaluations)``."""
        evaluations: List[Evaluation] = []
        fired = _evaluate(self.predicate, context, evaluations)
        return fired, evaluations


@dataclass
class RuleFiring:
    """A rule that fired, and the numbers that made it fire."""

    rule: ConcealedRule
    region_id: str
    room_id: str
    surface_id: str
    evaluations: List[Evaluation] = field(default_factory=list)

    def triggering_values(self) -> Dict[str, Any]:
        """The inputs that satisfied the rule -- what the contract asks to see."""
        return {e.field: e.actual for e in self.evaluations if e.passed}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule.id,
            "region_id": self.region_id,
            "surface_id": self.surface_id,
            "evaluations": [e.as_dict() for e in self.evaluations],
            "triggering_values": self.triggering_values(),
        }


def _evaluate(node: Mapping[str, Any], context: Mapping[str, Any],
              out: List[Evaluation]) -> bool:
    """Walk a predicate tree, recording every leaf it touches."""
    if "all_of" in node:
        # Every branch is evaluated, not short-circuited: the report is more
        # useful when it shows all the conditions, including the ones that failed.
        results = [_evaluate(child, context, out) for child in node["all_of"]]
        return all(results)
    if "any_of" in node:
        results = [_evaluate(child, context, out) for child in node["any_of"]]
        return any(results)
    if "none_of" in node:
        results = [_evaluate(child, context, out) for child in node["none_of"]]
        return not any(results)

    name, op = node["field"], node["op"]
    expected = node["value"]
    actual = context.get(name)
    try:
        passed = bool(OPERATORS[op](actual, expected))
    except TypeError:
        # Comparing None or mismatched types is a non-match, not a crash: a
        # region with an unmeasurable field simply does not satisfy the rule.
        passed = False
    out.append(Evaluation(field=name, op=op, expected=expected, actual=actual, passed=passed))
    return passed


def _validate(node: Mapping[str, Any], rule_id: str) -> None:
    for key in ("all_of", "any_of", "none_of"):
        if key in node:
            children = node[key]
            if not isinstance(children, list) or not children:
                raise RuleError(f"{rule_id}: '{key}' must be a non-empty list")
            for child in children:
                _validate(child, rule_id)
            return
    if "field" not in node or "op" not in node or "value" not in node:
        raise RuleError(f"{rule_id}: leaf condition needs field, op and value: {node!r}")
    if node["field"] not in RULE_FIELDS:
        raise RuleError(
            f"{rule_id}: unknown field '{node['field']}'. "
            f"Known fields: {sorted(RULE_FIELDS)}"
        )
    if node["op"] not in OPERATORS:
        raise RuleError(f"{rule_id}: unknown operator '{node['op']}'")


def load_rules(path: Optional[Path] = None) -> List[ConcealedRule]:
    path = Path(path or DEFAULT_RULES_PATH)
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict) or "rules" not in document:
        raise RuleError(f"{path}: expected a mapping with a 'rules' key")

    rules: List[ConcealedRule] = []
    seen = set()
    for entry in document["rules"]:
        rule_id = entry.get("id")
        if not rule_id:
            raise RuleError(f"{path}: a rule has no id")
        if rule_id in seen:
            raise RuleError(f"{path}: duplicate rule id '{rule_id}'")
        seen.add(rule_id)
        if not entry.get("text"):
            raise RuleError(f"{rule_id}: a rule must explain itself in 'text'")
        predicate = entry.get("predicate")
        if not predicate:
            raise RuleError(f"{rule_id}: a rule must have a predicate")
        _validate(predicate, rule_id)

        rules.append(ConcealedRule(
            id=rule_id,
            text=" ".join(str(entry["text"]).split()),
            predicate=predicate,
            severity=entry.get("severity", "medium"),
            probability=float(entry.get("probability", 0.5)),
            recommended_action=" ".join(str(entry.get("recommended_action", "")).split()),
            inspection_priority=int(entry.get("inspection_priority", 3)),
        ))

    log.info("loaded %d concealed-damage rules from %s", len(rules), path.name)
    return rules


class RuleEngine:
    """Applies the rule set to damage regions in their geometric context."""

    def __init__(self, rules: Optional[Sequence[ConcealedRule]] = None) -> None:
        self.rules = list(rules) if rules is not None else load_rules()

    def evaluate_region(self, context: Mapping[str, Any]) -> List[RuleFiring]:
        firings: List[RuleFiring] = []
        for rule in self.rules:
            fired, evaluations = rule.evaluate(context)
            if fired:
                firings.append(RuleFiring(
                    rule=rule,
                    region_id=str(context.get("region_id", "")),
                    room_id=str(context.get("room_id", "")),
                    surface_id=str(context.get("surface_id", "")),
                    evaluations=evaluations,
                ))
        return firings

    def evaluate(self, contexts: Sequence[Mapping[str, Any]]) -> List[RuleFiring]:
        firings: List[RuleFiring] = []
        for context in contexts:
            firings.extend(self.evaluate_region(context))
        log.info("concealed-damage rules: %d firing(s) across %d region(s)",
                 len(firings), len(contexts))
        return firings
