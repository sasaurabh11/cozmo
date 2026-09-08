"""The scoreboard.

Reads laser/tape ground truth and one or more ``plan.json`` files, and reports
each gate as PASS, FAIL or SKIP. Two rules shape the whole file:

1. **A gate with no ground truth behind it is SKIP, never PASS.** Silence must
   never read as success.
2. **Detection is scored, not just dimension.** A missed opening and a phantom
   opening each count as a miss, so a pipeline cannot buy accuracy by only
   reporting the openings it is sure about.

Ground truth CSV columns::

    room,element,element_id,dimension,value_m,method,notes

``element`` is one of wall | opening | room | property. ``element_id`` matches
the id in the plan (wall id, opening id, room id, or ``property``).
``dimension`` is length | width | height | ceiling_height | floor_area |
footprint_area. ``value_m`` is metres (or m^2 for the area dimensions).
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..schema import Measurement, Plan, Tier

SCORER_VERSION = "1.0.0"
RESULTS_FILENAME = "results.json"

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

# Binary floating point puts 2.450 - 2.435 at 1.5000000000000013 cm. Without
# this slack a measurement exactly on a gate would fail it, which is not what
# "<= 1.5 cm" means to anyone holding a laser measurer.
EPS = 1e-9

AREA_DIMENSIONS = {"floor_area", "footprint_area"}
VALID_ELEMENTS = {"wall", "opening", "room", "property"}


# --------------------------------------------------------------------------
# Tolerances
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Tolerances:
    """One tier's gates.

    Values taken directly from the brief where it states them. Where the brief
    states a gate without a tier (ceiling height, opening widths), it is applied
    literally at the LiDAR tier and widened at photo and video -- an assumption,
    flagged here rather than buried.

    Ceiling height is widened *less* than the tier's plan-scale looseness (3% at
    photo, 1.5% at video, against +-8% and +-3% on wall lengths). A ceiling is a
    single vertical extent measured in one place; it does not accumulate the
    pose drift that stretches a wall run across a room, so inheriting the wall
    tolerance would have handed the photo tier a 19.6 cm ceiling gate, which is
    not a gate.
    """

    label: str
    wall_abs_m: float
    wall_rel: float
    wall_pass_fraction: float
    ceiling_abs_m: float
    ceiling_rel: float
    opening_abs_m: float
    opening_rel: float
    opening_pass_fraction: float
    footprint_rel: float

    def wall_ok(self, error_m: float, truth_m: float) -> bool:
        return error_m <= max(self.wall_abs_m, self.wall_rel * abs(truth_m)) + EPS

    def ceiling_ok(self, error_m: float, truth_m: float) -> bool:
        return error_m <= max(self.ceiling_abs_m, self.ceiling_rel * abs(truth_m)) + EPS

    def opening_ok(self, error_m: float, truth_m: float) -> bool:
        return error_m <= max(self.opening_abs_m, self.opening_rel * abs(truth_m)) + EPS


# Gates that are tier-independent in the brief.
CEILING_SPREAD_MAX_M = 0.01       # spread across repeat captures of one room
REPEATABILITY_ABS_M = 0.01        # per wall, between two captures
REPEATABILITY_REL = 0.005         # ... or 0.5%, whichever is kinder
INTERVAL_COVERAGE_MIN = 0.90      # 95% intervals; 0.90 allows finite-sample slack

TOLERANCES: Dict[Tier, Tolerances] = {
    Tier.LIDAR: Tolerances(
        label="LiDAR",
        wall_abs_m=0.02, wall_rel=0.01, wall_pass_fraction=0.85,
        ceiling_abs_m=0.015, ceiling_rel=0.0,
        opening_abs_m=0.02, opening_rel=0.0, opening_pass_fraction=0.85,
        footprint_rel=0.02,
    ),
    Tier.VIDEO: Tolerances(
        label="Video",
        wall_abs_m=0.02, wall_rel=0.03, wall_pass_fraction=0.85,
        ceiling_abs_m=0.015, ceiling_rel=0.015,
        opening_abs_m=0.02, opening_rel=0.03, opening_pass_fraction=0.85,
        footprint_rel=0.03,
    ),
    Tier.PHOTO: Tolerances(
        label="Photo",
        wall_abs_m=0.02, wall_rel=0.08, wall_pass_fraction=0.85,
        ceiling_abs_m=0.015, ceiling_rel=0.03,
        opening_abs_m=0.02, opening_rel=0.08, opening_pass_fraction=0.85,
        footprint_rel=0.08,
    ),
}


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GTRow:
    room: str
    element: str
    element_id: str
    dimension: str
    value_m: float
    method: str = ""
    notes: str = ""


@dataclass
class GroundTruth:
    path: Path
    rows: List[GTRow]

    def by_element(self, element: str, dimension: Optional[str] = None) -> List[GTRow]:
        return [
            r for r in self.rows
            if r.element == element and (dimension is None or r.dimension == dimension)
        ]

    def lookup(self, element_id: str, dimension: str) -> Optional[GTRow]:
        for row in self.rows:
            if row.element_id == element_id and row.dimension == dimension:
                return row
        return None

    def opening_ids_for_room(self, room: str) -> List[str]:
        return [r.element_id for r in self.rows if r.element == "opening" and r.room == room]

    @property
    def rooms_with_openings(self) -> set:
        return {r.room for r in self.rows if r.element == "opening"}


def load_ground_truth(path: Path) -> GroundTruth:
    path = Path(path)
    rows: List[GTRow] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {"room", "element", "element_id", "dimension", "value_m"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")
        for line_no, raw in enumerate(reader, start=2):
            if not (raw.get("element_id") or "").strip():
                continue
            element = (raw["element"] or "").strip().lower()
            if element not in VALID_ELEMENTS:
                raise ValueError(
                    f"{path}:{line_no} unknown element '{element}' (expected one of {sorted(VALID_ELEMENTS)})"
                )
            try:
                value = float(raw["value_m"])
            except (TypeError, ValueError):
                raise ValueError(f"{path}:{line_no} value_m '{raw.get('value_m')}' is not a number")
            rows.append(
                GTRow(
                    room=(raw.get("room") or "").strip(),
                    element=element,
                    element_id=raw["element_id"].strip(),
                    dimension=(raw["dimension"] or "").strip().lower(),
                    value_m=value,
                    method=(raw.get("method") or "").strip(),
                    notes=(raw.get("notes") or "").strip(),
                )
            )
    if not rows:
        raise ValueError(f"{path} contains no usable ground-truth rows")
    return GroundTruth(path=path, rows=rows)


# --------------------------------------------------------------------------
# Loaded plans
# --------------------------------------------------------------------------


@dataclass
class LoadedPlan:
    path: Path
    plan: Plan

    @property
    def capture_id(self) -> str:
        return self.plan.capture_id

    @property
    def tier(self) -> Tier:
        return self.plan.tier

    def walls(self) -> Dict[str, Tuple[str, Measurement]]:
        """wall id -> (room id, length)."""
        return {w.id: (room.id, w.length) for room in self.plan.rooms for w in room.walls}

    def openings(self) -> Dict[str, Tuple[str, Measurement]]:
        """opening id -> (room id, width)."""
        return {o.id: (room.id, o.width) for room in self.plan.rooms for o in room.openings}


def load_plans(results_dir: Path) -> List[LoadedPlan]:
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise NotADirectoryError(f"results directory does not exist: {results_dir}")
    if results_dir.is_file():
        candidates = [results_dir]
    else:
        candidates = sorted(results_dir.rglob("plan.json"))
    if not candidates:
        raise FileNotFoundError(f"no plan.json found under {results_dir}")

    plans: List[LoadedPlan] = []
    for path in candidates:
        try:
            plans.append(LoadedPlan(path=path, plan=Plan.from_json(path.read_text())))
        except Exception as exc:  # noqa: BLE001 - surface the file that broke
            raise ValueError(f"{path} is not a valid plan.json: {exc}") from exc
    return plans


# --------------------------------------------------------------------------
# Gate results
# --------------------------------------------------------------------------


@dataclass
class GateResult:
    gate: str
    scope: str                 # capture id, or "tier:room" for cross-capture gates
    tier: str
    status: str                # PASS | FAIL | SKIP
    metric: str                # human-readable measured value
    threshold: str             # human-readable gate
    value: Optional[float] = None
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        out = asdict(self)
        return out


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _rel_error(pred: float, truth: float) -> Optional[float]:
    return abs(pred - truth) / abs(truth) if truth else None


# --------------------------------------------------------------------------
# Per-capture gates
# --------------------------------------------------------------------------


def gate_wall_lengths(lp: LoadedPlan, gt: GroundTruth) -> GateResult:
    tol = TOLERANCES[lp.tier]
    comparisons: List[Dict[str, Any]] = []
    for wall_id, (room_id, measurement) in sorted(lp.walls().items()):
        truth_row = gt.lookup(wall_id, "length")
        if truth_row is None:
            continue
        err = abs(measurement.value - truth_row.value_m)
        comparisons.append({
            "wall_id": wall_id,
            "room": room_id,
            "predicted_m": measurement.value,
            "truth_m": truth_row.value_m,
            "abs_error_m": round(err, 4),
            "rel_error": round(_rel_error(measurement.value, truth_row.value_m) or 0.0, 5),
            "within_tolerance": tol.wall_ok(err, truth_row.value_m),
            "interval_covers_truth": measurement.contains(truth_row.value_m),
        })

    if not comparisons:
        return GateResult(
            gate="wall_lengths", scope=lp.capture_id, tier=lp.tier.value, status=SKIP,
            metric="no wall ground truth", threshold="-",
            detail={"reason": "ground truth contains no wall lengths matching this plan's wall ids"},
        )

    passed = sum(1 for c in comparisons if c["within_tolerance"])
    fraction = passed / len(comparisons)
    worst = max(comparisons, key=lambda c: c["abs_error_m"])
    threshold = (
        f"<= max({tol.wall_abs_m * 100:.1f} cm, {_pct(tol.wall_rel)}) on >= {_pct(tol.wall_pass_fraction)}"
    )
    return GateResult(
        gate="wall_lengths", scope=lp.capture_id, tier=lp.tier.value,
        status=PASS if fraction >= tol.wall_pass_fraction - EPS else FAIL,
        metric=f"{passed}/{len(comparisons)} within tol ({_pct(fraction)}), worst {worst['abs_error_m'] * 100:.1f} cm",
        threshold=threshold, value=round(fraction, 4),
        detail={"worst_wall": worst["wall_id"], "comparisons": comparisons},
    )


def gate_ceiling_height(lp: LoadedPlan, gt: GroundTruth) -> GateResult:
    tol = TOLERANCES[lp.tier]
    comparisons: List[Dict[str, Any]] = []
    for room in lp.plan.rooms:
        truth_row = gt.lookup(room.id, "ceiling_height")
        if truth_row is None:
            continue
        err = abs(room.ceiling_height.value - truth_row.value_m)
        comparisons.append({
            "room": room.id,
            "predicted_m": room.ceiling_height.value,
            "truth_m": truth_row.value_m,
            "abs_error_m": round(err, 4),
            "within_tolerance": tol.ceiling_ok(err, truth_row.value_m),
            "interval_covers_truth": room.ceiling_height.contains(truth_row.value_m),
        })

    if not comparisons:
        return GateResult(
            gate="ceiling_height", scope=lp.capture_id, tier=lp.tier.value, status=SKIP,
            metric="no ceiling ground truth", threshold="-",
            detail={"reason": "ground truth contains no ceiling_height rows for this plan's rooms"},
        )

    worst = max(comparisons, key=lambda c: c["abs_error_m"])
    all_ok = all(c["within_tolerance"] for c in comparisons)
    limit_cm = max(tol.ceiling_abs_m, tol.ceiling_rel * worst["truth_m"]) * 100
    return GateResult(
        gate="ceiling_height", scope=lp.capture_id, tier=lp.tier.value,
        status=PASS if all_ok else FAIL,
        metric=f"worst {worst['abs_error_m'] * 100:.1f} cm ({worst['room']}), {len(comparisons)} room(s)",
        threshold=f"<= {limit_cm:.1f} cm every room",
        value=round(worst["abs_error_m"], 4),
        detail={"comparisons": comparisons},
    )


def gate_opening_widths(lp: LoadedPlan, gt: GroundTruth) -> GateResult:
    """Dimension and detection in one row.

    Denominator = matched + missed + phantom. A missed opening (in ground truth,
    absent from the plan) and a phantom opening (in the plan, absent from ground
    truth) each count as a miss, exactly as the brief specifies.
    """
    tol = TOLERANCES[lp.tier]
    predicted = lp.openings()
    plan_room_ids = {r.id for r in lp.plan.rooms}

    truth_openings = [
        r for r in gt.by_element("opening", "width")
        if (not r.room) or r.room in plan_room_ids
    ]

    matched: List[Dict[str, Any]] = []
    missed: List[str] = []
    for row in truth_openings:
        if row.element_id not in predicted:
            missed.append(row.element_id)
            continue
        _room, measurement = predicted[row.element_id]
        err = abs(measurement.value - row.value_m)
        matched.append({
            "opening_id": row.element_id,
            "room": row.room,
            "predicted_m": measurement.value,
            "truth_m": row.value_m,
            "abs_error_m": round(err, 4),
            "within_tolerance": tol.opening_ok(err, row.value_m),
            "interval_covers_truth": measurement.contains(row.value_m),
        })

    # Phantoms are only counted in rooms ground truth actually covers; otherwise
    # an unmeasured room would penalise a correct detection.
    covered_rooms = gt.rooms_with_openings
    truth_ids = {r.element_id for r in truth_openings}
    phantom = [
        opening_id for opening_id, (room_id, _m) in sorted(predicted.items())
        if room_id in covered_rooms and opening_id not in truth_ids
    ]

    denominator = len(matched) + len(missed) + len(phantom)
    if denominator == 0:
        return GateResult(
            gate="opening_widths", scope=lp.capture_id, tier=lp.tier.value, status=SKIP,
            metric="no opening ground truth", threshold="-",
            detail={"reason": "ground truth contains no opening widths for this plan's rooms"},
        )

    hits = sum(1 for m in matched if m["within_tolerance"])
    fraction = hits / denominator
    return GateResult(
        gate="opening_widths", scope=lp.capture_id, tier=lp.tier.value,
        status=PASS if fraction >= tol.opening_pass_fraction - EPS else FAIL,
        metric=(
            f"{hits}/{denominator} ({_pct(fraction)}); "
            f"{len(missed)} missed, {len(phantom)} phantom"
        ),
        threshold=f"<= {tol.opening_abs_m * 100:.0f} cm on >= {_pct(tol.opening_pass_fraction)} incl. detection",
        value=round(fraction, 4),
        detail={"matched": matched, "missed": missed, "phantom": phantom, "denominator": denominator},
    )


def gate_footprint(lp: LoadedPlan, gt: GroundTruth) -> GateResult:
    tol = TOLERANCES[lp.tier]
    truth_row = gt.lookup("property", "footprint_area") or gt.lookup(
        lp.capture_id, "footprint_area"
    )
    if truth_row is None:
        return GateResult(
            gate="footprint", scope=lp.capture_id, tier=lp.tier.value, status=SKIP,
            metric="no footprint ground truth", threshold="-",
            detail={"reason": "ground truth has no property/footprint_area row"},
        )

    predicted = lp.plan.property_totals.footprint_area
    rel = _rel_error(predicted.value, truth_row.value_m) or 0.0
    return GateResult(
        gate="footprint", scope=lp.capture_id, tier=lp.tier.value,
        status=PASS if rel <= tol.footprint_rel + EPS else FAIL,
        metric=f"{predicted.value:.2f} m2 vs {truth_row.value_m:.2f} m2 ({_pct(rel)})",
        threshold=f"within +-{_pct(tol.footprint_rel)}",
        value=round(rel, 5),
        detail={
            "predicted_m2": predicted.value,
            "truth_m2": truth_row.value_m,
            "interval_covers_truth": predicted.contains(truth_row.value_m),
            "drift_correction_enabled": lp.plan.drift_correction.enabled,
            "drift_method": lp.plan.drift_correction.method.value,
        },
    )


def gate_interval_coverage(lp: LoadedPlan, gt: GroundTruth) -> GateResult:
    """Calibration: do the 95% intervals actually contain the truth?

    Scored at every tier. Wide intervals pass this row cheaply, which is why the
    mean half-width is reported alongside -- a pipeline that widens its way to
    coverage is visible here rather than hidden.
    """
    checks: List[Dict[str, Any]] = []

    def add(kind: str, element_id: str, measurement: Measurement, truth: float) -> None:
        checks.append({
            "kind": kind,
            "element_id": element_id,
            "predicted": measurement.value,
            "truth": truth,
            "half_width": round(measurement.half_width, 4),
            "covered": measurement.contains(truth),
        })

    for wall_id, (_room, measurement) in sorted(lp.walls().items()):
        row = gt.lookup(wall_id, "length")
        if row:
            add("wall_length", wall_id, measurement, row.value_m)
    for opening_id, (_room, measurement) in sorted(lp.openings().items()):
        row = gt.lookup(opening_id, "width")
        if row:
            add("opening_width", opening_id, measurement, row.value_m)
    for room in lp.plan.rooms:
        row = gt.lookup(room.id, "ceiling_height")
        if row:
            add("ceiling_height", room.id, room.ceiling_height, row.value_m)
        row = gt.lookup(room.id, "floor_area")
        if row:
            add("floor_area", room.id, room.floor_area, row.value_m)
    row = gt.lookup("property", "footprint_area")
    if row:
        add("footprint_area", "property", lp.plan.property_totals.footprint_area, row.value_m)

    if not checks:
        return GateResult(
            gate="interval_coverage", scope=lp.capture_id, tier=lp.tier.value, status=SKIP,
            metric="nothing to check", threshold="-",
            detail={"reason": "no measurement in this plan has matching ground truth"},
        )

    covered = sum(1 for c in checks if c["covered"])
    fraction = covered / len(checks)
    mean_half_width = sum(c["half_width"] for c in checks) / len(checks)
    return GateResult(
        gate="interval_coverage", scope=lp.capture_id, tier=lp.tier.value,
        status=PASS if fraction >= INTERVAL_COVERAGE_MIN - EPS else FAIL,
        metric=f"{covered}/{len(checks)} covered ({_pct(fraction)}), mean +-{mean_half_width * 100:.1f} cm",
        threshold=f">= {_pct(INTERVAL_COVERAGE_MIN)} of 95% intervals",
        value=round(fraction, 4),
        detail={"checks": checks, "mean_half_width_m": round(mean_half_width, 4)},
    )


# --------------------------------------------------------------------------
# Cross-capture gates
# --------------------------------------------------------------------------


def _repeat_groups(plans: Sequence[LoadedPlan]) -> Dict[Tuple[str, str], List[LoadedPlan]]:
    """Group plans by (tier, room id): two captures of the same room at the same
    tier are exactly what the repeatability gate compares."""
    groups: Dict[Tuple[str, str], List[LoadedPlan]] = {}
    for lp in plans:
        for room in lp.plan.rooms:
            groups.setdefault((lp.tier.value, room.id), []).append(lp)
    return {key: members for key, members in groups.items() if len(members) > 1}


def gate_repeatability(plans: Sequence[LoadedPlan]) -> List[GateResult]:
    groups = _repeat_groups(plans)
    if not groups:
        return [GateResult(
            gate="repeatability", scope="-", tier="-", status=SKIP,
            metric="no repeated captures", threshold=f"<= {REPEATABILITY_ABS_M * 100:.0f} cm or {_pct(REPEATABILITY_REL)} per wall",
            detail={"reason": "benchmark set contains no room captured twice at the same tier"},
        )]

    results: List[GateResult] = []
    for (tier, room_id), members in sorted(groups.items()):
        comparisons: List[Dict[str, Any]] = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                walls_a, walls_b = a.walls(), b.walls()
                for wall_id in sorted(set(walls_a) & set(walls_b)):
                    room_a, m_a = walls_a[wall_id]
                    if room_a != room_id:
                        continue
                    m_b = walls_b[wall_id][1]
                    diff = abs(m_a.value - m_b.value)
                    reference = max(abs(m_a.value), abs(m_b.value)) or 1.0
                    comparisons.append({
                        "wall_id": wall_id,
                        "capture_a": a.capture_id, "capture_b": b.capture_id,
                        "a_m": m_a.value, "b_m": m_b.value,
                        "diff_m": round(diff, 4),
                        "within_tolerance": diff <= max(
                            REPEATABILITY_ABS_M, REPEATABILITY_REL * reference
                        ) + EPS,
                    })

        if not comparisons:
            continue
        worst = max(comparisons, key=lambda c: c["diff_m"])
        all_ok = all(c["within_tolerance"] for c in comparisons)
        results.append(GateResult(
            gate="repeatability", scope=f"{tier}:{room_id}", tier=tier,
            status=PASS if all_ok else FAIL,
            metric=f"worst {worst['diff_m'] * 100:.1f} cm on {worst['wall_id']}, {len(comparisons)} wall pair(s)",
            threshold=f"<= {REPEATABILITY_ABS_M * 100:.0f} cm or {_pct(REPEATABILITY_REL)} per wall",
            value=round(worst["diff_m"], 4),
            detail={"captures": [m.capture_id for m in members], "comparisons": comparisons},
        ))
    return results


def gate_ceiling_spread(plans: Sequence[LoadedPlan]) -> List[GateResult]:
    """Spread of ceiling height across repeat captures of one room.

    Separate from the ceiling-height row on purpose: this is the gate that tells
    repeatable-but-biased apart from unrepeatable. A room that is consistently
    3 cm short fails ceiling_height and passes here; a room that wanders fails
    here. The report has to say which one it has.
    """
    groups = _repeat_groups(plans)
    if not groups:
        return [GateResult(
            gate="ceiling_spread", scope="-", tier="-", status=SKIP,
            metric="no repeated captures", threshold=f"<= {CEILING_SPREAD_MAX_M * 100:.0f} cm spread",
            detail={"reason": "benchmark set contains no room captured twice at the same tier"},
        )]

    results: List[GateResult] = []
    for (tier, room_id), members in sorted(groups.items()):
        values = []
        for lp in members:
            room = lp.plan.room_by_id.get(room_id)
            if room is not None:
                values.append({"capture_id": lp.capture_id, "ceiling_m": room.ceiling_height.value})
        if len(values) < 2:
            continue
        heights = [v["ceiling_m"] for v in values]
        spread = max(heights) - min(heights)
        results.append(GateResult(
            gate="ceiling_spread", scope=f"{tier}:{room_id}", tier=tier,
            status=PASS if spread <= CEILING_SPREAD_MAX_M + EPS else FAIL,
            metric=f"spread {spread * 100:.1f} cm across {len(values)} captures",
            threshold=f"<= {CEILING_SPREAD_MAX_M * 100:.0f} cm",
            value=round(spread, 4),
            detail={"values": values},
        ))
    return results


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


@dataclass
class Report:
    gates: List[GateResult]
    plans: List[Dict[str, Any]]
    ground_truth: Dict[str, Any]
    generated_at: str

    @property
    def counts(self) -> Dict[str, int]:
        return {
            "pass": sum(1 for g in self.gates if g.status == PASS),
            "fail": sum(1 for g in self.gates if g.status == FAIL),
            "skip": sum(1 for g in self.gates if g.status == SKIP),
        }

    @property
    def failed(self) -> bool:
        return self.counts["fail"] > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scorer_version": SCORER_VERSION,
            "generated_at": self.generated_at,
            "ground_truth": self.ground_truth,
            "plans": self.plans,
            "summary": self.counts,
            "gates": [g.to_json() for g in self.gates],
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_results(results_dir: Path, ground_truth_csv: Path) -> Report:
    """Score every plan.json under ``results_dir`` against the ground truth."""
    gt = load_ground_truth(ground_truth_csv)
    plans = load_plans(results_dir)

    gates: List[GateResult] = []
    for lp in sorted(plans, key=lambda p: (p.tier.value, p.capture_id)):
        gates.append(gate_wall_lengths(lp, gt))
        gates.append(gate_ceiling_height(lp, gt))
        gates.append(gate_opening_widths(lp, gt))
        gates.append(gate_footprint(lp, gt))
        gates.append(gate_interval_coverage(lp, gt))
    gates.extend(gate_repeatability(plans))
    gates.extend(gate_ceiling_spread(plans))

    return Report(
        gates=gates,
        plans=[
            {
                "capture_id": lp.capture_id,
                "tier": lp.tier.value,
                "path": str(lp.path),
                "sha256": _sha256(lp.path),
                "pipeline_version": lp.plan.pipeline_version,
                "schema_version": lp.plan.schema_version,
                "rooms": [r.id for r in lp.plan.rooms],
            }
            for lp in plans
        ],
        ground_truth={
            "path": str(Path(ground_truth_csv).resolve()),
            "sha256": _sha256(Path(ground_truth_csv)),
            "row_count": len(gt.rows),
            "methods": sorted({r.method for r in gt.rows if r.method}),
        },
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_table(report: Report) -> str:
    headers = ("GATE", "SCOPE", "TIER", "MEASURED", "GATE THRESHOLD", "STATUS")
    rows: List[Tuple[str, ...]] = [
        (g.gate, g.scope, g.tier, g.metric, g.threshold, g.status) for g in report.gates
    ]
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def line(cells: Iterable[str]) -> str:
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)).rstrip()

    out = [line(headers), "  ".join("-" * w for w in widths)]
    out.extend(line(r) for r in rows)

    counts = report.counts
    out.append("")
    out.append(f"{counts['pass']} PASS   {counts['fail']} FAIL   {counts['skip']} SKIP")
    if counts["skip"]:
        out.append("SKIP means no ground truth covered that gate. It is not a pass.")
    return "\n".join(out)


def write_results(report: Report, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / RESULTS_FILENAME
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    return path
