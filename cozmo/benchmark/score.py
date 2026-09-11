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

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..schema import DriftMethod, Measurement, Plan, Tier

SCORER_VERSION = "1.0.0"
RESULTS_FILENAME = "results.json"

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

# Binary floating point puts 2.450 - 2.435 at 1.5000000000000013 cm. Without
# this slack a measurement exactly on a gate would fail it, which is not what
# "<= 1.5 cm" means to anyone holding a laser measurer.
EPS = 1e-9

AREA_DIMENSIONS = {"floor_area", "footprint_area"}
VALID_ELEMENTS = {"wall", "opening", "room", "property", "adjacency"}


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
# Two rooms occupying the same ground: the brief calls this an automatic
# failure, and this is the slack given only to polygon-simplification noise.
ROOM_OVERLAP_TOLERANCE_M2 = 0.02

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
    def space_id(self) -> str:
        """Declared physical-space identity from the run manifest.

        ``plan.json`` deliberately stays focused on reconstruction output, so
        capture identity lives beside it in ``run_manifest.json``. Falling
        back to ``capture_id`` keeps hand-authored fixture plans and older
        reports deterministic singletons.
        """
        manifest = self.path.with_name("run_manifest.json")
        try:
            raw = json.loads(manifest.read_text())
            return str(raw.get("capture", {}).get("space_id") or self.capture_id)
        except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError):
            # Hand-authored unit fixtures have no run manifest. The grouping
            # helper then falls back to their room id, while real benchmark
            # runs always carry the explicit capture identity.
            return ""

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


def _has_plan_truth(lp: LoadedPlan, gt: GroundTruth) -> bool:
    """Return whether non-property truth actually belongs to this plan.

    A property-level row with ``element_id=property`` is valid for a plan only
    when the same CSV also identifies at least one wall, opening, or room in
    that plan. Without this guard, one synthetic fixture's footprint becomes a
    false global truth value for every unrelated real capture in the benchmark.
    """
    wall_ids = set(lp.walls())
    opening_ids = set(lp.openings())
    room_ids = set(lp.plan.room_by_id)
    return any(
        row.element_id in wall_ids
        or row.element_id in opening_ids
        or row.element_id in room_ids
        for row in gt.rows
        if row.element != "property"
    )


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


def gate_footprint(
    lp: LoadedPlan, gt: GroundTruth, *, allow_global_property: bool = True,
) -> GateResult:
    tol = TOLERANCES[lp.tier]
    truth_row = gt.lookup(lp.capture_id, "footprint_area")
    if truth_row is None and (allow_global_property or _has_plan_truth(lp, gt)):
        truth_row = gt.lookup("property", "footprint_area")
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


def _room_polygon(room: Any) -> Optional["Polygon"]:
    """A room's own polygon, from its walls' already-placed coordinates."""
    from shapely.geometry import Polygon

    points = [(w.start.x, w.start.y) for w in room.walls]
    if len(points) < 3:
        return None
    polygon = Polygon(points)
    return polygon if polygon.is_valid else polygon.buffer(0)


def gate_room_overlap(lp: LoadedPlan) -> GateResult:
    """Two rooms occupying the same ground is a stitching failure the brief
    calls out by name: overlap area must be zero. No ground truth needed --
    this is checkable from the plan alone, the same way the schema's own
    referential-integrity checks are.
    """
    rooms = lp.plan.rooms
    if len(rooms) < 2:
        return GateResult(
            gate="room_overlap", scope=lp.capture_id, tier=lp.tier.value, status=SKIP,
            metric="single room", threshold="-",
            detail={"reason": "overlap is only meaningful with more than one room"},
        )

    polygons = {room.id: _room_polygon(room) for room in rooms}
    pairs = []
    worst = 0.0
    for i, room_a in enumerate(rooms):
        for room_b in rooms[i + 1:]:
            poly_a, poly_b = polygons[room_a.id], polygons[room_b.id]
            if poly_a is None or poly_b is None:
                continue
            overlap = float(poly_a.intersection(poly_b).area)
            if overlap > ROOM_OVERLAP_TOLERANCE_M2:
                pairs.append({"room_a": room_a.id, "room_b": room_b.id, "overlap_m2": round(overlap, 4)})
            worst = max(worst, overlap)

    return GateResult(
        gate="room_overlap", scope=lp.capture_id, tier=lp.tier.value,
        status=PASS if not pairs else FAIL,
        metric=(
            f"worst pair overlaps {worst:.3f} m2" if pairs else
            f"no overlap above {ROOM_OVERLAP_TOLERANCE_M2} m2 across {len(rooms)} room(s)"
        ),
        threshold=f"<= {ROOM_OVERLAP_TOLERANCE_M2} m2 for every room pair",
        value=round(worst, 4),
        detail={"overlapping_pairs": pairs, "room_count": len(rooms)},
    )


def gate_drift_accountability(lp: LoadedPlan) -> GateResult:
    """Check that accumulated drift is named and has an on/off ablation.

    The assignment's automatic-fail case is an output that silently trusts raw
    poses. For multi-room photo/video plans the same accountability applies to
    the stitch correction; a method name without a second footprint is not an
    ablation.
    """
    drift = lp.plan.drift_correction
    requires_ablation = lp.tier is Tier.LIDAR or len(lp.plan.rooms) > 1
    if not requires_ablation:
        return GateResult(
            gate="drift_accountability", scope=lp.capture_id, tier=lp.tier.value,
            status=SKIP, metric="single-room route; no accumulated multi-room drift",
            threshold="named correction + on/off footprint ablation",
            detail={"reason": "gate applies to LiDAR and multi-room stitching"},
        )

    method_ok = drift.enabled and drift.method is not DriftMethod.NONE_POSES_AS_IS
    ablation = drift.ablation_footprint_area
    ablation_ok = ablation is not None
    on_area = lp.plan.property_totals.footprint_area.value
    changed = bool(ablation_ok and abs(ablation.value - on_area) > 1e-6)
    passed = method_ok and ablation_ok
    return GateResult(
        gate="drift_accountability", scope=lp.capture_id, tier=lp.tier.value,
        status=PASS if passed else FAIL,
        metric=(
            f"method={drift.method.value}; footprint on={on_area:.2f} m2, "
            f"off={ablation.value:.2f} m2" if ablation_ok else
            f"method={drift.method.value}; correction ablation missing"
        ),
        threshold="named correction + on/off footprint ablation; poses_as_is fails",
        value=round(abs(ablation.value - on_area), 5) if ablation_ok else None,
        detail={
            "enabled": drift.enabled,
            "method": drift.method.value,
            "ablation_present": ablation_ok,
            "footprint_changed": changed,
            "ablation_footprint_area_m2": ablation.value if ablation_ok else None,
        },
    )


def gate_adjacency_correctness(lp: LoadedPlan, gt: GroundTruth) -> GateResult:
    """Ground truth rows with ``element=adjacency`` state which room pairs
    should (``value_m=1``) or should not (``value_m=0``) be connected;
    ``element_id`` is ``room_a_id:room_b_id``. A missed real connection and a
    phantom one both count as a miss, the same rule the opening-detection gate
    already uses, for the same reason: a stitcher that only reports the edges
    it is sure of must not be able to buy accuracy by staying silent.
    """
    truth_rows = gt.by_element("adjacency")
    if not truth_rows:
        return GateResult(
            gate="adjacency_correctness", scope=lp.capture_id, tier=lp.tier.value, status=SKIP,
            metric="no adjacency ground truth", threshold="-",
            detail={"reason": "ground truth contains no element=adjacency rows"},
        )

    def pair_key(a: str, b: str) -> Tuple[str, str]:
        return tuple(sorted((a, b)))

    predicted_pairs = {
        pair_key(adj.room_a_id, adj.room_b_id) for adj in lp.plan.adjacencies
    }

    correct = 0
    mistakes: List[Dict[str, Any]] = []
    for row in truth_rows:
        if ":" not in row.element_id:
            continue
        a, b = row.element_id.split(":", 1)
        expected_connected = row.value_m >= 0.5
        actual_connected = pair_key(a, b) in predicted_pairs
        if expected_connected == actual_connected:
            correct += 1
        else:
            mistakes.append({
                "room_a": a, "room_b": b, "expected_connected": expected_connected,
                "actual_connected": actual_connected,
            })

    total = correct + len(mistakes)
    fraction = correct / total if total else 1.0
    return GateResult(
        gate="adjacency_correctness", scope=lp.capture_id, tier=lp.tier.value,
        status=PASS if not mistakes else FAIL,
        metric=f"{correct}/{total} room pair(s) correct" + (
            f"; wrong: {[(m['room_a'], m['room_b']) for m in mistakes]}" if mistakes else ""
        ),
        threshold="every declared room pair's connectivity matches ground truth",
        value=round(fraction, 4),
        detail={"mistakes": mistakes, "predicted_pairs": sorted(predicted_pairs)},
    )


def covered_measurements(lp: LoadedPlan, gt: GroundTruth) -> List[Dict[str, Any]]:
    """Every measurement in `lp` with a matching ground-truth row: predicted
    value, truth, half-width, whether the interval covers it, and which
    quantity kind it is (wall_length | opening_width | ceiling_height |
    floor_area | footprint_area).

    This is the one definition of "which measurements get checked against
    truth" -- used by :func:`gate_interval_coverage` and
    :func:`gate_interval_coverage_by_kind` here, and reused as-is by
    ``cozmo.calibrate`` to fit calibration factors against exactly the same
    pairing the benchmark scores against.
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
    row = gt.lookup("property", "footprint_area") if _has_plan_truth(lp, gt) else None
    if row:
        add("footprint_area", "property", lp.plan.property_totals.footprint_area, row.value_m)
    return checks


def gate_interval_coverage(lp: LoadedPlan, gt: GroundTruth) -> GateResult:
    """Calibration: do the 95% intervals actually contain the truth?

    Scored at every tier. Wide intervals pass this row cheaply, which is why the
    mean half-width is reported alongside -- a pipeline that widens its way to
    coverage is visible here rather than hidden. See
    :func:`gate_interval_coverage_by_kind` for the same question broken out by
    quantity kind across the whole benchmark set.
    """
    checks = covered_measurements(lp, gt)

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


def _repeat_groups(plans: Sequence[LoadedPlan]) -> Dict[Tuple[str, str, str], List[LoadedPlan]]:
    """Group by declared physical space, tier, and reconstructed room id.

    Room names are not identities: three unrelated captures in the benchmark
    all called their room ``living_room``. Pairing those by name produced a
    spectacular but meaningless repeatability failure.
    """
    groups: Dict[Tuple[str, str, str], List[LoadedPlan]] = {}
    for lp in plans:
        for room in lp.plan.rooms:
            groups.setdefault((lp.tier.value, lp.space_id or room.id, room.id), []).append(lp)
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
    for (tier, space_id, room_id), members in sorted(groups.items()):
        comparisons: List[Dict[str, Any]] = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                walls_a, walls_b = a.walls(), b.walls()
                ids_a = [wall_id for wall_id, (rid, _m) in sorted(walls_a.items()) if rid == room_id]
                ids_b = [wall_id for wall_id, (rid, _m) in sorted(walls_b.items()) if rid == room_id]
                if not ids_a or not ids_b:
                    continue

                # Wall ids are traversal-order labels and are not stable when
                # a noisy scan splits or merges a boundary. Pair by minimum
                # absolute length cost, then count unmatched topology as a
                # failure instead of comparing unrelated positional ids.
                cost = np.array([
                    [abs(walls_a[wa][1].value - walls_b[wb][1].value) for wb in ids_b]
                    for wa in ids_a
                ])
                rows, cols = linear_sum_assignment(cost)
                paired_a = set(rows.tolist())
                paired_b = set(cols.tolist())
                for row, col in zip(rows, cols):
                    wall_id, wall_b_id = ids_a[row], ids_b[col]
                    m_a = walls_a[wall_id][1]
                    m_b = walls_b[wall_b_id][1]
                    diff = abs(m_a.value - m_b.value)
                    reference = max(abs(m_a.value), abs(m_b.value)) or 1.0
                    comparisons.append({
                        "wall_id": wall_id,
                        "wall_id_b": wall_b_id,
                        "capture_a": a.capture_id, "capture_b": b.capture_id,
                        "a_m": m_a.value, "b_m": m_b.value,
                        "diff_m": round(diff, 4),
                        "unmatched": False,
                        "within_tolerance": diff <= max(
                            REPEATABILITY_ABS_M, REPEATABILITY_REL * reference
                        ) + EPS,
                    })

                for row, wall_id in enumerate(ids_a):
                    if row in paired_a:
                        continue
                    m_a = walls_a[wall_id][1]
                    comparisons.append({
                        "wall_id": wall_id, "wall_id_b": None,
                        "capture_a": a.capture_id, "capture_b": b.capture_id,
                        "a_m": m_a.value, "b_m": None,
                        "diff_m": round(abs(m_a.value), 4), "unmatched": True,
                        "within_tolerance": False,
                    })
                for col, wall_id in enumerate(ids_b):
                    if col in paired_b:
                        continue
                    m_b = walls_b[wall_id][1]
                    comparisons.append({
                        "wall_id": None, "wall_id_b": wall_id,
                        "capture_a": a.capture_id, "capture_b": b.capture_id,
                        "a_m": None, "b_m": m_b.value,
                        "diff_m": round(abs(m_b.value), 4), "unmatched": True,
                        "within_tolerance": False,
                    })

        if not comparisons:
            continue
        worst = max(comparisons, key=lambda c: c["diff_m"])
        all_ok = all(c["within_tolerance"] for c in comparisons)
        unmatched = sum(1 for c in comparisons if c.get("unmatched"))
        results.append(GateResult(
            gate="repeatability", scope=f"{tier}:{space_id}:{room_id}", tier=tier,
            status=PASS if all_ok else FAIL,
            metric=(
                f"worst {worst['diff_m'] * 100:.1f} cm on "
                f"{worst.get('wall_id') or worst.get('wall_id_b')}, "
                f"{len(comparisons) - unmatched} matched pair(s), {unmatched} unmatched"
            ),
            threshold=f"<= {REPEATABILITY_ABS_M * 100:.0f} cm or {_pct(REPEATABILITY_REL)} per wall",
            value=round(worst["diff_m"], 4),
            detail={
                "captures": [m.capture_id for m in members],
                "comparisons": comparisons,
                "unmatched_wall_count": unmatched,
                "matching": "minimum absolute wall-length cost; unmatched topology fails",
            },
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
    for (tier, space_id, room_id), members in sorted(groups.items()):
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
            gate="ceiling_spread", scope=f"{tier}:{space_id}:{room_id}", tier=tier,
            status=PASS if spread <= CEILING_SPREAD_MAX_M + EPS else FAIL,
            metric=f"spread {spread * 100:.1f} cm across {len(values)} captures",
            threshold=f"<= {CEILING_SPREAD_MAX_M * 100:.0f} cm",
            value=round(spread, 4),
            detail={"values": values},
        ))
    return results


def gate_interval_coverage_by_kind(plans: Sequence[LoadedPlan], gt: GroundTruth) -> List[GateResult]:
    """Achieved 95%-interval coverage, broken out by tier and quantity kind,
    across every plan in the benchmark set.

    ``gate_interval_coverage`` answers "does this one capture's intervals hold
    up"; this answers "which kind of measurement, at which tier, is actually
    calibrated" -- the number ``cozmo calibrate``'s own report is fit against,
    surfaced here so a benchmark run shows it without needing calibrate.py.
    """
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for lp in plans:
        for check in covered_measurements(lp, gt):
            groups.setdefault((lp.tier.value, check["kind"]), []).append(check)

    if not groups:
        return [GateResult(
            gate="interval_coverage_by_kind", scope="-", tier="-", status=SKIP,
            metric="nothing to check", threshold="-",
            detail={"reason": "no measurement in any plan has matching ground truth"},
        )]

    results: List[GateResult] = []
    for (tier, kind), checks in sorted(groups.items()):
        covered = sum(1 for c in checks if c["covered"])
        fraction = covered / len(checks)
        mean_half_width = sum(c["half_width"] for c in checks) / len(checks)
        results.append(GateResult(
            gate="interval_coverage_by_kind", scope=f"{tier}:{kind}", tier=tier,
            status=PASS if fraction >= INTERVAL_COVERAGE_MIN - EPS else FAIL,
            metric=f"{covered}/{len(checks)} covered ({_pct(fraction)}), mean +-{mean_half_width:.4f}",
            threshold=f">= {_pct(INTERVAL_COVERAGE_MIN)} of 95% intervals ({kind})",
            value=round(fraction, 4),
            detail={"n": len(checks), "mean_half_width": round(mean_half_width, 4)},
        ))
    return results


# --------------------------------------------------------------------------
# Device matrix: tier x device class x measured accuracy per gate
# --------------------------------------------------------------------------

RUN_MANIFEST_FILENAME = "run_manifest.json"

# Per-capture gates whose `value` is a meaningful accuracy number for one
# device. room_overlap and adjacency_correctness are structural (multi-room)
# checks, not a per-device accuracy figure, so they are left out of the matrix.
DEVICE_MATRIX_GATES = ("wall_lengths", "ceiling_height", "opening_widths", "footprint", "interval_coverage")

# Whether a bigger `value` is better, per gate -- wall_lengths/opening_widths/
# interval_coverage are pass fractions (bigger is better); ceiling_height's
# value is an absolute error and footprint's is a relative error (smaller is
# better). Used only to pick which end of the range to call "worst".
GATE_HIGHER_IS_BETTER: Dict[str, bool] = {
    "wall_lengths": True, "opening_widths": True, "interval_coverage": True,
    "ceiling_height": False, "footprint": False,
}


@dataclass(frozen=True)
class DeviceMatrixRow:
    tier: str
    device_class: str
    gate: str
    n_captures: int
    pass_rate: Optional[float]
    worst_value: Optional[float]

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


def _device_class(plan_path: Path) -> str:
    """Device class for the capture that produced `plan_path`, read from the
    sibling run_manifest.json -- not asserted, and "unknown" when that
    manifest is not sitting next to the plan being scored."""
    manifest_path = Path(plan_path).parent / RUN_MANIFEST_FILENAME
    if not manifest_path.is_file():
        return "unknown"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return "unknown"
    device = manifest.get("capture", {}).get("device", {})
    model = str(device.get("model") or "unknown")
    return f"{model} (LiDAR)" if device.get("has_lidar") else model


def build_device_matrix(gates: Sequence[GateResult], plans: Sequence[LoadedPlan]) -> List[DeviceMatrixRow]:
    """One row per (tier, device class, gate), aggregated across every
    capture the benchmark scored -- generated from this run's own gate
    results and each capture's own run_manifest.json, not written by hand.
    """
    device_by_capture: Dict[str, str] = {lp.capture_id: _device_class(lp.path) for lp in plans}

    groups: Dict[Tuple[str, str, str], List[GateResult]] = {}
    for g in gates:
        if g.gate not in DEVICE_MATRIX_GATES:
            continue
        device = device_by_capture.get(g.scope)
        if device is None:   # scope is not a capture id for this gate; skip
            continue
        groups.setdefault((g.tier, device, g.gate), []).append(g)

    rows: List[DeviceMatrixRow] = []
    for (tier, device, gate), members in sorted(groups.items()):
        scored = [g for g in members if g.status != SKIP and g.value is not None]
        n = len(scored)
        if n == 0:
            rows.append(DeviceMatrixRow(tier, device, gate, 0, None, None))
            continue
        pass_rate = sum(1 for g in scored if g.status == PASS) / n
        values = [g.value for g in scored]
        worst = min(values) if GATE_HIGHER_IS_BETTER.get(gate, True) else max(values)
        rows.append(DeviceMatrixRow(tier, device, gate, n, round(pass_rate, 4), round(worst, 4)))
    return rows


def render_device_matrix(rows: Sequence[DeviceMatrixRow]) -> str:
    if not rows:
        return "no per-capture gate results to build a device matrix from"
    headers = ("TIER", "DEVICE", "GATE", "N", "PASS RATE", "WORST VALUE")
    table_rows: List[Tuple[str, ...]] = [
        (
            r.tier, r.device_class, r.gate, str(r.n_captures),
            f"{r.pass_rate * 100:.0f}%" if r.pass_rate is not None else "-",
            f"{r.worst_value:.4f}" if r.worst_value is not None else "-",
        )
        for r in rows
    ]
    widths = [max(len(headers[i]), *(len(row[i]) for row in table_rows)) for i in range(len(headers))]

    def line(cells: Iterable[str]) -> str:
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)).rstrip()

    out = [line(headers), "  ".join("-" * w for w in widths)]
    out.extend(line(r) for r in table_rows)
    return "\n".join(out)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


@dataclass
class Report:
    gates: List[GateResult]
    plans: List[Dict[str, Any]]
    ground_truth: Dict[str, Any]
    generated_at: str
    device_matrix_rows: List[DeviceMatrixRow] = field(default_factory=list)

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
            "device_matrix": [r.to_json() for r in self.device_matrix_rows],
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
        gates.append(gate_footprint(
            lp, gt, allow_global_property=(len(plans) == 1 or _has_plan_truth(lp, gt)),
        ))
        gates.append(gate_interval_coverage(lp, gt))
        gates.append(gate_room_overlap(lp))
        gates.append(gate_drift_accountability(lp))
        gates.append(gate_adjacency_correctness(lp, gt))
    gates.extend(gate_repeatability(plans))
    gates.extend(gate_ceiling_spread(plans))
    gates.extend(gate_interval_coverage_by_kind(plans, gt))

    return Report(
        gates=gates,
        device_matrix_rows=build_device_matrix(gates, plans),
        plans=[
            {
                "capture_id": lp.capture_id,
                "space_id": lp.space_id,
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
