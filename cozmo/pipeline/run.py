"""Single-capture orchestrator.

STUB. There is no vision code in this repo yet. :func:`build_stub_plan` emits a
hardcoded property with plausible numbers so the contract, the CLI and the
scoreboard can be exercised end to end before any algorithm exists. Every stub
plan says so in ``quality.warnings`` -- a plan that looks measured but is not
would poison the benchmark it is meant to validate.

What is *not* stubbed, and will not change when the real stages land:

* seeding happens before anything else and is recorded;
* the input directory is hashed, so a reported number can always be tied to the
  bytes that produced it;
* the git commit, the exact command and the pipeline version are written next to
  every plan.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import PIPELINE_VERSION, SCHEMA_VERSION, __version__
from ..io.capture import CaptureBundle, load_capture
from ..schema import (
    Adjacency,
    ConcealedFlag,
    DamageClass,
    DamageRegion,
    DriftCorrection,
    DriftMethod,
    Measurement,
    Opening,
    OpeningType,
    Plan,
    Point2D,
    Pose2D,
    PropertyTotals,
    QualityReport,
    Room,
    ScaleInfo,
    ScaleSource,
    ScopeItem,
    Surface,
    SurfaceKind,
    Tier,
    Unit,
    Wall,
)
from ..seed import DEFAULT_SEED, set_global_seeds

PLAN_FILENAME = "plan.json"
MANIFEST_FILENAME = "run_manifest.json"

STUB_WARNING = (
    "STUB PIPELINE: this plan is hardcoded geometry, not a reconstruction of the "
    "input capture. Do not treat any dimension here as measured."
)


# --------------------------------------------------------------------------
# Tier error model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TierProfile:
    """Interval half-widths per tier, and the jitter the stub applies.

    These are the numbers the gates are argued against, so they live in one
    place: photo gates are +-8%, video +-3%, LiDAR centimetric. Intervals sit
    just inside the gate, which is the honest position for a pipeline that
    expects to pass but not by much.
    """

    scale_source: ScaleSource
    wall_rel: float          # relative half-width on wall lengths
    wall_floor_m: float      # absolute floor on that half-width
    ceiling_m: float         # absolute half-width on ceiling height
    opening_m: float         # absolute half-width on opening widths
    area_rel: float          # relative half-width on areas
    jitter_m: float          # deterministic per-quantity perturbation of the stub
    confidence: float


TIER_PROFILES: Dict[Tier, TierProfile] = {
    Tier.LIDAR: TierProfile(
        scale_source=ScaleSource.LIDAR_DEPTH,
        wall_rel=0.004, wall_floor_m=0.012, ceiling_m=0.012, opening_m=0.015,
        area_rel=0.015, jitter_m=0.006, confidence=0.88,
    ),
    Tier.VIDEO: TierProfile(
        scale_source=ScaleSource.ARKIT_VIO,
        wall_rel=0.024, wall_floor_m=0.020, ceiling_m=0.030, opening_m=0.030,
        area_rel=0.045, jitter_m=0.020, confidence=0.72,
    ),
    Tier.PHOTO: TierProfile(
        scale_source=ScaleSource.STRUCTURAL_PRIOR,
        wall_rel=0.065, wall_floor_m=0.050, ceiling_m=0.070, opening_m=0.055,
        area_rel=0.120, jitter_m=0.060, confidence=0.51,
    ),
}


# --------------------------------------------------------------------------
# The hardcoded property
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StubRoom:
    """An axis-aligned rectangle placed in the property frame."""

    room_id: str
    name: str
    origin: Tuple[float, float]
    width: float   # extent along x, metres
    depth: float   # extent along y, metres


@dataclass(frozen=True)
class StubOpening:
    opening_id: str
    room_id: str
    wall_index: int
    type: OpeningType
    width: float
    height: float
    offset: float
    connects_room_id: Optional[str] = None


# Four rooms and a connector: enough to exercise stitching, adjacency and the
# multi-room drift story.
STUB_ROOMS: Sequence[StubRoom] = (
    StubRoom("living_room", "Living Room", (0.00, 0.00), 4.20, 3.60),
    StubRoom("hallway", "Hallway", (4.20, 1.00), 3.10, 1.20),
    StubRoom("kitchen", "Kitchen", (4.20, 2.20), 3.00, 2.80),
    StubRoom("bedroom", "Bedroom", (7.30, 0.30), 3.40, 3.20),
)

STUB_CEILING_HEIGHT = 2.44

# wall_index: 0 south (y=min), 1 east (x=max), 2 north (y=max), 3 west (x=min)
STUB_OPENINGS: Sequence[StubOpening] = (
    StubOpening("op_lr_door_hall", "living_room", 1, OpeningType.DOOR, 0.815, 2.030, 1.20, "hallway"),
    StubOpening("op_lr_window_s", "living_room", 0, OpeningType.WINDOW, 1.220, 1.400, 1.50),
    StubOpening("op_hall_door_bed", "hallway", 1, OpeningType.DOOR, 0.760, 2.030, 0.25, "bedroom"),
    StubOpening("op_hall_pass_kitchen", "hallway", 2, OpeningType.PASS_THROUGH, 0.910, 2.100, 0.90, "kitchen"),
    StubOpening("op_kitchen_window_n", "kitchen", 2, OpeningType.WINDOW, 0.900, 1.100, 1.05),
    StubOpening("op_bed_window_e", "bedroom", 1, OpeningType.WINDOW, 1.100, 1.400, 1.30),
)

STUB_ADJACENCIES: Sequence[Tuple[str, str, str]] = (
    ("living_room", "hallway", "op_lr_door_hall"),
    ("hallway", "bedroom", "op_hall_door_bed"),
    ("hallway", "kitchen", "op_hall_pass_kitchen"),
)


def _jitter(profile: TierProfile, *key_parts: object) -> float:
    """Deterministic pseudo-random offset in metres, keyed by identity.

    Not drawn from an RNG on purpose: the value depends only on what is being
    measured and the tier, so two runs over the same capture -- and two captures
    of the same room -- reproduce exactly. That is what the repeatability gate
    is asking about, and a stub that jittered per-call would fail it for reasons
    that tell us nothing.
    """
    key = "|".join(str(part) for part in key_parts).encode()
    digest = hashlib.sha256(key).digest()
    unit = int.from_bytes(digest[:8], "big") / float(1 << 64)  # [0, 1)
    return (unit * 2.0 - 1.0) * profile.jitter_m


def _wall_measurement(true_length: float, profile: TierProfile, *key: object) -> Measurement:
    value = true_length + _jitter(profile, *key)
    half = max(abs(value) * profile.wall_rel, profile.wall_floor_m)
    return Measurement.symmetric(round(value, 4), round(half, 4))


def _rect_corners(room: StubRoom) -> List[Tuple[float, float]]:
    x0, y0 = room.origin
    x1, y1 = x0 + room.width, y0 + room.depth
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _build_room(room: StubRoom, tier: Tier, profile: TierProfile) -> Room:
    corners = _rect_corners(room)
    openings_here = [o for o in STUB_OPENINGS if o.room_id == room.room_id]

    walls: List[Wall] = []
    surfaces: List[Surface] = []
    ceiling = Measurement.symmetric(
        round(STUB_CEILING_HEIGHT + _jitter(profile, tier.value, room.room_id, "ceiling"), 4),
        profile.ceiling_m,
    )

    for index in range(4):
        (ax, ay), (bx, by) = corners[index], corners[(index + 1) % 4]
        true_length = abs(bx - ax) + abs(by - ay)  # axis-aligned
        wall_id = f"{room.room_id}_w{index}"
        length = _wall_measurement(true_length, profile, tier.value, wall_id)
        walls.append(
            Wall(
                id=wall_id,
                start=Point2D(x=ax, y=ay),
                end=Point2D(x=bx, y=by),
                length=length,
                height=ceiling,
                opening_ids=[o.opening_id for o in openings_here if o.wall_index == index],
            )
        )
        surfaces.append(
            Surface(
                id=f"{wall_id}_surface",
                room_id=room.room_id,
                kind=SurfaceKind.WALL,
                wall_id=wall_id,
                area=Measurement.relative(
                    round(length.value * ceiling.value, 4), profile.area_rel, Unit.SQUARE_METERS
                ),
            )
        )

    floor_area_true = room.width * room.depth
    floor_area = Measurement.relative(
        round(floor_area_true * (1.0 + _jitter(profile, tier.value, room.room_id, "area") / 4.0), 4),
        profile.area_rel,
        Unit.SQUARE_METERS,
    )
    for kind in (SurfaceKind.FLOOR, SurfaceKind.CEILING):
        surfaces.append(
            Surface(
                id=f"{room.room_id}_{kind.value}",
                room_id=room.room_id,
                kind=kind,
                area=floor_area.model_copy(deep=True),
            )
        )

    openings = [
        Opening(
            id=o.opening_id,
            wall_id=f"{room.room_id}_w{o.wall_index}",
            type=o.type,
            width=Measurement.symmetric(
                round(o.width + _jitter(profile, tier.value, o.opening_id, "w"), 4), profile.opening_m
            ),
            height=Measurement.symmetric(
                round(o.height + _jitter(profile, tier.value, o.opening_id, "h"), 4), profile.opening_m
            ),
            offset_along_wall=Measurement.symmetric(o.offset, max(profile.opening_m, 0.02)),
            connects_room_id=o.connects_room_id,
            detection_confidence=round(profile.confidence, 3),
        )
        for o in openings_here
    ]

    perimeter_value = round(sum(w.length.value for w in walls), 4)
    return Room(
        id=room.room_id,
        name=room.name,
        pose=Pose2D(x=room.origin[0], y=room.origin[1], theta_rad=0.0),
        walls=walls,
        openings=openings,
        surfaces=surfaces,
        ceiling_height=ceiling,
        floor_area=floor_area,
        perimeter=Measurement.relative(perimeter_value, profile.wall_rel, Unit.METERS),
        source_frame_count=None,
    )


def _build_damage(profile: TierProfile) -> Tuple[List[DamageRegion], List[ConcealedFlag], List[ScopeItem]]:
    """One furnished room, two damage classes -- the benchmark composition."""
    water = DamageRegion(
        id="dmg_lr_water_01",
        room_id="living_room",
        surface_id="living_room_w2_surface",
        damage_class=DamageClass.WATER,
        polygon=[Point2D(x=1.10, y=0.00), Point2D(x=2.05, y=0.00),
                 Point2D(x=2.05, y=0.62), Point2D(x=1.10, y=0.55)],
        area=Measurement.relative(0.56, profile.area_rel, Unit.SQUARE_METERS),
        max_extent=Measurement.symmetric(0.95, max(profile.opening_m, 0.03)),
        severity=0.62,
        confidence=round(profile.confidence, 3),
        evidence_frames=["frame_000142", "frame_000151"],
    )
    impact = DamageRegion(
        id="dmg_lr_impact_01",
        room_id="living_room",
        surface_id="living_room_w1_surface",
        damage_class=DamageClass.IMPACT,
        polygon=[Point2D(x=2.40, y=0.90), Point2D(x=2.61, y=0.90),
                 Point2D(x=2.61, y=1.08), Point2D(x=2.40, y=1.08)],
        area=Measurement.relative(0.038, profile.area_rel, Unit.SQUARE_METERS),
        max_extent=Measurement.symmetric(0.21, max(profile.opening_m, 0.02)),
        severity=0.40,
        confidence=round(profile.confidence * 0.95, 3),
        evidence_frames=["frame_000206"],
    )

    flag = ConcealedFlag(
        id="cf_lr_cavity_01",
        room_id="living_room",
        surface_id="living_room_w2_surface",
        rule_id="CD-WATER-01",
        rule_text=(
            "Water staining reaching within 0.15 m of the wall base on a wall shared with a wet "
            "area implies moisture in the cavity behind it; the visible region understates extent."
        ),
        triggered_by_damage_ids=[water.id],
        probability=0.68,
        recommended_action="Moisture-meter the lower 0.6 m of the shared wall; open cavity if >18% WME.",
        inspection_priority=2,
    )

    scope = [
        ScopeItem(
            id="scope_001", room_id="living_room", surface_id="living_room_w2_surface",
            damage_region_ids=[water.id], code="DRY-RMV-2",
            description="Remove and dispose water-damaged drywall, lower course",
            quantity=Measurement.relative(1.85, profile.area_rel, Unit.SQUARE_METERS),
        ),
        ScopeItem(
            id="scope_002", room_id="living_room", surface_id="living_room_w1_surface",
            damage_region_ids=[impact.id], code="DRY-PATCH-1",
            description="Patch impact penetration, tape, sand and prime",
            quantity=Measurement(value=1.0, ci_95=(1.0, 1.0), unit=Unit.EACH),
        ),
        ScopeItem(
            id="scope_003", room_id="living_room", surface_id="living_room_w2_surface",
            damage_region_ids=[water.id], code="PNT-WALL-2",
            description="Seal and repaint affected wall, two coats",
            quantity=Measurement.relative(8.78, profile.area_rel, Unit.SQUARE_METERS),
            notes="Full wall repaint; spot painting will flash against the existing finish.",
        ),
    ]
    return [water, impact], [flag], scope


def build_stub_plan(
    bundle: CaptureBundle,
    drift_correction: bool = True,
    generated_at: Optional[datetime] = None,
) -> Plan:
    """The hardcoded plan. Replace this body, keep the signature."""
    tier = bundle.tier
    profile = TIER_PROFILES[tier]

    rooms = [_build_room(spec, tier, profile) for spec in STUB_ROOMS]
    damage, flags, scope = _build_damage(profile)

    total_area = round(sum(r.floor_area.value for r in rooms), 4)
    # Uncorrected poses inflate the stitched footprint; the ablation reports both.
    uncorrected = round(total_area * 1.037, 4)
    footprint_value = total_area if drift_correction else uncorrected
    ablation_value = uncorrected if drift_correction else total_area

    drift = DriftCorrection(
        enabled=drift_correction,
        method=DriftMethod.POSE_GRAPH if drift_correction else DriftMethod.NONE_POSES_AS_IS,
        loop_closures=2 if drift_correction else 0,
        residual_closure_error=(
            Measurement.symmetric(0.031, 0.010) if drift_correction else None
        ),
        ablation_footprint_area=Measurement.relative(
            ablation_value, profile.area_rel, Unit.SQUARE_METERS
        ),
        notes=(
            "Pose graph with plane-anchored constraints on the connector; two loop closures."
            if drift_correction
            else "Poses used as-is. This is the ablation arm and fails the drift-accountability gate."
        ),
    )

    warnings = [STUB_WARNING] + list(bundle.warnings)
    degradations: List[str] = []
    if tier is Tier.PHOTO:
        degradations.append("no metric sensor; scale from structural prior (door height)")

    plan = Plan(
        schema_version=SCHEMA_VERSION,
        capture_id=bundle.capture_id,
        tier=tier,
        pipeline_version=PIPELINE_VERSION,
        generated_at=generated_at or datetime.now(timezone.utc),
        scale=ScaleInfo(
            source=profile.scale_source,
            scale_factor=Measurement(
                value=1.0,
                ci_95=(1.0 - profile.wall_rel, 1.0 + profile.wall_rel),
                unit=Unit.RATIO,
            ),
            reference_description=(
                "interior door leaf height prior, 2.032 m +- 0.02"
                if tier is Tier.PHOTO
                else "sensor-metric; no external reference used"
            ),
        ),
        drift_correction=drift,
        property_totals=PropertyTotals(
            room_count=len(rooms),
            total_floor_area=Measurement.relative(total_area, profile.area_rel, Unit.SQUARE_METERS),
            footprint_area=Measurement.relative(footprint_value, profile.area_rel, Unit.SQUARE_METERS),
            total_wall_area=Measurement.relative(
                round(sum(r.perimeter.value * r.ceiling_height.value for r in rooms), 4),
                profile.area_rel,
                Unit.SQUARE_METERS,
            ),
            bounding_box_m=(10.70, 5.00),
        ),
        rooms=rooms,
        adjacencies=[
            Adjacency(room_a_id=a, room_b_id=b, via_opening_id=op, confidence=round(profile.confidence, 3))
            for a, b, op in STUB_ADJACENCIES
        ],
        damage=damage,
        concealed_flags=flags,
        scope=scope,
        quality=QualityReport(
            overall_confidence=profile.confidence,
            interval_method=(
                "Stub: fixed per-tier half-widths from TIER_PROFILES. Replace with a propagated "
                "error budget (sensor noise + pose covariance + plane-fit residual) when the "
                "reconstruction stages land."
            ),
            calibration_note="Intervals are asserted, not yet calibrated against ground truth.",
            degradations=degradations,
            warnings=warnings,
            coverage={"rooms_declared_vs_reconstructed": 1.0},
        ),
    )
    return plan


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_provenance() -> Dict[str, Any]:
    """Which commit produced this run. ``null`` when run outside a checkout --
    stated rather than faked, because a benchmark number with an unknown commit
    behind it is not reproducible."""
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": _git("describe", "--always", "--dirty"),
        "dirty": bool(status) if status is not None else None,
    }


def hash_directory(root: Path) -> Dict[str, Any]:
    """SHA-256 over the input directory: relative paths and file bytes.

    Paths are included so that renaming a file changes the hash, and the walk is
    sorted so the digest does not depend on filesystem ordering.
    """
    root = Path(root)
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0

    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name == ".DS_Store":
            continue
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode())
        digest.update(b"\0")
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
                total_bytes += len(chunk)
        file_count += 1

    return {
        "path": str(root.resolve()),
        "sha256": digest.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deterministic_now() -> Optional[datetime]:
    """Honour SOURCE_DATE_EPOCH so plan.json is byte-identical across reruns.

    The reproduction bundle has to regenerate every reported number; a wall-clock
    timestamp in the output makes a byte-diff useless for checking that.
    """
    raw = os.environ.get("SOURCE_DATE_EPOCH")
    if not raw:
        return None
    try:
        return datetime.fromtimestamp(int(raw), tz=timezone.utc)
    except (ValueError, OverflowError, OSError):
        return None


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


@dataclass
class RunResult:
    plan: Plan
    plan_path: Path
    manifest_path: Path
    manifest: Dict[str, Any]


def run_capture(
    input_dir: Path,
    out_dir: Path,
    drift_correction: bool = True,
    seed: int = DEFAULT_SEED,
    command: Optional[Sequence[str]] = None,
) -> RunResult:
    """Run one capture: load, reconstruct (stubbed), write plan + manifest."""
    started = time.time()
    seed_record = set_global_seeds(seed)

    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_capture(input_dir)
    input_hash = hash_directory(input_dir)
    plan = build_stub_plan(bundle, drift_correction=drift_correction, generated_at=_deterministic_now())

    plan_path = out_dir / PLAN_FILENAME
    plan_path.write_text(plan.to_json() + "\n")

    manifest: Dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "package_version": __version__,
        "stub": True,
        "git": git_provenance(),
        "command": " ".join(command) if command else " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "argv": list(command) if command else list(sys.argv),
        "seed": seed_record,
        "options": {"drift_correction": drift_correction},
        "input": input_hash,
        "capture": bundle.summary(),
        "outputs": {PLAN_FILENAME: sha256_file(plan_path)},
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "source_date_epoch": os.environ.get("SOURCE_DATE_EPOCH"),
        },
        "started_at": datetime.fromtimestamp(started, tz=timezone.utc).isoformat(),
        "duration_s": round(time.time() - started, 4),
    }
    manifest_path = out_dir / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")

    return RunResult(plan=plan, plan_path=plan_path, manifest_path=manifest_path, manifest=manifest)
