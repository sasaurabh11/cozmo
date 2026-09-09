"""Single-capture orchestrator.

Every tier is real. :func:`build_lidar_plan` fuses depth frames, fits the
floor, extracts a wall polygon, estimates the ceiling, detects openings and
renders the plan. :func:`~cozmo.pipeline.photo.build_photo_plan` reconstructs
a room from unposed stills. The video tier (:func:`~cozmo.pipeline.video.build_video_plan`)
samples frames from a walkthrough clip and feeds them to that same photo path
-- no separate video reconstruction pipeline, by design (see cozmo/io/video.py).

What is true of every tier:

* seeding happens before anything else and is recorded;
* the input directory is hashed, so a reported number can always be tied to the
  bytes that produced it;
* the git commit, the exact command and the pipeline version are written next to
  every plan.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import PIPELINE_VERSION, SCHEMA_VERSION, __version__
from ..calibration import (
    CalibrationSet,
    CALIBRATION,
    IDENTITY_CALIBRATION,
    calibration_note_for,
    recalibrate_measurement,
    recalibrate_room,
)
from ..geometry.fuse import DEFAULT_STRIDE, DEFAULT_VOXEL_M, fuse_capture
from ..geometry.layout import extract_layout
from ..geometry.openings import detect_openings, wall_observation_fractions
from ..geometry.planes import estimate_ceiling, fit_floor
from ..geometry.render import render_plan
from ..io.capture import CaptureBundle, load_capture
from ..io.lidar import LidarCapture
from ..io.stray import quaternion_to_rotation
from ..io.video import (
    DEFAULT_BLUR_THRESHOLD as VIDEO_DEFAULT_BLUR_THRESHOLD,
    DEFAULT_MAX_FRAMES as VIDEO_DEFAULT_MAX_FRAMES,
    DEFAULT_STRIDE_FRAMES as VIDEO_DEFAULT_STRIDE_FRAMES,
)
from ..stitch.drift import correct_trajectory
from .photo import DEFAULT_BACKBONE, build_photo_plan
from .video import build_video_plan
from ..semantics.project import surfaces_from_layout
from ..semantics.stage import DEFAULT_FRAME_STRIDE, DEFAULT_MAX_FRAMES, SemanticResult
from ..semantics.worker import surface_to_dict
from ..schema import (
    ConcealedFlag,
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

log = logging.getLogger("cozmo.pipeline.run")

PLAN_FILENAME = "plan.json"
MANIFEST_FILENAME = "run_manifest.json"

# --------------------------------------------------------------------------
# The semantic stage, in its own process
# --------------------------------------------------------------------------

SEMANTICS_TIMEOUT_S = 1800


def run_semantics_subprocess(
    capture_root: Path,
    layout: Any,
    ceiling_height_m: float,
    geometric_openings: Sequence[Any],
    room_id: str,
    weights_dir: Optional[Path] = None,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    max_frames: int = DEFAULT_MAX_FRAMES,
    timeout_s: int = SEMANTICS_TIMEOUT_S,
) -> SemanticResult:
    """Run detection, projection, rules and scope in a child process.

    open3d and torch cannot share an address space on macOS: each bundles its
    own OpenMP runtime, and a process holding both aborts with OMP error 179 or
    deadlocks inside the first inference call. Geometry has already used open3d
    by this point, so the semantic stage gets its own interpreter. The surfaces
    it needs cross as JSON, which also keeps the stage honestly tier-agnostic --
    it cannot reach back into geometry even by accident.
    """
    openings_by_wall: Dict[int, int] = {}
    for detection in geometric_openings:
        openings_by_wall[detection.wall_index] = openings_by_wall.get(detection.wall_index, 0) + 1

    surfaces = surfaces_from_layout(
        layout.walls, layout.frame, room_id, ceiling_height_m,
        layout.floor_area_m2, openings_by_wall,
    )

    request = {
        "capture_root": str(capture_root),
        "room_id": room_id,
        "wall_count": len(layout.walls),
        "ceiling_height_m": ceiling_height_m,
        "surfaces": [surface_to_dict(s) for s in surfaces],
        "geometric_openings": [
            {
                "wall_index": d.wall_index, "kind": d.kind, "width_m": d.width_m,
                "height_m": d.height_m, "offset_along_wall_m": d.offset_along_wall_m,
                "sill_height_m": d.sill_height_m, "confidence": d.confidence,
            }
            for d in geometric_openings
        ],
        "weights_dir": str(weights_dir) if weights_dir else None,
        "frame_stride": frame_stride,
        "max_frames": max_frames,
    }

    with tempfile.TemporaryDirectory(prefix="cozmo-semantics-") as directory:
        request_path = Path(directory) / "request.json"
        response_path = Path(directory) / "response.json"
        request_path.write_text(json.dumps(request))

        command = [sys.executable, "-m", "cozmo.semantics.worker",
                   str(request_path), str(response_path)]
        log.info("running the semantic stage in a child process")
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout_s, check=False
            )
        except subprocess.TimeoutExpired:
            message = f"semantic stage timed out after {timeout_s}s; plan carries geometry only"
            log.warning(message)
            return SemanticResult(warnings=[message], details={"detector": "timeout"})

        if not response_path.is_file():
            message = (
                f"semantic stage produced no response (exit {completed.returncode}); "
                f"plan carries geometry only"
            )
            log.warning("%s: %s", message, (completed.stderr or "").strip()[-500:])
            return SemanticResult(
                warnings=[message],
                details={"detector": "failed", "exit_code": completed.returncode,
                         "stderr_tail": (completed.stderr or "").strip()[-2000:]},
            )

        payload = json.loads(response_path.read_text())

    if not payload.get("ok"):
        message = f"semantic stage failed: {payload.get('error')}; plan carries geometry only"
        log.warning(message)
        return SemanticResult(
            warnings=[message],
            details={"detector": "failed", "error": payload.get("error"),
                     "traceback": payload.get("traceback")},
        )

    return SemanticResult(
        damage=[DamageRegion.model_validate(d) for d in payload["damage"]],
        concealed_flags=[ConcealedFlag.model_validate(f) for f in payload["concealed_flags"]],
        scope=[ScopeItem.model_validate(s) for s in payload["scope"]],
        openings=[_ConsensusOpening(o) for o in payload["openings"]],
        warnings=list(payload.get("warnings", [])),
        details=dict(payload.get("details", {})),
        available=bool(payload.get("available")),
    )


class _ConsensusOpening:
    """A cross-checked opening, rebuilt from the worker's JSON."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.wall_index = int(data["wall_index"])
        self.kind = str(data["kind"])
        self.width_m = float(data["width_m"])
        self.height_m = float(data["height_m"])
        self.offset_along_wall_m = float(data["offset_along_wall_m"])
        self.sill_height_m = float(data.get("sill_height_m", 0.0))
        self.sources = list(data.get("sources", []))
        self.confidence = float(data.get("confidence", 0.0))
        self.note = data.get("note", "")


# --------------------------------------------------------------------------
# LiDAR tier: the real reconstruction
# --------------------------------------------------------------------------

# Interval placeholders. These are asserted, not calibrated -- the calibration
# pass comes later and will replace every one of them with a propagated error
# budget. They are here because a measurement without an interval is not a
# measurement, and because the benchmark's coverage row needs something to score.
LIDAR_WALL_ABS_M = 0.02
LIDAR_WALL_REL = 0.01
# Openings are found on a 5 cm occupancy grid, so quantisation alone is +-2.5 cm.
LIDAR_OPENING_ABS_M = 0.05
LIDAR_AREA_REL = 0.04


def _wall_measurement_lidar(length_m: float) -> Measurement:
    half = LIDAR_WALL_ABS_M + LIDAR_WALL_REL * abs(length_m)
    return Measurement.symmetric(round(length_m, 4), round(half, 4))


def _opening_measurement(value_m: float) -> Measurement:
    return Measurement.symmetric(round(value_m, 4), LIDAR_OPENING_ABS_M)


def build_lidar_plan(
    bundle: CaptureBundle,
    drift_correction: bool = True,
    generated_at: Optional[datetime] = None,
    stride: int = DEFAULT_STRIDE,
    voxel_size_m: float = DEFAULT_VOXEL_M,
    run_ablation: bool = True,
    semantics: bool = True,
    weights_dir: Optional[Path] = None,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    max_frames: int = DEFAULT_MAX_FRAMES,
    calibration: Optional[CalibrationSet] = None,
) -> Tuple[Plan, Dict[str, Any]]:
    """Reconstruct one room from a LiDAR capture.

    ``drift_correction`` selects the room's own axes (estimated from wall
    normals) over the capture's world axes. With it off the layout is built in
    the pose frame exactly as ARKit produced it, which is the "poses used as-is"
    arm of the drift ablation.
    """
    lidar = bundle.payload
    if not isinstance(lidar, LidarCapture):
        raise TypeError(f"expected a LiDAR capture, got {type(lidar).__name__}")

    timings: Dict[str, float] = {}

    # Trajectory-level drift correction: loop closure detection + a pose graph
    # over the whole walkthrough. This is the thing --drift-correction actually
    # switches now -- see cozmo/stitch/drift.py for why it moved off open3d's
    # own (crashing, on this build) PoseGraph.
    mark = time.time()
    odometry = lidar.capture.odometry
    raw_poses = [
        (i, quaternion_to_rotation(row.qx, row.qy, row.qz, row.qw), np.array([row.x, row.y, row.z]))
        for i, row in enumerate(odometry.itertuples(index=False))
    ]
    pose_correction = correct_trajectory(raw_poses, enabled=drift_correction)
    timings["drift_s"] = round(time.time() - mark, 3)

    mark = time.time()
    fused = fuse_capture(
        lidar, stride=stride, voxel_size_m=voxel_size_m, pose_correction=pose_correction
    )
    timings["fuse_s"] = round(time.time() - mark, 3)

    mark = time.time()
    floor = fit_floor(fused.points)
    timings["floor_s"] = round(time.time() - mark, 3)

    mark = time.time()
    layout = extract_layout(
        fused.points, floor, fused.trajectory, manhattan_snap=drift_correction
    )
    timings["layout_s"] = round(time.time() - mark, 3)

    mark = time.time()
    ceiling = estimate_ceiling(
        fused.points, floor,
        camera_height=fused.camera_height_median,
        wall_top_heights=layout.wall_top_heights,
    )
    timings["ceiling_s"] = round(time.time() - mark, 3)

    mark = time.time()
    detections = detect_openings(
        layout.points_uv, layout.points_height, layout.walls, ceiling.height_above_floor_m
    )
    observation = wall_observation_fractions(
        layout.points_uv, layout.points_height, layout.walls, ceiling.height_above_floor_m
    )
    timings["openings_s"] = round(time.time() - mark, 3)

    # The drift ablation. This re-fuses with the correction flipped -- not just
    # a re-run of extract_layout on the same points -- because the point at
    # this task is that --drift-correction has to change the geometry itself
    # (corrected poses feed fuse_capture), not merely which axes a room is
    # drawn in. Costs one extra fuse + floor + layout pass, which is cheap
    # next to everything else in this pipeline.
    ablation_area: Optional[float] = None
    if run_ablation:
        mark = time.time()
        try:
            other_correction = correct_trajectory(raw_poses, enabled=not drift_correction)
            other_fused = fuse_capture(
                lidar, stride=stride, voxel_size_m=voxel_size_m, pose_correction=other_correction
            )
            other_floor = fit_floor(other_fused.points)
            other = extract_layout(
                other_fused.points, other_floor, other_fused.trajectory,
                manhattan_snap=not drift_correction,
            )
            ablation_area = other.floor_area_m2
        except ValueError as exc:
            log.warning("drift ablation failed: %s", exc)
        timings["ablation_s"] = round(time.time() - mark, 3)

    room_id = (bundle.manifest.declared_rooms or ["room_1"])[0]

    # The semantic half of the contract: damage, concealed flags, scope, and a
    # second opinion on the openings. Degrades to geometry-only if the detector
    # or the weights are missing.
    semantic = None
    if semantics:
        mark = time.time()
        semantic = run_semantics_subprocess(
            capture_root=lidar.root,
            layout=layout,
            ceiling_height_m=ceiling.height_above_floor_m,
            geometric_openings=detections,
            room_id=room_id,
            weights_dir=weights_dir,
            frame_stride=frame_stride,
            max_frames=max_frames,
        )
        timings["semantics_s"] = round(time.time() - mark, 3)

    ceiling_measurement = Measurement(
        value=round(ceiling.height_above_floor_m, 4),
        ci_95=(round(ceiling.ci_95[0], 4), round(ceiling.ci_95[1], 4)),
    )

    walls: List[Wall] = []
    surfaces: List[Surface] = []
    openings: List[Opening] = []
    # Openings come from the cross-check when it ran, so a detector-only
    # opening reaches the plan and every opening records which sources agreed.
    opening_source: Sequence[Any] = (
        semantic.openings if semantic is not None and semantic.openings else detections
    )
    by_wall: Dict[int, List[Any]] = {}
    for detection in opening_source:
        by_wall.setdefault(detection.wall_index, []).append(detection)

    for index, segment in enumerate(layout.walls):
        wall_id = f"{room_id}_w{index}"
        wall_openings = by_wall.get(index, [])
        opening_ids = []
        for number, detection in enumerate(wall_openings):
            opening_id = f"{wall_id}_op{number}"
            opening_ids.append(opening_id)
            sources = list(getattr(detection, "sources", ["geometry"]))
            # A detector-only opening is bounded by a projected mask, not
            # measured from depth, so its interval is widened to say so.
            half = LIDAR_OPENING_ABS_M if "geometry" in sources else 3 * LIDAR_OPENING_ABS_M
            openings.append(Opening(
                id=opening_id,
                wall_id=wall_id,
                type=OpeningType.DOOR if detection.kind == "door" else OpeningType.WINDOW,
                width=Measurement.symmetric(round(detection.width_m, 4), half),
                height=Measurement.symmetric(round(detection.height_m, 4), half),
                offset_along_wall=Measurement.symmetric(
                    round(detection.offset_along_wall_m, 4), half
                ),
                sill_height=(
                    Measurement.symmetric(round(detection.sill_height_m, 4), half)
                    if detection.kind == "window" else None
                ),
                detection_confidence=round(float(detection.confidence), 3),
                detection_sources=sources,
                source_note=getattr(detection, "note", "") or None,
            ))

        walls.append(Wall(
            id=wall_id,
            start=Point2D(x=round(segment.start[0], 4), y=round(segment.start[1], 4)),
            end=Point2D(x=round(segment.end[0], 4), y=round(segment.end[1], 4)),
            length=_wall_measurement_lidar(segment.length_m),
            height=ceiling_measurement.model_copy(deep=True),
            opening_ids=opening_ids,
            observation_note=(
                f"observed across {observation[index]:.0%} of its length"
                + (f", to {segment.top_height_m:.2f} m above floor"
                   if segment.top_height_m is not None else "")
            ),
        ))
        surfaces.append(Surface(
            id=f"{wall_id}_surface", room_id=room_id, kind=SurfaceKind.WALL, wall_id=wall_id,
            area=Measurement.relative(
                round(segment.length_m * ceiling.height_above_floor_m, 4),
                LIDAR_AREA_REL, Unit.SQUARE_METERS,
            ),
        ))

    floor_area = Measurement.relative(round(layout.floor_area_m2, 4), LIDAR_AREA_REL, Unit.SQUARE_METERS)
    for kind in (SurfaceKind.FLOOR, SurfaceKind.CEILING):
        surfaces.append(Surface(
            id=f"{room_id}_{kind.value}", room_id=room_id, kind=kind,
            area=floor_area.model_copy(deep=True),
        ))

    room = Room(
        id=room_id,
        name=room_id.replace("_", " ").title(),
        pose=Pose2D(x=0.0, y=0.0, theta_rad=float(layout.frame.rotation_rad)),
        walls=walls,
        openings=openings,
        surfaces=surfaces,
        ceiling_height=ceiling_measurement,
        floor_area=floor_area,
        perimeter=Measurement.symmetric(
            round(layout.perimeter_m, 4),
            round(sum(w.length.half_width for w in walls), 4) if walls else 0.05,
        ),
        source_frame_count=fused.frames_used,
    )
    room = recalibrate_room(room, Tier.LIDAR.value, calibration)

    degradations: List[str] = []
    warnings_out: List[str] = list(bundle.warnings)
    if semantic is not None:
        warnings_out.extend(semantic.warnings)
    elif semantics:
        warnings_out.append("semantic stage did not run")
    if not semantics:
        warnings_out.append(
            "semantic detection disabled for this run; damage, concealed flags and "
            "scope are empty because nothing looked for them"
        )
    if not ceiling.measured:
        degradations.append(f"ceiling not measured ({ceiling.method})")
    if layout.stats.get("manhattan_score", 1.0) < 0.5:
        degradations.append(
            f"weak rectilinear structure (score {layout.stats['manhattan_score']:.2f}); "
            f"the room may not be rectilinear, or wall normals are noisy"
        )
    if fused.stats.get("fraction_above_camera", 1.0) < 0.15:
        degradations.append(
            f"only {fused.stats['fraction_above_camera']:.1%} of points above camera height"
        )
    poorly_seen = [i for i, f in enumerate(observation) if f < 0.6]
    if poorly_seen:
        degradations.append(
            f"walls {poorly_seen} observed across less than 60% of their length; "
            f"absence of openings in them is not evidence they are solid"
        )
    axis_fallback = layout.stats.get("axis_fallback") or []
    if axis_fallback:
        axis_names = ", ".join("u" if a == 0 else "v" for a in axis_fallback)
        degradations.append(
            f"too few walls detected across the room's {axis_names} axis; that boundary is "
            f"padded from where the data happens to stop, not a measured wall -- treat this "
            f"room's size and the walls on that axis as unverified, not just wide"
        )

    if not drift_correction:
        drift_method = DriftMethod.NONE_POSES_AS_IS
    elif pose_correction.loop_closures_used > 0:
        drift_method = DriftMethod.LOOP_CLOSURE
    else:
        drift_method = DriftMethod.POSE_GRAPH

    drift = DriftCorrection(
        enabled=drift_correction,
        method=drift_method,
        loop_closures=pose_correction.loop_closures_used,
        residual_closure_error=(
            Measurement.symmetric(round(pose_correction.residual_closure_error_m, 4), 0.05)
            if pose_correction.residual_closure_error_m is not None else None
        ),
        ablation_footprint_area=(
            Measurement.relative(round(ablation_area, 4), LIDAR_AREA_REL, Unit.SQUARE_METERS)
            if ablation_area is not None else None
        ),
        notes=(
            f"{pose_correction.notes} Wall normals additionally give the room's own axes "
            f"({layout.rotation_deg:.1f} deg off the pose frame)."
            if drift_correction else
            "Poses used as-is: no loop closure detection, no pose graph, the layout is built "
            "on ARKit's world axes with no correction. This is the ablation arm and fails the "
            "drift-accountability gate by design."
        ),
    )

    plan = Plan(
        schema_version=SCHEMA_VERSION,
        capture_id=bundle.capture_id,
        tier=Tier.LIDAR,
        pipeline_version=PIPELINE_VERSION,
        generated_at=generated_at or datetime.now(timezone.utc),
        scale=ScaleInfo(
            source=ScaleSource.LIDAR_DEPTH,
            scale_factor=Measurement(value=1.0, ci_95=(0.995, 1.005), unit=Unit.RATIO),
            reference_description="sensor-metric depth; no external scale reference used",
        ),
        drift_correction=drift,
        property_totals=PropertyTotals(
            room_count=1,
            total_floor_area=recalibrate_measurement(
                floor_area.model_copy(deep=True), Tier.LIDAR.value, "floor_area", calibration
            ),
            footprint_area=recalibrate_measurement(
                floor_area.model_copy(deep=True), Tier.LIDAR.value, "footprint_area", calibration
            ),
            total_wall_area=Measurement.relative(
                round(layout.perimeter_m * ceiling.height_above_floor_m, 4),
                LIDAR_AREA_REL, Unit.SQUARE_METERS,
            ),
            bounding_box_m=(
                round(float(layout.polygon.bounds[2] - layout.polygon.bounds[0]), 3),
                round(float(layout.polygon.bounds[3] - layout.polygon.bounds[1]), 3),
            ),
        ),
        rooms=[room],
        adjacencies=[],
        damage=list(semantic.damage) if semantic else [],
        concealed_flags=list(semantic.concealed_flags) if semantic else [],
        scope=list(semantic.scope) if semantic else [],
        quality=QualityReport(
            overall_confidence=0.70 if ceiling.measured else 0.55,
            interval_method=(
                "Asserted intervals: wall +-(2 cm + 1% of length), openings +-5 cm (one "
                "occupancy cell), areas +-4%, then scaled by this tier's fitted calibration "
                "factor (1.0 if uncalibrated). The ceiling interval is real -- it comes from "
                "the ceiling estimator -- and is calibrated the same way."
            ),
            calibration_note=calibration_note_for(Tier.LIDAR.value, calibration),
            ceiling_method=ceiling.method,
            semantics_available=bool(semantic.available) if semantic else False,
            degradations=degradations,
            warnings=warnings_out,
            coverage={
                "fraction_above_camera": float(fused.stats.get("fraction_above_camera", 0.0)),
                "wall_band_points": float(layout.stats.get("wall_band_points", 0)),
                "mean_wall_observation": round(
                    float(np.mean(observation)) if observation else 0.0, 4
                ),
                "walls_well_observed": float(
                    sum(1 for f in observation if f >= 0.6)
                ),
            },
        ),
    )

    details: Dict[str, Any] = {
        "fusion": fused.summary(),
        "floor_plane": floor.summary(),
        "layout": layout.summary(),
        "ceiling": ceiling.summary(),
        "wall_observation_fraction": [round(f, 3) for f in observation],
        "semantics": semantic.details if semantic else {"detector": "disabled"},
        # The plan's own openings come from opening_source (detector-confirmed
        # openings included, not just geometry's), so the drawing has to be
        # built from the same list -- rendering from `detections` alone drew
        # an empty plan.png whenever every opening came from the semantic
        # cross-check (a doorway geometry's sparse occupancy grid missed).
        "openings": [
            {
                "wall_index": d.wall_index, "kind": d.kind,
                "width_m": round(d.width_m, 3), "height_m": round(d.height_m, 3),
                "offset_along_wall_m": round(d.offset_along_wall_m, 3),
                "sill_height_m": round(d.sill_height_m, 3), "confidence": d.confidence,
                **getattr(d, "stats", {}),
            }
            for d in opening_source
        ],
        "ablation_floor_area_m2": round(ablation_area, 4) if ablation_area is not None else None,
        "timings_s": timings,
        "layout_result": layout,     # not serialised; used by the renderer
    }
    return plan, details


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
    rendered: List[Path] = field(default_factory=list)


@dataclass
class _RenderWall:
    """The few fields render_plan needs from a wall, as plain tuples/floats."""

    start: Tuple[float, float]
    end: Tuple[float, float]
    length_m: float
    normal: Tuple[float, float] = (0.0, 0.0)


@dataclass
class _RenderOpening:
    """The few fields the renderer needs from a detected opening."""

    wall_index: int
    kind: str
    width_m: float
    offset_along_wall_m: float


def _detections_for_render(openings: Sequence[Dict[str, Any]]) -> List[_RenderOpening]:
    return [
        _RenderOpening(
            wall_index=int(o["wall_index"]), kind=str(o["kind"]),
            width_m=float(o["width_m"]),
            offset_along_wall_m=float(o.get("offset_along_wall_m", 0.0)),
        )
        for o in openings
    ]


def run_capture(
    input_dir: Path,
    out_dir: Path,
    drift_correction: bool = True,
    seed: int = DEFAULT_SEED,
    command: Optional[Sequence[str]] = None,
    stride: int = DEFAULT_STRIDE,
    voxel_size_m: float = DEFAULT_VOXEL_M,
    semantics: bool = True,
    weights_dir: Optional[Path] = None,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    max_frames: int = DEFAULT_MAX_FRAMES,
    backbone_name: str = DEFAULT_BACKBONE,
    unsafe_scale_cues: bool = False,
    video_stride: int = VIDEO_DEFAULT_STRIDE_FRAMES,
    video_blur_threshold: float = VIDEO_DEFAULT_BLUR_THRESHOLD,
    video_max_frames: int = VIDEO_DEFAULT_MAX_FRAMES,
    ignore_calibration: bool = False,
) -> RunResult:
    """Run one capture: load, reconstruct, write plan, manifest and drawing.

    ``ignore_calibration`` forces every interval factor to 1.0 for this run --
    used by ``cozmo calibrate`` itself, which has to measure the pipeline's
    *asserted* half-widths to fit a factor, not one already widened by a
    previous fit.
    """
    started = time.time()
    seed_record = set_global_seeds(seed)
    calibration = IDENTITY_CALIBRATION if ignore_calibration else CALIBRATION

    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_capture(input_dir)
    input_hash = hash_directory(input_dir)
    generated_at = _deterministic_now()

    rendered: List[Path] = []
    if bundle.tier in (Tier.PHOTO, Tier.VIDEO):
        if bundle.tier is Tier.PHOTO:
            plan, reconstruction = build_photo_plan(
                bundle, generated_at=generated_at, backbone_name=backbone_name,
                weights_dir=weights_dir,
                run_metric_depth_cue=unsafe_scale_cues, run_door_cue=unsafe_scale_cues,
                calibration=calibration,
            )
        else:
            plan, reconstruction = build_video_plan(
                bundle, generated_at=generated_at, backbone_name=backbone_name,
                weights_dir=weights_dir,
                run_metric_depth_cue=unsafe_scale_cues, run_door_cue=unsafe_scale_cues,
                stride_frames=video_stride, blur_threshold=video_blur_threshold,
                max_frames=video_max_frames, calibration=calibration,
            )
        reconstruction.pop("layout_result", None)
        reconstruction.pop("scale_factor", None)
        if plan.rooms:
            try:
                # Every room's Wall already carries its FINAL, globally-placed
                # coordinates -- build_photo_plan's single room sits at its own
                # origin (which *is* the global frame for one room);
                # build_multi_room_photo_plan bakes each room's stitched pose
                # into its walls directly. So rendering needs nothing from the
                # reconstruction beyond the plan itself, single- or multi-room
                # alike -- only a flat wall list (render_plan indexes openings
                # positionally into it) and plain (start, end, length_m)
                # segments rather than the pydantic Point2D/Measurement the
                # plan itself uses.
                all_walls: List[_RenderWall] = []
                all_openings: List[_RenderOpening] = []
                room_wall_counts: List[int] = []
                for room in plan.rooms:
                    base = len(all_walls)
                    room_wall_counts.append(len(room.walls))
                    wall_index_by_id = {w.id: base + i for i, w in enumerate(room.walls)}
                    for wall in room.walls:
                        all_walls.append(_RenderWall(
                            start=(wall.start.x, wall.start.y), end=(wall.end.x, wall.end.y),
                            length_m=wall.length.value,
                        ))
                    for opening in room.openings:
                        all_openings.append(_RenderOpening(
                            wall_index=wall_index_by_id.get(opening.wall_id, base),
                            kind=opening.type.value, width_m=opening.width.value,
                            offset_along_wall_m=opening.offset_along_wall.value,
                        ))

                room_names = ", ".join(r.name for r in plan.rooms[:4])
                if len(plan.rooms) > 4:
                    room_names += f" + {len(plan.rooms) - 4} more"
                rendered = render_plan(
                    all_walls, all_openings, out_dir,
                    title=f"{plan.capture_id} - {room_names}",
                    subtitle=(
                        f"{plan.property_totals.footprint_area.value:.2f} m2 footprint "
                        f"({bundle.tier.value} tier, {len(plan.rooms)} room(s), "
                        f"{len(plan.adjacencies)} connection(s))"
                    ),
                    rotation_deg=0.0,   # rooms are already placed in one shared global frame
                    room_wall_counts=room_wall_counts,
                )
            except Exception as exc:  # noqa: BLE001 - a failed drawing must not lose the plan
                log.warning("plan rendering failed: %s", exc)
    elif bundle.tier is Tier.LIDAR:
        plan, details = build_lidar_plan(
            bundle,
            drift_correction=drift_correction,
            generated_at=generated_at,
            stride=stride,
            voxel_size_m=voxel_size_m,
            semantics=semantics,
            weights_dir=weights_dir,
            frame_stride=frame_stride,
            max_frames=max_frames,
            calibration=calibration,
        )
        layout = details.pop("layout_result")
        try:
            rendered = render_plan(
                layout.walls,
                [d for d in _detections_for_render(details["openings"])],
                out_dir,
                title=f"{plan.capture_id} - {plan.rooms[0].name}",
                subtitle=(
                    f"{plan.rooms[0].floor_area.value:.2f} m2 floor area  |  ceiling "
                    f"{plan.rooms[0].ceiling_height.value:.2f} m ({plan.quality.ceiling_method})  |  "
                    f"drift correction {'on' if drift_correction else 'off'}"
                ),
                trajectory_uv=layout.frame.project(bundle.payload.trajectory()),
                rotation_deg=layout.rotation_deg,
            )
        except Exception as exc:  # noqa: BLE001 - a failed drawing must not lose the plan
            log.warning("plan rendering failed: %s", exc)
        reconstruction: Optional[Dict[str, Any]] = details
    else:  # pragma: no cover - Tier is exhaustive (photo, video, lidar)
        raise ValueError(f"unhandled tier {bundle.tier}")

    plan_path = out_dir / PLAN_FILENAME
    plan_path.write_text(plan.to_json() + "\n")

    manifest: Dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "package_version": __version__,
        "git": git_provenance(),
        "command": " ".join(command) if command else " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "argv": list(command) if command else list(sys.argv),
        "seed": seed_record,
        "options": {"drift_correction": drift_correction, "ignore_calibration": ignore_calibration},
        "input": input_hash,
        "capture": bundle.summary(),
        "outputs": {
            PLAN_FILENAME: sha256_file(plan_path),
            **{path.name: sha256_file(path) for path in rendered},
        },
        "reconstruction": reconstruction,
        "reconstruction_settings": (
            {
                "stride": stride, "voxel_size_m": voxel_size_m,
                "semantics": semantics, "frame_stride": frame_stride,
                "max_frames": max_frames,
                "weights_dir": str(weights_dir) if weights_dir else None,
            }
            if bundle.tier is Tier.LIDAR
            else {
                "backbone": backbone_name, "weights_dir": str(weights_dir) if weights_dir else None,
                "video_stride": video_stride, "video_blur_threshold": video_blur_threshold,
                "video_max_frames": video_max_frames,
            }
            if bundle.tier is Tier.VIDEO
            else {"backbone": backbone_name, "weights_dir": str(weights_dir) if weights_dir else None}
            if bundle.tier is Tier.PHOTO else None
        ),
        "calibration": {
            "loaded": calibration.is_loaded,
            "path": str(calibration.path) if calibration.path else None,
            "calibration_version": calibration.calibration_version,
            "fitted_at": calibration.fitted_at,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "source_date_epoch": os.environ.get("SOURCE_DATE_EPOCH"),
        },
        "started_at": datetime.fromtimestamp(started, tz=timezone.utc).isoformat(),
        "duration_s": round(time.time() - started, 4),
    }
    manifest_path = out_dir / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False, default=str) + "\n")

    return RunResult(
        plan=plan, plan_path=plan_path, manifest_path=manifest_path,
        manifest=manifest, rendered=rendered,
    )
