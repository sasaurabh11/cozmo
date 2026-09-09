"""Photo tier: the real reconstruction.

    frames -> VGGT (scale-free) -> layout (Plane-DUSt3R ordering) -> scale
    recovery -> metric layout -> openings (geometry/openings.py, reused as-is)

Single room only: a photo folder is one room's worth of stills, and stitching
several rooms' reconstructions together into one property is the next phase.

Every interval leaving this module is honest about carrying two independent
sources of uncertainty: the layout's own per-wall confidence (view count and
cross-view agreement -- see recon/layout.py) and the scale factor's own
interval (cue agreement -- see recon/scale.py). They are combined in
quadrature, not added, and not silently dropped in favour of whichever is
smaller.
"""

from __future__ import annotations

import logging
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import itertools

import numpy as np
from shapely.affinity import affine_transform
from shapely.ops import unary_union

from .. import PIPELINE_VERSION, SCHEMA_VERSION
from ..calibration import CalibrationSet, calibration_note_for, recalibrate_measurement, recalibrate_room
from ..geometry.openings import detect_openings, wall_observation_fractions
from ..semantics.detect import cross_check_openings
from ..io.capture import CaptureBundle
from ..recon.backbone import ReconstructorUnavailable, get_reconstructor
from ..recon.frames import load_photo_folder, summarize as summarize_frames
from ..geometry.layout import FloorFrame, WallSegment
from ..recon.layout import extract_layout_photo
from ..recon.scale import recover_scale
from ..stitch.graph import RoomForStitch, StitchResult, build_stitch_graph
from ..stitch.match import RoomFrames, RoomPairMatch
from ..schema import (
    Adjacency,
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
    Surface,
    SurfaceKind,
    Tier,
    Unit,
    Wall,
)

log = logging.getLogger("cozmo.pipeline.photo")

# Photo-tier gate: wall lengths within +-8% (Round 2 gate). A wall's own
# confidence (view count, cross-view agreement) narrows or widens around that
# floor -- a wall seen from every photo and agreeing tightly can claim better
# than 8%; one seen from a single view cannot claim better than the gate itself.
WALL_REL_FLOOR = 0.05
WALL_REL_CEILING = 0.15
AREA_REL_FLOOR = 0.08

DEFAULT_BACKBONE = "vggt"

PHOTO_OPENING_TIMEOUT_S = 900
PHOTO_OPENING_GRID = 7

# A capture set can contain several folders from the same physical room. A
# high visual similarity plus many geometric keypoint matches is stronger
# evidence of a repeated capture than of two adjacent rooms.
REPEATED_CAPTURE_SIMILARITY = 0.75
REPEATED_CAPTURE_MIN_MATCHES = 20


@dataclass
class PhotoOpeningRegion:
    """A 2D detector box projected onto one reconstructed wall."""

    wall_id: str
    label: str
    score: float
    bbox: Tuple[float, float, float, float]


def _select_photo_openings(regions: List[PhotoOpeningRegion]) -> List[PhotoOpeningRegion]:
    """Keep one strongest semantic hypothesis per opening kind.

    VGGT poses are scale-free and can be weakly constrained when several phone
    photos come from nearly the same viewpoint. In that case a detector box
    from a side view can project onto a neighbouring wall and look like a new
    opening. The strongest door and strongest window hypotheses are stable;
    weaker duplicates are not. Geometry remains free to add independently
    measured openings, and LiDAR still uses its full multi-view consensus.
    """
    chosen: Dict[str, PhotoOpeningRegion] = {}
    for region in regions:
        kind = "door" if region.label in {"door", "doorway"} else "window"
        if kind not in chosen or region.score > chosen[kind].score:
            chosen[kind] = region
    return list(chosen.values())


def _run_photo_opening_detector(
    frames: List[Any], weights_dir: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    """Run Grounding DINO in a child process and return image-space boxes.

    The parent has already imported Open3D. Loading torch here is unsafe on
    macOS, so this follows the same process boundary as the LiDAR semantic
    stage. A failed detector is reported and leaves geometry as the fallback.
    """
    request = {
        "weights_dir": str(weights_dir) if weights_dir else None,
        "frames": [{"index": int(f.index), "path": str(f.path)} for f in frames],
    }
    with tempfile.TemporaryDirectory(prefix="cozmo-photo-openings-") as scratch:
        request_path = Path(scratch) / "request.json"
        response_path = Path(scratch) / "response.json"
        request_path.write_text(json.dumps(request))
        command = [sys.executable, "-m", "cozmo.semantics.photo_worker",
                   str(request_path), str(response_path)]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True,
                timeout=PHOTO_OPENING_TIMEOUT_S, check=False,
                cwd=str(Path(__file__).resolve().parents[2]),
            )
        except subprocess.TimeoutExpired:
            message = (
                f"photo opening detector timed out after {PHOTO_OPENING_TIMEOUT_S}s; "
                "the plan carries geometry-only openings"
            )
            return [], [message], {"detector": "timeout"}

        if not response_path.is_file():
            message = (
                f"photo opening detector produced no response (exit {completed.returncode}); "
                "the plan carries geometry-only openings"
            )
            return [], [message], {
                "detector": "failed", "exit_code": completed.returncode,
                "stderr_tail": (completed.stderr or "").strip()[-1200:],
            }
        payload = json.loads(response_path.read_text())

    if not payload.get("ok"):
        message = (
            f"photo opening detector unavailable ({payload.get('error', 'unknown error')}); "
            "the plan carries geometry-only openings"
        )
        return [], [message], {
            "detector": "unavailable", "reason": payload.get("error", "unknown error"),
        }
    return list(payload.get("detections", [])), [], {
        "detector": payload.get("detector", "grounding-dino-tiny"),
        "device": payload.get("device", "unknown"),
        "detections": len(payload.get("detections", [])),
    }


def _project_photo_opening(
    detection: Dict[str, Any], frame: Any, pose: Any, layout: Any,
    scale: float, room_id: str,
) -> Optional[PhotoOpeningRegion]:
    """Project a Grounding-DINO box through the VGGT pose onto a wall.

    The detector supplies image semantics; the reconstructed wall supplies the
    metric coordinate system. Sampling several rays inside the box prevents a
    corner pixel or a partial occlusion from inventing an opening span.
    """
    image_height, image_width = frame.image.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in detection["box_xyxy"])
    x0, x1 = sorted((max(0.0, min(image_width - 1.0, x0)),
                     max(0.0, min(image_width - 1.0, x1))))
    y0, y1 = sorted((max(0.0, min(image_height - 1.0, y0)),
                     max(0.0, min(image_height - 1.0, y1))))
    if x1 - x0 < 12.0 or y1 - y0 < 12.0:
        return None

    fractions = np.linspace(0.12, 0.88, PHOTO_OPENING_GRID)
    pixels = np.array([(x0 + fx * (x1 - x0), y0 + fy * (y1 - y0))
                       for fy in fractions for fx in fractions], dtype=float)
    fx, fy = float(frame.K[0, 0]), float(frame.K[1, 1])
    cx, cy = float(frame.K[0, 2]), float(frame.K[1, 2])
    rays_camera = np.column_stack(((pixels[:, 0] - cx) / fx,
                                   (pixels[:, 1] - cy) / fy,
                                   np.ones(len(pixels))))
    rays_world = rays_camera @ np.asarray(pose.R, dtype=float).T
    origin = np.asarray(pose.t, dtype=float)
    best: Optional[Tuple[int, np.ndarray, np.ndarray, Any]] = None

    for wall_index, wall in enumerate(layout.walls):
        start = np.asarray(wall.start, dtype=float)
        end = np.asarray(wall.end, dtype=float)
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length <= 1e-6:
            continue
        direction /= length
        normal_2d = np.asarray(wall.normal, dtype=float)
        if np.linalg.norm(normal_2d) <= 1e-6:
            normal_2d = np.array([-direction[1], direction[0]])
        normal_2d /= np.linalg.norm(normal_2d)
        wall_point = layout.frame.to_world(start[None, :], height=0.0)[0]
        normal_world = normal_2d[0] * layout.frame.e1 + normal_2d[1] * layout.frame.e2
        denominator = rays_world @ normal_world
        usable = np.abs(denominator) > 0.08
        if not np.any(usable):
            continue
        distance = np.full(len(rays_world), np.nan)
        distance[usable] = ((wall_point - origin) @ normal_world) / denominator[usable]
        hits = origin + distance[:, None] * rays_world
        uv = layout.frame.project(hits)
        along = (uv - start[None, :]) @ direction
        heights = (hits - layout.frame.origin[None, :]) @ layout.frame.up
        valid = (
            np.isfinite(distance) & (distance > 0.15) & (distance < 15.0)
            & (along >= -0.15) & (along <= length + 0.15)
            & (heights >= -0.10) & (heights <= layout.floor_to_ceiling_span * 1.2
                                      if layout.floor_to_ceiling_span else heights <= 4.0)
        )
        count = int(np.count_nonzero(valid))
        if count < max(5, int(0.18 * len(pixels))):
            continue
        if best is None or count > best[0]:
            best = (count, along[valid], heights[valid], wall_index)

    if best is None:
        return None
    _count, along, heights, wall_index = best
    # Quantiles discard the few rays that graze a wall edge or pass through a
    # partially occluded detector box.
    along_low, along_high = np.quantile(along, [0.05, 0.95])
    height_low, height_high = np.quantile(heights, [0.05, 0.95])
    width_m = float((along_high - along_low) * scale)
    height_m = float((height_high - height_low) * scale)
    kind = "door" if detection["prompt"] in {"door", "doorway"} else "window"
    width_range = (0.55, 1.40) if kind == "door" else (0.35, 2.60)
    height_min = 1.20 if kind == "door" else 0.30
    if not (width_range[0] <= width_m <= width_range[1]) or height_m < height_min:
        return None
    return PhotoOpeningRegion(
        wall_id=f"{room_id}_w{wall_index}",
        label=detection["prompt"], score=float(detection["score"]),
        bbox=(float(along_low * scale), float(height_low * scale),
              float(along_high * scale), float(height_high * scale)),
    )


def _wall_relative_uncertainty(agreement: float, view_count: int) -> float:
    """A wall's own claimed precision, before the scale factor's uncertainty
    is folded in. Single-view walls and low cross-view agreement both widen
    this -- they are different failure modes (an unmeasured wall vs. a
    disputed one) and both push toward the same place: wider, not narrower."""
    base = WALL_REL_CEILING - agreement * (WALL_REL_CEILING - WALL_REL_FLOOR)
    if view_count <= 1:
        base = max(base, WALL_REL_CEILING)
    return float(np.clip(base, WALL_REL_FLOOR, WALL_REL_CEILING))


def _combine_relative(*relatives: float) -> float:
    """Independent relative uncertainties combine in quadrature, not by adding
    or by taking whichever is larger -- the standard rule for uncorrelated
    error sources, and the reason a photo-tier interval is wider than either
    the layout's or the scale factor's alone."""
    return float(np.sqrt(sum(r ** 2 for r in relatives)))


@dataclass
class RoomReconstruction:
    """Everything one room's reconstruction produced, before it is either
    wrapped into a single-room Plan or handed to the stitcher."""

    room_id: str
    room: Room
    layout: Any
    reconstruction: Any
    scale_estimate: Any
    scale: float
    frames: List[Any]
    detections: List[Any]
    opening_consensus: List[Any]
    semantic_openings_available: bool
    semantic_opening_details: Dict[str, Any]
    timings: Dict[str, float]
    warnings: List[str]
    degradations: List[str]


def _reconstruct_room(
    room_dir: Path,
    room_id: str,
    backbone_name: str = DEFAULT_BACKBONE,
    weights_dir: Optional[Path] = None,
    run_metric_depth_cue: bool = False,
    run_door_cue: bool = False,
) -> RoomReconstruction:
    """One room, start to finish: photos in, a placed-at-its-own-origin Room
    (with its own walls, openings, surfaces) out. Shared by the single-room
    path (:func:`build_photo_plan`) and the multi-room stitching path
    (:func:`build_multi_room_photo_plan`) -- everything either of them needs
    to do *after* reconstruction is the same; only what happens to the room's
    pose differs (left at the origin, or moved by the stitcher).
    """
    timings: Dict[str, float] = {}
    mark = time.time()
    frames = load_photo_folder(room_dir)
    timings["load_frames_s"] = round(time.time() - mark, 3)
    if not (2 <= len(frames) <= 8):
        log.warning(
            "%d photos in %s; the contract's floor/ceiling is 2-8 per room", len(frames), room_dir
        )

    mark = time.time()
    try:
        reconstructor = get_reconstructor(backbone_name, weights_dir=weights_dir)
        reconstruction = reconstructor.reconstruct(frames)
    except ReconstructorUnavailable as exc:
        raise ReconstructorUnavailable(f"backbone '{backbone_name}' unavailable: {exc}") from exc
    timings["reconstruct_s"] = round(time.time() - mark, 3)

    mark = time.time()
    layout = extract_layout_photo(reconstruction)
    timings["layout_s"] = round(time.time() - mark, 3)

    mark = time.time()
    scale_estimate = recover_scale(
        frames, reconstruction, layout, weights_dir=weights_dir,
        run_metric_depth=run_metric_depth_cue, run_door_cue=run_door_cue,
    )
    timings["scale_s"] = round(time.time() - mark, 3)
    scale = scale_estimate.scale_factor
    scale_rel = max(
        abs(scale_estimate.ci_95[1] - scale) / scale,
        abs(scale - scale_estimate.ci_95[0]) / scale,
    ) if scale > 0 else 0.5

    ceiling_height_units = layout.floor_to_ceiling_span
    scaled_points_uv = layout.points_uv * scale
    scaled_points_height = layout.points_height * scale
    scaled_ceiling_m = ceiling_height_units * scale if ceiling_height_units else 2.44 * 1.0

    mark = time.time()
    detections = detect_openings(
        scaled_points_uv, scaled_points_height, layout.walls, scaled_ceiling_m
    )
    detector_mark = time.time()
    if backbone_name == "stub":
        # The stub backbone is used by plumbing tests and intentionally does
        # not represent real camera geometry; running an expensive detector on
        # its synthetic frames would add noise to those tests.
        semantic_detections, semantic_warnings, semantic_details = [], [], {"detector": "disabled"}
    else:
        semantic_detections, semantic_warnings, semantic_details = _run_photo_opening_detector(
            frames, weights_dir=weights_dir,
        )
    semantic_regions: List[PhotoOpeningRegion] = []
    frame_by_index = {frame.index: frame for frame in frames}
    pose_by_index = {pose.frame_index: pose for pose in reconstruction.poses}
    for semantic_detection in semantic_detections:
        frame = frame_by_index.get(int(semantic_detection.get("frame_index", -1)))
        pose = pose_by_index.get(int(semantic_detection.get("frame_index", -1)))
        if frame is None or pose is None:
            continue
        projected = _project_photo_opening(
            semantic_detection, frame, pose, layout, scale, room_id,
        )
        if projected is not None:
            semantic_regions.append(projected)
    semantic_regions = _select_photo_openings(semantic_regions)
    opening_consensus = cross_check_openings(
        detections,
        semantic_regions,
        wall_index_by_id={f"{room_id}_w{i}": i for i in range(len(layout.walls))},
    )
    observation = wall_observation_fractions(
        scaled_points_uv, scaled_points_height, layout.walls, scaled_ceiling_m
    )
    timings["opening_detector_s"] = round(time.time() - detector_mark, 3)
    timings["openings_s"] = round(time.time() - mark, 3)

    walls: List[Wall] = []
    surfaces: List[Surface] = []
    openings: List[Opening] = []
    by_wall: Dict[int, List[Any]] = {}
    for detection in opening_consensus:
        by_wall.setdefault(detection.wall_index, []).append(detection)

    wall_rel = WALL_REL_CEILING  # overwritten per-wall below; kept for area_rel's use after the loop
    for index, segment in enumerate(layout.walls):
        wall_id = f"{room_id}_w{index}"
        agreement = layout.wall_confidence.get(index, 0.4)
        view_count = layout.stats.get("view_details", {}).get(index, {}).get("view_count", 1)
        wall_rel = _combine_relative(_wall_relative_uncertainty(agreement, view_count), scale_rel)

        length_m = segment.length_m * scale
        half = max(length_m * wall_rel, 0.02)

        wall_openings = by_wall.get(index, [])
        opening_ids = []
        for number, detection in enumerate(wall_openings):
            opening_id = f"{wall_id}_op{number}"
            opening_ids.append(opening_id)
            op_half = max(detection.width_m * (wall_rel + 0.05), 0.03)
            openings.append(Opening(
                id=opening_id, wall_id=wall_id,
                type=OpeningType.DOOR if detection.kind == "door" else OpeningType.WINDOW,
                width=Measurement.symmetric(round(detection.width_m, 4), round(op_half, 4)),
                height=Measurement.symmetric(round(detection.height_m, 4), round(op_half, 4)),
                offset_along_wall=Measurement.symmetric(round(detection.offset_along_wall_m, 4), round(op_half, 4)),
                sill_height=(
                    Measurement.symmetric(round(detection.sill_height_m, 4), round(op_half, 4))
                    if detection.kind == "window" else None
                ),
                detection_confidence=round(float(detection.confidence), 3),
                detection_sources=list(detection.sources),
                source_note=detection.note or None,
            ))

        walls.append(Wall(
            id=wall_id,
            start=Point2D(x=round(segment.start[0] * scale, 4), y=round(segment.start[1] * scale, 4)),
            end=Point2D(x=round(segment.end[0] * scale, 4), y=round(segment.end[1] * scale, 4)),
            length=Measurement.symmetric(round(length_m, 4), round(half, 4)),
            height=Measurement.symmetric(round(scaled_ceiling_m, 4), round(scaled_ceiling_m * wall_rel, 4)),
            opening_ids=opening_ids,
            observation_note=(
                f"{view_count} view(s), cross-view agreement {agreement:.2f}, "
                f"observed across {observation[index]:.0%} of its length"
            ),
        ))
        surfaces.append(Surface(
            id=f"{wall_id}_surface", room_id=room_id, kind=SurfaceKind.WALL, wall_id=wall_id,
            area=Measurement.relative(round(length_m * scaled_ceiling_m, 4), wall_rel, Unit.SQUARE_METERS),
        ))

    area_rel = _combine_relative(max(AREA_REL_FLOOR, 2 * scale_rel), 0.03)
    floor_area_m2 = layout.floor_area_m2 * scale * scale
    floor_area = Measurement.relative(round(floor_area_m2, 4), area_rel, Unit.SQUARE_METERS)
    for kind in (SurfaceKind.FLOOR, SurfaceKind.CEILING):
        surfaces.append(Surface(
            id=f"{room_id}_{kind.value}", room_id=room_id, kind=kind,
            area=floor_area.model_copy(deep=True),
        ))

    ceiling_measurement = Measurement.symmetric(
        round(scaled_ceiling_m, 4), round(scaled_ceiling_m * max(scale_rel, 0.05), 4)
    )
    perimeter_m = layout.perimeter_m * scale
    room = Room(
        id=room_id, name=room_id.replace("_", " ").title(),
        pose=Pose2D(x=0.0, y=0.0, theta_rad=float(layout.frame.rotation_rad)),
        walls=walls, openings=openings, surfaces=surfaces,
        ceiling_height=ceiling_measurement, floor_area=floor_area,
        perimeter=Measurement.relative(round(perimeter_m, 4), wall_rel if walls else 0.10),
        source_frame_count=len(frames),
    )

    degradations: List[str] = []
    warnings_out: List[str] = list(summarize_frames(frames).get("warnings", []))
    warnings_out.extend(semantic_warnings)
    if not (2 <= len(frames) <= 8):
        degradations.append(f"{len(frames)} photos supplied; contract floor/ceiling is 2-8 per room")
    if scale_estimate.agreement < 0.5:
        degradations.append(
            f"scale cues disagree (agreement {scale_estimate.agreement:.2f}); "
            f"the scale interval is widened accordingly, not narrowed by averaging over it"
        )
    axis_fallback = layout.stats.get("axis_fallback") or []
    if axis_fallback:
        axis_names = ", ".join("u" if a == 0 else "v" for a in axis_fallback)
        degradations.append(
            f"too few walls detected across the room's {axis_names} axis; that boundary is "
            f"padded from where the data happens to stop, not a measured wall -- treat this "
            f"room's size and the walls on that axis as unverified, not just wide"
        )

    return RoomReconstruction(
        room_id=room_id, room=room, layout=layout, reconstruction=reconstruction,
        scale_estimate=scale_estimate, scale=scale, frames=frames, detections=detections,
        opening_consensus=opening_consensus,
        semantic_openings_available=bool(semantic_details.get("detector"))
            and semantic_details.get("detector") not in {
                "disabled", "failed", "timeout", "unavailable",
            },
        semantic_opening_details=semantic_details,
        timings=timings, warnings=warnings_out, degradations=degradations,
    )


def _assemble_single_room_plan(
    bundle: CaptureBundle,
    result: RoomReconstruction,
    tier: Tier,
    generated_at: Optional[datetime],
    backbone_name: str,
    extra_warnings: Optional[List[str]] = None,
    extra_degradations: Optional[List[str]] = None,
    video_sampling: Optional[Dict[str, Any]] = None,
    drift_note_prefix: str = "",
    calibration: Optional[CalibrationSet] = None,
) -> Tuple[Plan, Dict[str, Any]]:
    """Build a single-room Plan from one room's reconstruction.

    Shared by the photo tier (:func:`build_photo_plan`'s single-room branch)
    and the video tier (:mod:`cozmo.pipeline.video`) -- a video capture is
    nothing but more views of the same unposed-stills problem once frames
    have been sampled from it (see cozmo/io/video.py), so everything from
    reconstruction onward is identical; only the tier tag, a note about how
    the views were obtained, and the calibration factors applied (calibration
    is fit separately per tier) differ.
    """
    layout, scale, frames = result.layout, result.scale, result.frames
    scale_estimate, room, detections = result.scale_estimate, result.room, result.detections
    warnings_out = list(bundle.warnings) + result.warnings + list(extra_warnings or [])
    degradations = list(result.degradations) + list(extra_degradations or [])
    timings = result.timings

    room = recalibrate_room(room, tier.value, calibration)

    drift = DriftCorrection(
        enabled=True, method=DriftMethod.PLANE_ANCHORED, loop_closures=0,
        notes=(
            f"{drift_note_prefix}Single-room reconstruction ({layout.layout_method}); no "
            f"multi-room stitch, so no cross-room drift to correct. Per-wall confidence comes "
            f"from view count and cross-view plane agreement instead of a pose graph residual."
        ),
    )

    fired_cues = [c.name for c in scale_estimate.cues if c.fired]
    plan = Plan(
        schema_version=SCHEMA_VERSION, capture_id=bundle.capture_id, tier=tier,
        pipeline_version=PIPELINE_VERSION, generated_at=generated_at or datetime.now(timezone.utc),
        scale=ScaleInfo(
            source=(
                ScaleSource.STRUCTURAL_PRIOR if "door_height" in fired_cues or "ceiling_height" in fired_cues
                else ScaleSource.REFERENCE_OBJECT
            ),
            scale_factor=Measurement(
                value=round(scale, 5),
                ci_95=(round(scale_estimate.ci_95[0], 5), round(scale_estimate.ci_95[1], 5)),
                unit=Unit.RATIO,
            ),
            reference_description=(
                f"{len(fired_cues)} cue(s) fired: {', '.join(fired_cues) if fired_cues else 'none'} "
                f"(agreement {scale_estimate.agreement:.2f}, method {scale_estimate.method})"
            ),
        ),
        drift_correction=drift,
        property_totals=PropertyTotals(
            room_count=1,
            total_floor_area=recalibrate_measurement(
                room.floor_area.model_copy(deep=True), tier.value, "floor_area", calibration
            ),
            footprint_area=recalibrate_measurement(
                room.floor_area.model_copy(deep=True), tier.value, "footprint_area", calibration
            ),
            total_wall_area=Measurement.relative(
                round(room.perimeter.value * room.ceiling_height.value, 4),
                max(room.perimeter.relative_half_width or 0.08, 0.08), Unit.SQUARE_METERS
            ),
            bounding_box_m=(
                round(float(layout.polygon.bounds[2] - layout.polygon.bounds[0]) * scale, 3),
                round(float(layout.polygon.bounds[3] - layout.polygon.bounds[1]) * scale, 3),
            ),
        ),
        rooms=[room], adjacencies=[], damage=[], concealed_flags=[], scope=[],
        quality=QualityReport(
            overall_confidence=round(min(0.75, 0.35 + 0.2 * len(fired_cues) + 0.15 * scale_estimate.agreement), 3),
            interval_method=(
                f"{tier.value.title()} tier: per-wall relative uncertainty from view count and "
                f"cross-view plane agreement, combined in quadrature with the scale factor's own "
                f"interval (cue disagreement), then scaled by this tier's fitted calibration "
                f"factor (1.0 if uncalibrated). No sensor-noise error budget -- there is no depth "
                f"sensor at this tier."
            ),
            calibration_note=calibration_note_for(tier.value, calibration),
            ceiling_method=layout.layout_method,
            semantics_available=result.semantic_openings_available,
            degradations=degradations,
            warnings=warnings_out,
            coverage={
                "layout_method": 1.0 if layout.layout_method == "per_view_merge" else 0.0,
                "scale_cues_fired": float(len(fired_cues)),
                "scale_agreement": float(scale_estimate.agreement),
            },
            video_sampling=video_sampling,
        ),
    )

    details = {
        "layout_result": layout,
        "scale_factor": scale,
        "backbone": backbone_name,
        "reconstruction_stats": result.reconstruction.stats,
        "layout": layout.stats,
        "layout_method": layout.layout_method,
        "view_plane_counts": layout.view_plane_counts,
        "scale": scale_estimate.as_dict(),
        "frames": summarize_frames(frames),
        "openings": [
            {"wall_index": d.wall_index, "kind": d.kind,
             "width_m": round(d.width_m, 3),
             "confidence": d.confidence, "sources": list(d.sources)}
            for d in result.opening_consensus
        ],
        "opening_detector": result.semantic_opening_details,
        "timings_s": timings,
    }
    if video_sampling is not None:
        details["video_sampling"] = video_sampling
    return plan, details


def build_photo_plan(
    bundle: CaptureBundle,
    generated_at: Optional[datetime] = None,
    backbone_name: str = DEFAULT_BACKBONE,
    weights_dir: Optional[Path] = None,
    run_metric_depth_cue: bool = False,
    run_door_cue: bool = False,
    calibration: Optional[CalibrationSet] = None,
) -> Tuple[Plan, Dict[str, Any]]:
    """Reconstruct one room from a folder of 2-8 unposed photos.

    ``run_metric_depth_cue`` and ``run_door_cue`` default off: both load torch
    (ZoeDepth, Grounding DINO) into *this* process, which already has
    ``cozmo.geometry`` (open3d) imported by the LiDAR path sharing this module.
    open3d and torch cannot share a process -- see cozmo/semantics/worker.py
    and cozmo/recon/worker.py, both of which exist for exactly this reason.
    The ceiling-height cue is pure geometry (no torch) and always runs. Turning
    the other two on is safe only when this call is itself made from a process
    that has not imported cozmo.geometry -- e.g. a dedicated cue subprocess,
    which is not wired up yet (see README, "what is deliberately not here").
    """
    root = bundle.root
    room_folders = _room_folders(root)
    if len(room_folders) > 1:
        log.info(
            "%d room folders found; stitching them into one property plan", len(room_folders)
        )
        return build_multi_room_photo_plan(
            bundle, generated_at=generated_at, backbone_name=backbone_name,
            weights_dir=weights_dir, run_metric_depth_cue=run_metric_depth_cue,
            run_door_cue=run_door_cue, calibration=calibration,
        )

    room_dir = room_folders[0]
    room_id = (bundle.manifest.declared_rooms or [room_dir.name])[0]

    result = _reconstruct_room(
        room_dir, room_id, backbone_name=backbone_name, weights_dir=weights_dir,
        run_metric_depth_cue=run_metric_depth_cue, run_door_cue=run_door_cue,
    )
    return _assemble_single_room_plan(
        bundle, result, Tier.PHOTO, generated_at, backbone_name, calibration=calibration,
    )


def _rotate2(vec: Tuple[float, float], yaw: float) -> Tuple[float, float]:
    c, s_ = np.cos(yaw), np.sin(yaw)
    x, y = vec
    return c * x - s_ * y, s_ * x + c * y


def _group_repeated_captures(
    room_ids: List[str],
    per_room: Dict[str, RoomReconstruction],
    matches: List[RoomPairMatch],
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Collapse folders that are alternate captures of one physical room.

    The matcher already computes the two signals needed here. Doorway-only
    matches are deliberately excluded: adjacent rooms can share a doorway,
    while repeated views of the same room produce many image keypoints.
    """
    parent = {room_id: room_id for room_id in room_ids}

    def find(room_id: str) -> str:
        while parent[room_id] != room_id:
            parent[room_id] = parent[parent[room_id]]
            room_id = parent[room_id]
        return room_id

    def union(room_a: str, room_b: str) -> None:
        root_a, root_b = find(room_a), find(room_b)
        if root_a != root_b:
            parent[root_b] = root_a

    for match in matches:
        # DINO similarity drops when the camera turns toward a doorway, even
        # though LightGlue can still find a large, reliable set of points in
        # the same physical room. Keypoints are the stronger duplicate signal.
        if len(match.keypoint_matches) >= REPEATED_CAPTURE_MIN_MATCHES:
            union(match.room_a, match.room_b)

    components: Dict[str, List[str]] = {}
    for room_id in room_ids:
        components.setdefault(find(room_id), []).append(room_id)

    representative_for: Dict[str, str] = {}
    groups: Dict[str, List[str]] = {}
    for members in components.values():
        # Prefer a non-fallback layout, then the capture with the most views.
        # The final tie-breaker is stable and does not depend on dict order.
        representative = max(
            members,
            key=lambda room_id: (
                not bool(per_room[room_id].layout.stats.get("axis_fallback")),
                len(per_room[room_id].frames),
                room_id,
            ),
        )
        groups[representative] = sorted(members)
        for room_id in members:
            representative_for[room_id] = representative
    return representative_for, groups


def _capture_prefers_connected_layout(bundle: CaptureBundle, tier: Tier) -> bool:
    """Whether unconnected room folders should be laid out in capture order.

    A continuous walkthrough is evidence that consecutive segments belong to
    one connected property, even when a doorway has no stable visual match.
    An explicit ``unrelated`` note is the opt-out for test captures that
    intentionally mix independent spaces.
    """
    notes = str(getattr(bundle.manifest, "notes", "") or "").lower()
    if tier == Tier.VIDEO:
        return "unrelated" not in notes
    declared = [str(room).lower() for room in (bundle.manifest.declared_rooms or [])]
    connected_hint = any(
        marker in notes for marker in ("connected", "walkthrough", "corridor", "apartment")
    ) or "corridor" in declared
    return (
        connected_hint
        and "unrelated" not in notes
        and "not force-merged" not in notes
    )


def _infer_ordered_connections(
    room_ids: List[str],
    kept_edges: List[Tuple[Any, str, str]],
    per_room: Dict[str, RoomReconstruction],
    stitch_result: StitchResult,
) -> List[Tuple[str, str]]:
    """Place disconnected components edge-to-edge in capture/folder order.

    This is intentionally separate from evidence-backed stitch edges. It
    gives a useful connected floor-plan hypothesis for walkthroughs and
    ordered room folders, while the quality report records that the doorway
    itself was not observed strongly enough to prove the connection.
    """
    graph = {room_id: set() for room_id in room_ids}
    for _, room_a, room_b in kept_edges:
        graph[room_a].add(room_b)
        graph[room_b].add(room_a)

    components: List[List[str]] = []
    unseen = set(room_ids)
    for room_id in room_ids:
        if room_id not in unseen:
            continue
        component: List[str] = []
        stack = [room_id]
        unseen.remove(room_id)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in graph[current]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        components.append(sorted(component, key=room_ids.index))
    if len(components) <= 1:
        return []

    def room_polygon(room_id: str):
        result = per_room[room_id]
        scaled = affine_transform(
            result.layout.polygon,
            [result.scale, 0, 0, result.scale, 0, 0],
        )
        return stitch_result.transform_polygon(room_id, scaled)

    placed = unary_union([room_polygon(room_id) for room_id in components[0]])
    inferred: List[Tuple[str, str]] = []
    previous = components[0][-1]
    reference_yaw = stitch_result.global_yaw[components[0][0]]
    for component in components[1:]:
        # A capture-order connection is a low-evidence layout hypothesis. Use
        # the anchor component's Manhattan orientation so the inferred join is
        # a straight shared boundary, rather than a corner-to-corner contact
        # caused by two independently rotated rectangles.
        for room_id in component:
            stitch_result.global_yaw[room_id] = reference_yaw
        current = unary_union([room_polygon(room_id) for room_id in component])
        placed_bounds = placed.bounds
        current_bounds = current.bounds
        dx = float(placed_bounds[2] - current_bounds[0])
        dy = float(
            (placed_bounds[1] + placed_bounds[3]) / 2.0
            - (current_bounds[1] + current_bounds[3]) / 2.0
        )
        for room_id in component:
            x, y = stitch_result.global_xy[room_id]
            stitch_result.global_xy[room_id] = np.array([x + dx, y + dy])
        placed = unary_union([placed, *[room_polygon(room_id) for room_id in component]])
        current_anchor = component[0]
        inferred.append((previous, current_anchor))
        previous = component[-1]
    return inferred

def build_multi_room_photo_plan(
    bundle: CaptureBundle,
    generated_at: Optional[datetime] = None,
    backbone_name: str = DEFAULT_BACKBONE,
    weights_dir: Optional[Path] = None,
    run_metric_depth_cue: bool = False,
    run_door_cue: bool = False,
    max_frames_per_pair: int = 4,
    calibration: Optional[CalibrationSet] = None,
    tier: Tier = Tier.PHOTO,
) -> Tuple[Plan, Dict[str, Any]]:
    """Reconstruct every room folder, then stitch them into one property.

        per-room reconstruction (identical to build_photo_plan, one room at
        a time) -> cozmo.stitch.match (DINOv2 shortlist, SuperPoint+LightGlue,
        doorway width matching) -> cozmo.stitch.graph (pose graph, Manhattan
        snap, overlap resolution) -> one Plan, every room placed, Adjacency
        objects naming the connecting opening and wall.

    Each room is reconstructed at its own origin exactly as the single-room
    path does; only the final placement changes. A room the stitcher could not
    connect to anything stays at its own origin, offset so it does not
    overlap what has been placed -- present in the plan, and its isolation is
    visible in `adjacencies` rather than silently merged into a guess.
    """
    root = bundle.root
    room_folders = _room_folders(root)
    # The room id is the folder's own name, always -- never a positional pairing
    # with declared_rooms. _room_folders() returns folders sorted by path, and
    # capture.json's declared_rooms can legitimately list them in a different
    # order (the order someone typed them in); zipping the two by position once
    # mislabelled every room after the first name that happened not to sort
    # first, which is exactly the kind of silent wrong-but-confident mistake
    # this project exists to catch, not commit.
    room_ids = [folder.name for folder in room_folders]
    declared = set(bundle.manifest.declared_rooms or [])
    if declared and declared != set(room_ids):
        log.warning(
            "capture.json declares rooms %s but the folders found are %s; using folder names",
            sorted(declared), room_ids,
        )

    per_room: Dict[str, RoomReconstruction] = {}
    skipped_rooms: List[Tuple[str, str]] = []
    all_warnings: List[str] = list(bundle.warnings)
    all_degradations: List[str] = []
    timings: Dict[str, float] = {}

    for room_id, folder in zip(room_ids, room_folders):
        mark = time.time()
        try:
            result = _reconstruct_room(
                folder, room_id, backbone_name=backbone_name, weights_dir=weights_dir,
                run_metric_depth_cue=run_metric_depth_cue, run_door_cue=run_door_cue,
            )
        except (ValueError, FileNotFoundError, ReconstructorUnavailable) as exc:
            # One unusable room folder must not lose the whole property. A
            # folder holding a single photo lands here -- two views is the
            # floor for triangulating anything at all -- as does a room whose
            # geometry never closed into a polygon. The rooms that did
            # reconstruct are still worth a plan, and the one that did not is
            # named with its reason in `quality`, not silently dropped: a
            # property plan quietly missing a room reads as a property that
            # does not have that room.
            reason = f"{type(exc).__name__}: {exc}"
            log.warning("room '%s' is not in the plan: %s", room_id, reason)
            skipped_rooms.append((room_id, reason))
            all_warnings.append(f"{room_id}: not reconstructed ({reason})")
            all_degradations.append(
                f"{room_id}: excluded from this plan -- {reason}. Property totals cover only "
                f"the {len(room_folders) - len(skipped_rooms)} room(s) that did reconstruct."
            )
            timings[f"{room_id}_s"] = round(time.time() - mark, 3)
            continue
        per_room[room_id] = result
        all_warnings.extend(f"{room_id}: {w}" for w in result.warnings)
        all_degradations.extend(f"{room_id}: {d}" for d in result.degradations)
        timings[f"{room_id}_s"] = round(time.time() - mark, 3)

    if not per_room:
        raise ValueError(
            "no room folder could be reconstructed: "
            + "; ".join(f"{room} ({why})" for room, why in skipped_rooms)
        )
    # Everything below indexes per_room by id, so the skipped rooms leave here.
    room_ids = [room_id for room_id in room_ids if room_id in per_room]

    # -- match.py, in its own process (torch; this process already has open3d) --
    mark = time.time()
    match_request_rooms = []
    for room_id, result in per_room.items():
        openings_payload = [
            {"id": op.id, "kind": op.type.value, "width_m": op.width.value, "wall_id": op.wall_id}
            for op in result.room.openings
        ]
        match_request_rooms.append({
            "room_id": room_id,
            "frame_paths": [str(f.path) for f in result.frames],
            "frame_indices": [f.index for f in result.frames],
            "openings": openings_payload,
        })
    matches = _run_matcher_subprocess(match_request_rooms, weights_dir)
    timings["match_s"] = round(time.time() - mark, 3)

    # -- graph.py: build RoomForStitch from each room's own scaled reconstruction --
    mark = time.time()
    rooms_for_stitch = []
    for room_id, result in per_room.items():
        scale = result.scale
        layout = result.layout
        scaled_polygon = affine_transform(layout.polygon, [scale, 0, 0, scale, 0, 0])
        scaled_walls = [
            WallSegment(
                start=(w.start[0] * scale, w.start[1] * scale),
                end=(w.end[0] * scale, w.end[1] * scale),
                length_m=w.length_m * scale, normal=w.normal,
                top_height_m=w.top_height_m, support_points=w.support_points,
            )
            for w in layout.walls
        ]
        # Keep the clouds camera-local for keypoint reprojection. The stitch
        # graph applies each matching frame's camera-to-world pose only after
        # it has selected the nearest 3D point for that pixel.
        frame_points = {
            f.index: result.reconstruction.frame_points_local.get(f.index, np.empty((0, 3)))
            for f in result.frames
        }
        frame_K = {f.index: f.K for f in result.frames}
        frame_poses = {
            pose.frame_index: (pose.R, pose.t)
            for pose in result.reconstruction.poses
        }
        scaled_frame = FloorFrame(
            origin=layout.frame.origin * scale,
            up=layout.frame.up.copy(), e1=layout.frame.e1.copy(), e2=layout.frame.e2.copy(),
            rotation_rad=layout.frame.rotation_rad,
        )
        openings_map = {
            op.id: (int(op.wall_id.rsplit("_w", 1)[1]), op.width.value)
            for op in result.room.openings
        }
        rooms_for_stitch.append(RoomForStitch(
            room_id=room_id, frame=scaled_frame, polygon=scaled_polygon, walls=scaled_walls,
            frame_points=frame_points, frame_K=frame_K, frame_poses=frame_poses,
            metric_scale=scale, openings=openings_map,
        ))

    stitch_result = build_stitch_graph(rooms_for_stitch, matches)
    timings["graph_s"] = round(time.time() - mark, 3)

    representative_for, capture_groups = _group_repeated_captures(
        room_ids, per_room, matches,
    )
    kept_room_ids = [room_id for room_id in room_ids if representative_for[room_id] == room_id]
    merged_groups = [members for members in capture_groups.values() if len(members) > 1]
    kept_prefixes = tuple(f"{room_id}:" for room_id in kept_room_ids)
    all_degradations = [
        degradation for degradation in all_degradations
        if degradation.startswith(kept_prefixes)
    ]
    for members in merged_groups:
        representative = representative_for[members[0]]
        all_degradations.append(
            f"capture folders {', '.join(members)} describe one physical room; "
            f"represented once as {representative}"
        )

    # Edges between duplicate captures disappear. Edges from a duplicate to a
    # real neighbouring room are retained under the representative room.
    kept_edges = []
    for edge in stitch_result.edges_used:
        room_a = representative_for[edge.room_a]
        room_b = representative_for[edge.room_b]
        if room_a != room_b:
            kept_edges.append((edge, room_a, room_b))

    inferred_connections: List[Tuple[str, str]] = []
    if _capture_prefers_connected_layout(bundle, tier):
        inferred_connections = _infer_ordered_connections(
            kept_room_ids, kept_edges, per_room, stitch_result,
        )
        if inferred_connections:
            all_degradations.append(
                f"{len(inferred_connections)} room connection(s) inferred from capture order; "
                "no stable visual doorway evidence was available for every join"
            )

    # -- assemble one Plan: every room's Wall/Opening moved by its global pose --
    final_rooms: List[Room] = []
    for room_id in kept_room_ids:
        result = per_room[room_id]
        x, y = stitch_result.global_xy[room_id]
        yaw = stitch_result.global_yaw[room_id]
        source_room = result.room
        moved_walls = []
        for wall in source_room.walls:
            sx, sy = _rotate2((wall.start.x, wall.start.y), yaw)
            ex, ey = _rotate2((wall.end.x, wall.end.y), yaw)
            moved_walls.append(wall.model_copy(update={
                "start": Point2D(x=round(sx + x, 4), y=round(sy + y, 4)),
                "end": Point2D(x=round(ex + x, 4), y=round(ey + y, 4)),
            }))
        moved_room = source_room.model_copy(update={
            "walls": moved_walls,
            "pose": Pose2D(x=round(x, 4), y=round(y, 4), theta_rad=round(source_room.pose.theta_rad + yaw, 5)),
        })
        final_rooms.append(recalibrate_room(moved_room, tier.value, calibration))

    adjacencies = [
        Adjacency(
            room_a_id=room_a, room_b_id=room_b,
            via_opening_id=edge.via_opening_a if edge.room_a == room_a else edge.via_opening_b,
            shared_wall_ids=(edge.wall_a, edge.wall_b),
            confidence=round(min(1.0, edge.confidence / 4.0), 3),
        )
        for edge, room_a, room_b in kept_edges
    ]
    adjacencies.extend(
        Adjacency(
            room_a_id=room_a, room_b_id=room_b,
            via_opening_id=None, shared_wall_ids=(None, None), confidence=0.15,
        )
        for room_a, room_b in inferred_connections
    )

    total_floor_area = sum(r.floor_area.value for r in final_rooms)
    # Union, not sum: overlap resolution already guarantees ~zero intersection,
    # but the footprint is defined as the property's own outline, not an
    # assumption that summing room areas equals it.
    footprint_area = float(unary_union([
        stitch_result.transform_polygon(
            r_id,
            affine_transform(per_room[r_id].layout.polygon,
                             [per_room[r_id].scale, 0, 0, per_room[r_id].scale, 0, 0]),
        )
        for r_id in kept_room_ids
    ]).area)

    mean_scale_rel = float(np.mean([
        max(abs(r.scale_estimate.ci_95[1] - r.scale) / r.scale,
            abs(r.scale - r.scale_estimate.ci_95[0]) / r.scale) if r.scale > 0 else 0.3
        for r in per_room.values()
    ]))
    footprint_rel = _combine_relative(max(AREA_REL_FLOOR, 2 * mean_scale_rel), 0.05)

    if not stitch_result.overlap_resolved:
        all_degradations.append(
            f"overlap resolution did not fully converge after {stitch_result.overlap_iterations} "
            f"iteration(s); some rooms may still touch"
        )
    if stitch_result.edges_rejected:
        for room_a, room_b, reason in stitch_result.edges_rejected:
            if inferred_connections and room_a in kept_room_ids and room_b in kept_room_ids:
                continue
            all_warnings.append(f"{room_a} <-> {room_b}: not connected ({reason})")

    all_connections = kept_edges + [(None, room_a, room_b) for room_a, room_b in inferred_connections]
    unplaced = set(kept_room_ids) - {room_a for _, room_a, _ in all_connections} - {room_b for _, _, room_b in all_connections}
    if len(kept_room_ids) > 1 and unplaced == set(kept_room_ids):
        all_degradations.append(
            "no retained room pair matched; independent rooms remain separated rather than being force-merged"
        )

    mean_confidence = float(np.mean([r.scale_estimate.agreement for r in per_room.values()])) if per_room else 0.0
    plan = Plan(
        schema_version=SCHEMA_VERSION, capture_id=bundle.capture_id, tier=tier,
        pipeline_version=PIPELINE_VERSION, generated_at=generated_at or datetime.now(timezone.utc),
        scale=ScaleInfo(
            source=ScaleSource.STRUCTURAL_PRIOR,
            scale_factor=Measurement(value=1.0, ci_95=(1 - mean_scale_rel, 1 + mean_scale_rel), unit=Unit.RATIO),
            reference_description=(
                f"per-room scale, {len(per_room)} room(s); each room's own cues -- see "
                f"per-room detail in the manifest"
            ),
        ),
        drift_correction=DriftCorrection(
            enabled=True, method=DriftMethod.PLANE_ANCHORED, loop_closures=0,
            notes=(
                f"{len(all_connections)} of {len(kept_room_ids) - 1} needed connection(s) made "
                f"({sum(1 for e, _, _ in kept_edges if 'keypoints' in e.source)} from image "
                f"matches, {sum(1 for e, _, _ in kept_edges if e.source == 'doorway')} from "
                f"doorway width alone); property snapped to a shared Manhattan frame "
                f"({np.degrees(stitch_result.manhattan_rotation_rad):.1f} deg); overlap resolution "
                f"{'converged' if stitch_result.overlap_resolved else 'did not fully converge'} "
                f"after {stitch_result.overlap_iterations} step(s)."
            ),
        ),
        property_totals=PropertyTotals(
            room_count=len(final_rooms),
            total_floor_area=recalibrate_measurement(
                Measurement.relative(round(total_floor_area, 4), footprint_rel, Unit.SQUARE_METERS),
                tier.value, "floor_area", calibration,
            ),
            footprint_area=recalibrate_measurement(
                Measurement.relative(round(footprint_area, 4), footprint_rel, Unit.SQUARE_METERS),
                tier.value, "footprint_area", calibration,
            ),
        ),
        rooms=final_rooms, adjacencies=adjacencies, damage=[], concealed_flags=[], scope=[],
        quality=QualityReport(
            overall_confidence=round(min(0.7, 0.3 + 0.15 * len(kept_edges)), 3),
            interval_method=(
                "Multi-room photo tier: per-room intervals as in the single-room case, footprint "
                "additionally carries each room's own scale uncertainty, then this tier's fitted "
                "calibration factor. Adjacency confidence comes from match evidence (image inlier "
                "count, doorway width agreement), not calibration."
            ),
            calibration_note=(
                calibration_note_for(tier.value, calibration)
                + " Stitching additionally adds pose-graph and overlap-resolution error on top of "
                  "each room's own calibrated reconstruction, which calibration does not separately account for."
            ),
            ceiling_method="multi_room_stitch",
            semantics_available=any(
                per_room[room_id].semantic_openings_available for room_id in kept_room_ids
            ),
            degradations=all_degradations,
            warnings=all_warnings,
            coverage={
                "rooms_placed": float(len(final_rooms)),
                "edges_used": float(len(all_connections)),
                "edges_inferred": float(len(inferred_connections)),
                "edges_rejected": float(len(stitch_result.edges_rejected)),
            },
        ),
    )

    details = {
        "layout_result": None,
        "scale_factor": 1.0,
        "backbone": backbone_name,
        "rooms": {r_id: {
            "layout_method": res.layout.layout_method, "scale": res.scale_estimate.as_dict(),
            "timings_s": res.timings,
        } for r_id, res in per_room.items()},
        "skipped_rooms": [{"room_id": room, "reason": why} for room, why in skipped_rooms],
        "capture_groups": capture_groups,
        "representative_for": representative_for,
        "inferred_connections": [
            {"room_a": room_a, "room_b": room_b, "source": "capture_order_inferred"}
            for room_a, room_b in inferred_connections
        ],
        "matches": [m.as_dict() for m in matches],
        "edges_used": [
            {"room_a": e.room_a, "room_b": e.room_b, "source": e.source,
             "confidence": round(e.confidence, 3), "via_opening_a": e.via_opening_a,
             "via_opening_b": e.via_opening_b}
            for e, room_a, room_b in kept_edges
        ],
        "edges_rejected": stitch_result.edges_rejected,
        "manhattan_rotation_deg": round(float(np.degrees(stitch_result.manhattan_rotation_rad)), 2),
        "overlap_resolved": stitch_result.overlap_resolved,
        "timings_s": timings,
        "multi_room_plan": True,
        "per_room_reconstructions": per_room,   # not serialised; used by the caller for rendering
    }
    return plan, details


def _run_matcher_subprocess(
    rooms: List[Dict[str, Any]], weights_dir: Optional[Path],
) -> List[RoomPairMatch]:
    """Dispatch to cozmo.stitch.worker -- see its docstring for why."""
    import json
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory(prefix="cozmo-stitch-") as scratch:
        request = {"rooms": rooms, "weights_dir": str(weights_dir) if weights_dir else None}
        request_path = Path(scratch) / "request.json"
        response_path = Path(scratch) / "response.json"
        request_path.write_text(json.dumps(request))

        command = [sys.executable, "-m", "cozmo.stitch.worker", str(request_path), str(response_path)]
        log.info("running the room matcher in a child process")
        completed = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)

        if not response_path.is_file():
            log.warning(
                "stitch matcher produced no response (exit %s): %s",
                completed.returncode, (completed.stderr or "").strip()[-800:],
            )
            return [RoomPairMatch(r_a["room_id"], r_b["room_id"], 0.0)
                    for r_a, r_b in itertools.combinations(rooms, 2)]

        payload = json.loads(response_path.read_text())

    if not payload.get("ok"):
        log.warning("stitch matcher failed: %s", payload.get("error"))
        return [RoomPairMatch(r_a["room_id"], r_b["room_id"], 0.0)
                for r_a, r_b in itertools.combinations(rooms, 2)]

    from ..stitch.match import DoorwayMatch, KeypointMatch

    results = []
    for m in payload["matches"]:
        results.append(RoomPairMatch(
            room_a=m["room_a"], room_b=m["room_b"], dinov2_similarity=m["dinov2_similarity"],
            keypoint_matches=[
                KeypointMatch(k["frame_index_a"], k["frame_index_b"], k["u_a"], k["v_a"],
                              k["u_b"], k["v_b"], k["score"])
                for k in m["keypoint_matches"]
            ],
            doorway_matches=[
                DoorwayMatch(d["opening_id_a"], d["opening_id_b"], d["width_a_m"], d["width_b_m"])
                for d in m["doorway_matches"]
            ],
            best_frame_pair=tuple(m["best_frame_pair"]) if m.get("best_frame_pair") else None,
        ))
    return results


def _room_folders(root: Path) -> List[Path]:
    """Where the photo tier's room folders live: rooms/<room>/, or the root
    itself if photos are directly inside it (single-room capture, no
    subfolder)."""
    rooms_dir = root / "rooms"
    if rooms_dir.is_dir():
        folders = sorted(p for p in rooms_dir.iterdir() if p.is_dir())
        if folders:
            return folders
    image_suffixes = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
    if any(p.suffix.lower() in image_suffixes for p in root.iterdir() if p.is_file()):
        return [root]
    folders = sorted(p for p in root.iterdir() if p.is_dir())
    if folders:
        return folders
    raise FileNotFoundError(f"no room photo folders found under {root}")
