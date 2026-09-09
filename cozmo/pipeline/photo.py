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
from ..geometry.openings import detect_openings, wall_observation_fractions
from ..io.capture import CaptureBundle
from ..recon.backbone import ReconstructorUnavailable, get_reconstructor
from ..recon.frames import load_photo_folder, summarize as summarize_frames
from ..geometry.layout import WallSegment
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
    observation = wall_observation_fractions(
        scaled_points_uv, scaled_points_height, layout.walls, scaled_ceiling_m
    )
    timings["openings_s"] = round(time.time() - mark, 3)

    walls: List[Wall] = []
    surfaces: List[Surface] = []
    openings: List[Opening] = []
    by_wall: Dict[int, List[Any]] = {}
    for detection in detections:
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
                detection_sources=["geometry"],
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
    if not (2 <= len(frames) <= 8):
        degradations.append(f"{len(frames)} photos supplied; contract floor/ceiling is 2-8 per room")
    if scale_estimate.agreement < 0.5:
        degradations.append(
            f"scale cues disagree (agreement {scale_estimate.agreement:.2f}); "
            f"the scale interval is widened accordingly, not narrowed by averaging over it"
        )

    return RoomReconstruction(
        room_id=room_id, room=room, layout=layout, reconstruction=reconstruction,
        scale_estimate=scale_estimate, scale=scale, frames=frames, detections=detections,
        timings=timings, warnings=warnings_out, degradations=degradations,
    )


def build_photo_plan(
    bundle: CaptureBundle,
    generated_at: Optional[datetime] = None,
    backbone_name: str = DEFAULT_BACKBONE,
    weights_dir: Optional[Path] = None,
    run_metric_depth_cue: bool = False,
    run_door_cue: bool = False,
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
            run_door_cue=run_door_cue,
        )

    room_dir = room_folders[0]
    room_id = (bundle.manifest.declared_rooms or [room_dir.name])[0]

    result = _reconstruct_room(
        room_dir, room_id, backbone_name=backbone_name, weights_dir=weights_dir,
        run_metric_depth_cue=run_metric_depth_cue, run_door_cue=run_door_cue,
    )
    layout, scale, frames = result.layout, result.scale, result.frames
    scale_estimate, room, detections = result.scale_estimate, result.room, result.detections
    warnings_out = list(bundle.warnings) + result.warnings
    degradations = result.degradations
    timings = result.timings

    drift = DriftCorrection(
        enabled=True, method=DriftMethod.PLANE_ANCHORED, loop_closures=0,
        notes=(
            f"Single-room photo reconstruction ({layout.layout_method}); no multi-room stitch, "
            f"so no cross-room drift to correct. Per-wall confidence comes from view count and "
            f"cross-view plane agreement instead of a pose graph residual."
        ),
    )

    fired_cues = [c.name for c in scale_estimate.cues if c.fired]
    plan = Plan(
        schema_version=SCHEMA_VERSION, capture_id=bundle.capture_id, tier=Tier.PHOTO,
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
            total_floor_area=room.floor_area.model_copy(deep=True),
            footprint_area=room.floor_area.model_copy(deep=True),
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
                "Photo tier: per-wall relative uncertainty from view count and cross-view plane "
                "agreement, combined in quadrature with the scale factor's own interval (cue "
                "disagreement). No sensor-noise error budget -- there is no depth sensor at this tier."
            ),
            calibration_note=(
                "Uncalibrated. Widened deliberately by cue disagreement rather than narrowed by "
                "averaging over it; not yet checked against benchmark ground truth."
            ),
            ceiling_method=layout.layout_method,
            semantics_available=False,
            degradations=degradations,
            warnings=warnings_out,
            coverage={
                "layout_method": 1.0 if layout.layout_method == "per_view_merge" else 0.0,
                "scale_cues_fired": float(len(fired_cues)),
                "scale_agreement": float(scale_estimate.agreement),
            },
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
            {"wall_index": d.wall_index, "kind": d.kind, "width_m": round(d.width_m * scale, 3),
             "confidence": d.confidence}
            for d in detections
        ],
        "timings_s": timings,
    }
    return plan, details


def _rotate2(vec: Tuple[float, float], yaw: float) -> Tuple[float, float]:
    c, s_ = np.cos(yaw), np.sin(yaw)
    x, y = vec
    return c * x - s_ * y, s_ * x + c * y


def build_multi_room_photo_plan(
    bundle: CaptureBundle,
    generated_at: Optional[datetime] = None,
    backbone_name: str = DEFAULT_BACKBONE,
    weights_dir: Optional[Path] = None,
    run_metric_depth_cue: bool = False,
    run_door_cue: bool = False,
    max_frames_per_pair: int = 4,
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
    all_warnings: List[str] = list(bundle.warnings)
    all_degradations: List[str] = []
    timings: Dict[str, float] = {}

    for room_id, folder in zip(room_ids, room_folders):
        mark = time.time()
        result = _reconstruct_room(
            folder, room_id, backbone_name=backbone_name, weights_dir=weights_dir,
            run_metric_depth_cue=run_metric_depth_cue, run_door_cue=run_door_cue,
        )
        per_room[room_id] = result
        all_warnings.extend(f"{room_id}: {w}" for w in result.warnings)
        all_degradations.extend(f"{room_id}: {d}" for d in result.degradations)
        timings[f"{room_id}_s"] = round(time.time() - mark, 3)

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
        frame_points = {
            f.index: (result.reconstruction.frame_points_local.get(f.index, np.empty((0, 3))) * scale)
            for f in result.frames
        }
        frame_K = {f.index: f.K for f in result.frames}
        openings_map = {
            op.id: (int(op.wall_id.rsplit("_w", 1)[1]), op.width.value)
            for op in result.room.openings
        }
        rooms_for_stitch.append(RoomForStitch(
            room_id=room_id, frame=layout.frame, polygon=scaled_polygon, walls=scaled_walls,
            frame_points=frame_points, frame_K=frame_K, openings=openings_map,
        ))

    stitch_result = build_stitch_graph(rooms_for_stitch, matches)
    timings["graph_s"] = round(time.time() - mark, 3)

    # -- assemble one Plan: every room's Wall/Opening moved by its global pose --
    final_rooms: List[Room] = []
    for room_id, result in per_room.items():
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
        final_rooms.append(source_room.model_copy(update={
            "walls": moved_walls,
            "pose": Pose2D(x=round(x, 4), y=round(y, 4), theta_rad=round(source_room.pose.theta_rad + yaw, 5)),
        }))

    adjacencies = [
        Adjacency(
            room_a_id=edge.room_a, room_b_id=edge.room_b,
            via_opening_id=edge.via_opening_a,
            shared_wall_ids=(edge.wall_a, edge.wall_b),
            confidence=round(min(1.0, edge.confidence / 4.0), 3),
        )
        for edge in stitch_result.edges_used
    ]

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
        for r_id in room_ids
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
            all_warnings.append(f"{room_a} <-> {room_b}: not connected ({reason})")

    unplaced = set(room_ids) - {e.room_a for e in stitch_result.edges_used} - {e.room_b for e in stitch_result.edges_used}
    if len(room_ids) > 1 and unplaced == set(room_ids):
        all_degradations.append(
            "no room pair matched at all; every room sits at an independent origin, not a stitched property"
        )

    mean_confidence = float(np.mean([r.scale_estimate.agreement for r in per_room.values()])) if per_room else 0.0
    plan = Plan(
        schema_version=SCHEMA_VERSION, capture_id=bundle.capture_id, tier=Tier.PHOTO,
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
                f"{len(stitch_result.edges_used)} of {len(room_ids) - 1} needed connection(s) made "
                f"({sum(1 for e in stitch_result.edges_used if 'keypoints' in e.source)} from image "
                f"matches, {sum(1 for e in stitch_result.edges_used if e.source == 'doorway')} from "
                f"doorway width alone); property snapped to a shared Manhattan frame "
                f"({np.degrees(stitch_result.manhattan_rotation_rad):.1f} deg); overlap resolution "
                f"{'converged' if stitch_result.overlap_resolved else 'did not fully converge'} "
                f"after {stitch_result.overlap_iterations} step(s)."
            ),
        ),
        property_totals=PropertyTotals(
            room_count=len(final_rooms),
            total_floor_area=Measurement.relative(round(total_floor_area, 4), footprint_rel, Unit.SQUARE_METERS),
            footprint_area=Measurement.relative(round(footprint_area, 4), footprint_rel, Unit.SQUARE_METERS),
        ),
        rooms=final_rooms, adjacencies=adjacencies, damage=[], concealed_flags=[], scope=[],
        quality=QualityReport(
            overall_confidence=round(min(0.7, 0.3 + 0.15 * len(stitch_result.edges_used)), 3),
            interval_method=(
                "Multi-room photo tier: per-room intervals as in the single-room case, footprint "
                "additionally carries each room's own scale uncertainty. Adjacency confidence comes "
                "from match evidence (image inlier count, doorway width agreement), not calibration."
            ),
            calibration_note="Uncalibrated. Stitching adds pose-graph and overlap-resolution error on top "
                             "of each room's own uncalibrated reconstruction.",
            ceiling_method="multi_room_stitch",
            semantics_available=False,
            degradations=all_degradations,
            warnings=all_warnings,
            coverage={
                "rooms_placed": float(len(final_rooms)),
                "edges_used": float(len(stitch_result.edges_used)),
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
        "matches": [m.as_dict() for m in matches],
        "edges_used": [
            {"room_a": e.room_a, "room_b": e.room_b, "source": e.source,
             "confidence": round(e.confidence, 3), "via_opening_a": e.via_opening_a,
             "via_opening_b": e.via_opening_b}
            for e in stitch_result.edges_used
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
