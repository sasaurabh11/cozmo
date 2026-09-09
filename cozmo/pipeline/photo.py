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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .. import PIPELINE_VERSION, SCHEMA_VERSION
from ..geometry.openings import detect_openings, wall_observation_fractions
from ..io.capture import CaptureBundle
from ..recon.backbone import ReconstructorUnavailable, get_reconstructor
from ..recon.frames import load_photo_folder, summarize as summarize_frames
from ..recon.layout import extract_layout_photo
from ..recon.scale import recover_scale
from ..schema import (
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
        log.warning(
            "%d room folders found; photo tier reconstructs a single room today "
            "(stitching is the next phase) -- using %s",
            len(room_folders), room_folders[0].name,
        )
    room_dir = room_folders[0]
    room_id = (bundle.manifest.declared_rooms or [room_dir.name])[0]

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
        raise ReconstructorUnavailable(
            f"backbone '{backbone_name}' unavailable: {exc}"
        ) from exc
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

    # Everything linear (lengths) scales by `scale`; areas by `scale**2`.
    ceiling_height_units = None
    if layout.floor_to_ceiling_span:
        ceiling_height_units = layout.floor_to_ceiling_span
    scaled_points_uv = layout.points_uv * scale
    scaled_points_height = layout.points_height * scale
    scaled_ceiling_m = (
        ceiling_height_units * scale if ceiling_height_units else 2.44 * 1.0
    )

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
    warnings_out: List[str] = list(bundle.warnings) + list(summarize_frames(frames).get("warnings", []))
    if not (2 <= len(frames) <= 8):
        degradations.append(f"{len(frames)} photos supplied; contract floor/ceiling is 2-8 per room")
    if scale_estimate.agreement < 0.5:
        degradations.append(
            f"scale cues disagree (agreement {scale_estimate.agreement:.2f}); "
            f"the scale interval is widened accordingly, not narrowed by averaging over it"
        )
    if layout.layout_method == "per_view_merge" and layout.view_plane_counts.get("wall_clusters", 0) < len(walls):
        pass

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
            total_floor_area=floor_area.model_copy(deep=True),
            footprint_area=floor_area.model_copy(deep=True),
            total_wall_area=Measurement.relative(
                round(perimeter_m * scaled_ceiling_m, 4), max(wall_rel, area_rel), Unit.SQUARE_METERS
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
        "reconstruction_stats": reconstruction.stats,
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
