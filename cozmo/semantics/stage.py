"""The semantic stage: detection, projection, rules, scope.

Tier-agnostic by construction: it needs fitted surfaces, RGB frames with poses
and intrinsics, and the geometric opening detections to cross-check against.
Any tier that can supply those gets damage, concealed flags and scope out of it
unchanged.

**This module must not import cozmo.geometry.** open3d and torch each ship
their own OpenMP runtime, and loading both into one process on macOS either
crashes (``OMP: Error #179``) or deadlocks inside inference. The stage therefore
runs in a separate process -- see :mod:`cozmo.semantics.worker` -- and everything
it needs from geometry arrives as data.

Failure here degrades the plan rather than ending the run. A capture with no
RGB, or a machine with no weights, still produces geometry, with the reason
recorded in ``quality.warnings`` so nobody mistakes an absent detector for an
undamaged property.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..schema import (
    ConcealedFlag,
    DamageClass,
    DamageRegion,
    Measurement,
    Point2D,
    ScopeItem,
    Unit,
)
from .detect import (
    DetectorUnavailable,
    OpenVocabularyDetector,
    OpeningConsensus,
    cross_check_openings,
    detect_damage,
    detect_openings,
)
from .project import (
    ProjectedRegion,
    SurfacePlane,
    merge_regions,
    project_mask_to_surface,
    surfaces_from_layout,
)
from .rules import RuleEngine, RuleFiring
from .scope import ScopeCatalogue, load_catalogue, price_regions

log = logging.getLogger("cozmo.pipeline.semantics")

DEFAULT_FRAME_STRIDE = 60
DEFAULT_DETECT_LONG_EDGE = 960
DEFAULT_MAX_FRAMES = 40


@dataclass
class SemanticResult:
    damage: List[DamageRegion] = field(default_factory=list)
    concealed_flags: List[ConcealedFlag] = field(default_factory=list)
    scope: List[ScopeItem] = field(default_factory=list)
    openings: List[OpeningConsensus] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    available: bool = False


def _damage_class(value: str) -> Optional[DamageClass]:
    try:
        return DamageClass(value)
    except ValueError:
        return None


def _exterior_corner_distance(region: ProjectedRegion, surfaces: Sequence[SurfacePlane]) -> Optional[float]:
    """Distance from a ceiling region to the nearest wall/ceiling junction.

    Approximated as the distance from the region to the edge of the surface it
    sits on, which for a ceiling is exactly where it meets a wall. Rules that
    ask about exterior corners are asking about that junction.
    """
    if region.surface_kind != "ceiling":
        return None
    surface = next((s for s in surfaces if s.surface_id == region.surface_id), None)
    if surface is None:
        return None
    u0, v0, u1, v1 = region.bbox
    return float(min(u0, v0, max(0.0, surface.u_extent - u1), max(0.0, surface.v_extent - v1)))


def run_semantics(
    lidar_capture: Any,
    surfaces: Sequence[SurfacePlane],
    ceiling_height_m: float,
    geometric_openings: Sequence[Any],
    room_id: str,
    wall_count: int,
    weights_dir: Optional[Any] = None,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    max_frames: int = DEFAULT_MAX_FRAMES,
    long_edge: int = DEFAULT_DETECT_LONG_EDGE,
    rule_engine: Optional[RuleEngine] = None,
    catalogue: Optional[ScopeCatalogue] = None,
) -> SemanticResult:
    """Detect, project, apply rules, and price. Never raises for a missing model."""
    from ..io.lidar import iter_rgb_frames

    result = SemanticResult()
    timings: Dict[str, float] = {}

    wall_index_by_id = {f"{room_id}_w{i}": i for i in range(wall_count)}

    try:
        mark = time.time()
        detector = OpenVocabularyDetector(weights_dir=weights_dir)
        timings["load_models_s"] = round(time.time() - mark, 3)
    except DetectorUnavailable as exc:
        message = (
            f"semantic detection unavailable ({exc}); this plan carries geometry only. "
            f"Absence of damage here is absence of a detector, not absence of damage."
        )
        log.warning(message)
        result.warnings.append(message)
        result.details = {"detector": "unavailable", "reason": str(exc)}
        # Openings still pass through the cross-check, so the plan records that
        # only one source was available rather than implying both agreed.
        result.openings = cross_check_openings(geometric_openings, [], wall_index_by_id)
        return result

    damage_observations: List[ProjectedRegion] = []
    opening_observations: List[ProjectedRegion] = []
    frames_used = 0
    raw_damage = 0
    raw_openings = 0

    mark = time.time()
    for index, rgb, R, t, K in iter_rgb_frames(
        lidar_capture, stride=frame_stride, max_frames=max_frames, long_edge=long_edge
    ):
        frames_used += 1

        for detection in detect_damage(detector, rgb, index):
            raw_damage += 1
            damage_class = detection.damage_class
            if damage_class is None or detection.mask is None:
                continue
            region = project_mask_to_surface(
                detection.mask, K, R, t, surfaces,
                label=detection.prompt, damage_class=damage_class,
                score=detection.score, frame_index=index,
            )
            if region is not None:
                damage_observations.append(region)

        for detection in detect_openings(detector, rgb, index):
            raw_openings += 1
            if detection.mask is None:
                continue
            region = project_mask_to_surface(
                detection.mask, K, R, t, surfaces,
                label=detection.prompt, damage_class="",
                score=detection.score, frame_index=index,
            )
            if region is not None and region.surface_kind == "wall":
                opening_observations.append(region)
    timings["detect_s"] = round(time.time() - mark, 3)

    merged_damage = merge_regions(damage_observations)
    merged_openings = merge_regions(opening_observations)

    # -- damage regions ----------------------------------------------------
    contexts: List[Dict[str, Any]] = []
    surface_areas = {s.surface_id: s.area_m2 for s in surfaces}
    surface_by_id = {s.surface_id: s for s in surfaces}

    for number, region in enumerate(merged_damage):
        damage_class = _damage_class(region.damage_class)
        if damage_class is None:
            continue
        region_id = f"dmg_{region.surface_id}_{damage_class.value}_{number:02d}"
        surface = surface_by_id.get(region.surface_id)

        result.damage.append(DamageRegion(
            id=region_id,
            room_id=region.room_id,
            surface_id=region.surface_id,
            damage_class=damage_class,
            polygon=[Point2D(x=round(u, 4), y=round(v, 4)) for u, v in region.polygon_uv()],
            area=Measurement(
                value=round(region.area_m2, 4),
                ci_95=tuple(round(v, 4) for v in region.area_ci_95()),
                unit=Unit.SQUARE_METERS,
            ),
            max_extent=Measurement.symmetric(round(region.max_extent_m, 4), 0.05),
            severity=round(min(1.0, 0.3 + region.area_m2), 3),
            confidence=round(min(0.99, region.score), 3),
            evidence_frames=[f"frame_{i:06d}" for i in region.frames],
        ))

        contexts.append(region.as_context(region_id, extras={
            "distance_to_exterior_corner_m": _exterior_corner_distance(region, surfaces),
            "wall_has_opening": bool(surface.has_opening) if surface else False,
            "is_exterior_surface": surface.is_exterior if surface else None,
            "distance_to_opening_m": None,
        }))

    # -- concealed-damage rules -------------------------------------------
    engine = rule_engine or RuleEngine()
    firings: List[RuleFiring] = engine.evaluate(contexts)
    for number, firing in enumerate(firings):
        result.concealed_flags.append(ConcealedFlag(
            id=f"cf_{firing.rule.id.lower().replace('-', '_')}_{number:02d}",
            room_id=firing.room_id,
            surface_id=firing.surface_id,
            rule_id=firing.rule.id,
            rule_text=firing.rule.text,
            triggered_by_damage_ids=[firing.region_id],
            triggering_values={
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in firing.triggering_values().items()
            },
            severity=firing.rule.severity,
            probability=firing.rule.probability,
            recommended_action=firing.rule.recommended_action,
            inspection_priority=firing.rule.inspection_priority,
        ))

    # -- scope -------------------------------------------------------------
    catalogue = catalogue or load_catalogue()
    for number, line in enumerate(price_regions(contexts, catalogue, surface_areas)):
        result.scope.append(ScopeItem(
            id=f"scope_{number + 1:03d}",
            room_id=line.room_id,
            surface_id=line.surface_id,
            damage_region_ids=line.damage_region_ids,
            code=line.code,
            description=line.description,
            quantity=Measurement(
                value=round(line.quantity, 4),
                ci_95=(round(line.ci_95[0], 4), round(line.ci_95[1], 4)),
                unit=Unit.SQUARE_METERS if line.unit == "m2"
                else Unit.EACH if line.unit == "ea" else Unit.METERS,
            ),
            basis=line.basis,
            notes=line.notes or None,
        ))

    # -- openings: geometry and the detector, cross-checked -----------------
    result.openings = cross_check_openings(geometric_openings, merged_openings, wall_index_by_id)

    result.available = True
    result.details = {
        "detector": "grounding-dino-tiny + sam2.1-hiera-tiny",
        "device": detector.device,
        "frames_sampled": frames_used,
        "frame_stride": frame_stride,
        "detect_long_edge": long_edge,
        "raw_damage_detections": raw_damage,
        "raw_opening_detections": raw_openings,
        "rejected_implausible": detector.rejected,
        "damage_observations_projected": len(damage_observations),
        "damage_regions_merged": len(merged_damage),
        "opening_observations_projected": len(opening_observations),
        "rule_firings": [f.as_dict() for f in firings],
        "surfaces": len(surfaces),
        "timings_s": timings,
    }
    if not merged_damage:
        result.warnings.append(
            f"no damage regions survived projection from {raw_damage} raw detection(s) "
            f"across {frames_used} frame(s)"
        )
    return result
