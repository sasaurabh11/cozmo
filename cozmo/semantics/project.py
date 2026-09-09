"""Masks in image space to regions on surfaces, in metres.

A detector says "these pixels look like a water stain". That is not yet a
finding: the contract needs a metric extent on a named surface. This module
turns one into the other by intersecting the camera ray through each mask pixel
with the fitted surface planes, and measuring the result where it lands.

Two details do most of the work:

* **Area is counted, not hulled.** Occupancy cells on the surface, summed. A
  convex hull over an L-shaped stain invents area that was never wet.
* **Cells are indexed in the surface's own frame**, so the same stain seen from
  three frames produces three cell sets that can simply be unioned. Merging by
  bounding-box overlap would double-count the middle and lose the edges.

The intrinsics passed here must match the resolution of the mask, not the
resolution of the capture: detection usually runs on a downscaled frame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .._numeric import quiet_fp

log = logging.getLogger("cozmo.semantics.project")

# Surface occupancy cell. 2 cm keeps a 0.3 m stain to ~200 cells: fine enough to
# follow an irregular edge, coarse enough that a sparse mask stays connected.
CELL_M = 0.02

# A ray must strike the plane in front of the camera and within this range.
MIN_RANGE_M = 0.20
MAX_RANGE_M = 6.00

# Rays nearly parallel to a surface carry no information about where they land.
MIN_INCIDENCE_COS = 0.10

# Cap on mask pixels traced per frame, for cost. Sampling is deterministic.
MAX_PIXELS_PER_MASK = 4000

# Fraction of a mask's rays that must land on one surface for the hit to count.
MIN_SURFACE_SHARE = 0.35


@dataclass
class SurfacePlane:
    """A fitted surface with its own 2D frame.

    For walls ``v_axis`` points up, so a region's ``v`` coordinate is its height
    above the floor and the concealed-damage rules can ask about it directly.
    """

    surface_id: str
    room_id: str
    kind: str                       # wall | floor | ceiling
    origin: np.ndarray              # a point on the plane, at the wall's start and floor level
    normal: np.ndarray              # unit normal, pointing into the room
    u_axis: np.ndarray              # unit, along the surface
    v_axis: np.ndarray              # unit, up for walls
    u_extent: float
    v_extent: float
    area_m2: float
    wall_id: Optional[str] = None
    has_opening: bool = False
    is_exterior: Optional[bool] = None

    def to_local(self, points: np.ndarray) -> np.ndarray:
        rel = points - self.origin
        with quiet_fp():
            return np.stack([rel @ self.u_axis, rel @ self.v_axis], axis=1)

    def contains(self, uv: np.ndarray, margin: float = 0.05) -> np.ndarray:
        return (
            (uv[:, 0] >= -margin) & (uv[:, 0] <= self.u_extent + margin)
            & (uv[:, 1] >= -margin) & (uv[:, 1] <= self.v_extent + margin)
        )


@dataclass
class ProjectedRegion:
    """A damage region on a surface, measured in metres."""

    surface_id: str
    room_id: str
    surface_kind: str
    label: str
    damage_class: str
    cells: Set[Tuple[int, int]] = field(default_factory=set, repr=False)
    score: float = 0.0
    frames: List[int] = field(default_factory=list)
    pixel_count: int = 0
    wall_id: Optional[str] = None

    @property
    def area_m2(self) -> float:
        return len(self.cells) * CELL_M * CELL_M

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        if not self.cells:
            return (0.0, 0.0, 0.0, 0.0)
        us = [c[0] for c in self.cells]
        vs = [c[1] for c in self.cells]
        return (min(us) * CELL_M, min(vs) * CELL_M,
                (max(us) + 1) * CELL_M, (max(vs) + 1) * CELL_M)

    @property
    def bbox_width_m(self) -> float:
        u0, _v0, u1, _v1 = self.bbox
        return u1 - u0

    @property
    def bbox_height_m(self) -> float:
        _u0, v0, _u1, v1 = self.bbox
        return v1 - v0

    @property
    def max_extent_m(self) -> float:
        return float(np.hypot(self.bbox_width_m, self.bbox_height_m))

    @property
    def min_height_above_floor_m(self) -> float:
        """Height of the region's lowest point above the floor, never negative.

        The raw value can go slightly below zero: a mask that reaches the floor
        projects a few centimetres past it, and the 2 cm cell index rounds
        outward. A region 6 cm below the floor is not a finding, it is
        quantisation, and reporting it as one makes the rules look broken.
        """
        return max(0.0, self.bbox[1])

    @property
    def max_height_above_floor_m(self) -> float:
        return self.bbox[3]

    def polygon_uv(self) -> List[Tuple[float, float]]:
        """Outline for the plan: the cell set's boundary, as a rectangle ring.

        A full concave outline is not worth the complexity yet -- the area is
        already counted from cells, and this is only what gets drawn.
        """
        u0, v0, u1, v1 = self.bbox
        return [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]

    def area_ci_95(self) -> Tuple[float, float]:
        """Interval on the area.

        Placeholder, and a coarse one: the dominant error is the mask boundary,
        so the interval is the area with the boundary cells added and removed.
        A boundary cell count is approximated from the perimeter of the bbox.
        """
        area = self.area_m2
        perimeter = 2 * (self.bbox_width_m + self.bbox_height_m)
        boundary = perimeter * CELL_M
        return (max(0.0, area - boundary), area + boundary)

    def as_context(self, region_id: str, extras: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """The flat dict the rule engine and the scope catalogue consume."""
        context = {
            "region_id": region_id,
            "room_id": self.room_id,
            "surface_id": self.surface_id,
            "surface_kind": self.surface_kind,
            "damage_class": self.damage_class,
            "label": self.label,
            "area_m2": self.area_m2,
            "area_ci_95": self.area_ci_95(),
            "max_extent_m": self.max_extent_m,
            "bbox_width_m": self.bbox_width_m,
            "bbox_height_m": self.bbox_height_m,
            "min_height_above_floor_m": self.min_height_above_floor_m,
            "max_height_above_floor_m": self.max_height_above_floor_m,
            "confidence": self.score,
            "severity": min(1.0, 0.3 + self.area_m2),
            "frames": list(self.frames),
        }
        if extras:
            context.update(extras)
        return context


def _ray_directions(pixels: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pixel coordinates (N, 2) to camera-frame ray directions (N, 3)."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    return np.stack([
        (pixels[:, 0] - cx) / fx,
        (pixels[:, 1] - cy) / fy,
        np.ones(len(pixels)),
    ], axis=1)


def _sample_mask(mask: np.ndarray, limit: int = MAX_PIXELS_PER_MASK) -> np.ndarray:
    """Mask pixels as (N, 2) [u, v], deterministically subsampled."""
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return np.empty((0, 2))
    if len(rows) > limit:
        step = int(np.ceil(len(rows) / limit))
        rows, cols = rows[::step], cols[::step]
    return np.stack([cols, rows], axis=1).astype(float)


def project_mask_to_surface(
    mask: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    surfaces: Sequence[SurfacePlane],
    label: str = "",
    damage_class: str = "",
    score: float = 0.0,
    frame_index: int = -1,
) -> Optional[ProjectedRegion]:
    """Intersect a mask's camera rays with the surfaces; measure where they land.

    ``R`` and ``t`` are camera-to-world: ``P_world = R @ P_cam + t``.
    Returns ``None`` when the mask does not land coherently on any one surface.
    """
    pixels = _sample_mask(mask)
    if len(pixels) == 0 or not surfaces:
        return None

    with quiet_fp():
        directions = _ray_directions(pixels, K) @ R.T      # world-frame rays
    origin = np.asarray(t, dtype=float)

    best: Optional[Tuple[int, np.ndarray, SurfacePlane]] = None
    for surface in surfaces:
        with quiet_fp():
            denominator = directions @ surface.normal
            numerator = -(origin @ surface.normal - surface.origin @ surface.normal)

        usable = np.abs(denominator) > MIN_INCIDENCE_COS
        if not np.any(usable):
            continue
        distance = np.full(len(directions), np.nan)
        distance[usable] = numerator / denominator[usable]

        valid = np.isfinite(distance) & (distance > MIN_RANGE_M) & (distance < MAX_RANGE_M)
        if not np.any(valid):
            continue

        hits = origin + distance[valid, None] * directions[valid]
        uv = surface.to_local(hits)
        inside = surface.contains(uv)
        count = int(np.count_nonzero(inside))
        if count == 0:
            continue
        if best is None or count > best[0]:
            best = (count, uv[inside], surface)

    if best is None:
        return None

    count, uv, surface = best
    share = count / len(pixels)
    if share < MIN_SURFACE_SHARE:
        # The mask straddles surfaces, or mostly missed them. Claiming a metric
        # extent from that would be inventing one.
        log.debug("mask lands on %s for only %.0f%% of its rays; discarded", surface.surface_id, 100 * share)
        return None

    cells = {(int(np.floor(u / CELL_M)), int(np.floor(v / CELL_M))) for u, v in uv}
    return ProjectedRegion(
        surface_id=surface.surface_id,
        room_id=surface.room_id,
        surface_kind=surface.kind,
        label=label,
        damage_class=damage_class,
        cells=cells,
        score=float(score),
        frames=[frame_index] if frame_index >= 0 else [],
        pixel_count=count,
        wall_id=surface.wall_id,
    )


def merge_regions(
    regions: Sequence[ProjectedRegion], overlap_threshold: float = 0.15
) -> List[ProjectedRegion]:
    """Merge regions of the same class on the same surface when they overlap.

    Cells are indexed in the surface's own frame, so overlap is exact set
    intersection and merging is exact set union -- the same stain seen from six
    frames becomes one region of the right size, not six or one huge one.
    """
    merged: List[ProjectedRegion] = []

    for region in sorted(regions, key=lambda r: -len(r.cells)):
        for existing in merged:
            if existing.surface_id != region.surface_id:
                continue
            if existing.damage_class != region.damage_class:
                continue
            smaller = min(len(existing.cells), len(region.cells))
            if smaller == 0:
                continue
            overlap = len(existing.cells & region.cells) / smaller
            if overlap >= overlap_threshold:
                existing.cells |= region.cells
                existing.frames = sorted(set(existing.frames) | set(region.frames))
                existing.score = max(existing.score, region.score)
                existing.pixel_count += region.pixel_count
                break
        else:
            merged.append(ProjectedRegion(
                surface_id=region.surface_id, room_id=region.room_id,
                surface_kind=region.surface_kind, label=region.label,
                damage_class=region.damage_class, cells=set(region.cells),
                score=region.score, frames=list(region.frames),
                pixel_count=region.pixel_count, wall_id=region.wall_id,
            ))

    log.info("merged %d observations into %d region(s)", len(regions), len(merged))
    return merged


def surfaces_from_layout(
    walls: Sequence[Any],
    frame: Any,
    room_id: str,
    ceiling_height_m: float,
    floor_area_m2: float,
    openings_by_wall: Optional[Mapping[int, int]] = None,
) -> List[SurfacePlane]:
    """Build projectable surfaces from a fitted layout.

    This is the only place semantics touches geometry, and it touches it through
    plain wall segments and a floor frame -- which is what makes the rest of this
    package tier-agnostic. Any tier that can produce walls in a floor frame gets
    damage, flags and scope for free.
    """
    openings_by_wall = openings_by_wall or {}
    surfaces: List[SurfacePlane] = []
    up = np.asarray(frame.up, dtype=float)

    for index, wall in enumerate(walls):
        start_uv = np.array(wall.start, dtype=float)
        end_uv = np.array(wall.end, dtype=float)
        length = float(np.linalg.norm(end_uv - start_uv))
        if length <= 0:
            continue

        origin = frame.to_world(start_uv[None, :], height=0.0)[0]
        end_world = frame.to_world(end_uv[None, :], height=0.0)[0]
        u_axis = (end_world - origin) / length
        # Inward normal: the wall's own outward normal, flipped.
        normal_uv = np.asarray(wall.normal, dtype=float)
        normal = -(normal_uv[0] * frame.e1 + normal_uv[1] * frame.e2)
        normal = normal / np.linalg.norm(normal)

        surfaces.append(SurfacePlane(
            surface_id=f"{room_id}_w{index}_surface",
            room_id=room_id, kind="wall",
            origin=origin, normal=normal, u_axis=u_axis, v_axis=up,
            u_extent=length, v_extent=ceiling_height_m,
            area_m2=length * ceiling_height_m,
            wall_id=f"{room_id}_w{index}",
            has_opening=bool(openings_by_wall.get(index)),
        ))

    if walls:
        # Bound the floor and ceiling to the room's own footprint. An unbounded
        # plane accepts any ray that points downward, so masks that belong to no
        # wall land on "the floor" and are reported as floor damage -- on the
        # sample capture that turned tile grout lines into three cracks.
        corners = np.array([w.start for w in walls] + [w.end for w in walls], dtype=float)
        u0, v0 = corners.min(axis=0)
        u1, v1 = corners.max(axis=0)
        margin = 0.10
        origin_uv = np.array([u0 - margin, v0 - margin])

        for kind, height, plane_normal in (
            ("floor", 0.0, up),
            ("ceiling", ceiling_height_m, -up),
        ):
            surfaces.append(SurfacePlane(
                surface_id=f"{room_id}_{kind}",
                room_id=room_id, kind=kind,
                origin=frame.to_world(origin_uv[None, :], height=height)[0],
                normal=plane_normal,
                u_axis=frame.e1, v_axis=frame.e2,
                u_extent=float(u1 - u0) + 2 * margin,
                v_extent=float(v1 - v0) + 2 * margin,
                area_m2=floor_area_m2,
            ))

    return surfaces
