"""Photo-tier layout: the Plane-DUSt3R ordering.

A cloud fused from five photos is far too sparse for the LiDAR path's global
RANSAC (geometry/layout.py's ``extract_layout``): that path needs a dense wall
band to find lines in, and five photos give it a few thousand points scattered
across a whole room. Fitting one global plane to that sparse a cloud is fitting
noise.

So the order is inverted: fit planes **per image first**, where each frame's
own local point cloud is still dense enough to see its own floor and walls
clearly, and only then bring the per-view hypotheses together:

1. **Per-view planes.** For each frame, using only that frame's own
   camera-local points: find the floor (lowest, below the camera), the ceiling
   (highest, if visible), and one plane per visible wall (vertical normal
   clusters). This needs a "down" direction in a single, otherwise-arbitrary
   monocular frame; see ``ASSUMED_CAMERA_DOWN`` below for the assumption this
   makes and why.
2. **Transform to a common frame.** Each per-view plane is carried into the
   world frame by that frame's own pose (the first frame's camera, by
   backbone convention).
3. **Cluster and merge.** World-frame planes with a similar normal and a
   similar offset are the same physical surface seen from different photos,
   and are merged into one. The merged wall planes close into a room polygon
   using the exact same cell-arrangement code the LiDAR tier uses
   (:func:`cozmo.geometry.layout.assemble_polygon`) -- nothing here forks that
   logic, it only produces the ``WallLine`` list that logic already consumes.

Per-view plane count and cross-view agreement are the two things sparse photo
geometry can actually tell you about its own confidence, so both are recorded
per wall and used to widen (or trust) its interval.

**Fallback.** With enough views that the fused cloud is genuinely dense --
this is what the video tier will need -- steps 1-3 are unnecessary, and the
LiDAR path's own global RANSAC runs directly on the fused cloud instead. Which
path ran is always recorded (``layout_method``), because the two paths carry
different honest confidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..geometry._numeric import quiet_fp
from ..geometry.layout import (
    FloorFrame,
    LayoutResult,
    WallLine,
    WallSegment,
    assemble_polygon,
    dominant_rotation,
    estimate_normals,
    extract_layout,
)
from ..geometry.planes import Plane, fit_floor
from .backbone import ReconstructionResult

log = logging.getLogger("cozmo.recon.layout")

# Assumption: a photo is taken with the phone held roughly upright, so the
# camera's own local +Y axis (image-down) is close to gravity-down. This is the
# standard casual-photography case and is what lets a *single* monocular frame
# propose a floor/ceiling/wall split at all, without a dedicated vanishing-point
# or learned single-view layout model. It fails for a frame shot with the phone
# rolled or tilted a long way from upright, which is exactly why step 3 checks
# cross-view agreement rather than trusting any one frame's split.
ASSUMED_CAMERA_DOWN = np.array([0.0, 1.0, 0.0])

FLOOR_BAND_FRACTION = 0.20      # lowest 20% of a frame's own vertical range
CEILING_BAND_FRACTION = 0.20
VERTICAL_NORMAL_MAX = 0.35
MIN_POINTS_PER_VIEW_PLANE = 40

WALL_NORMAL_CLUSTER_COS = 0.90    # cos(~25 deg): same wall direction across views
WALL_OFFSET_CLUSTER_M = 0.35      # same wall position across views

# Below this many frames, a global RANSAC has too little to work with; above
# it, per-view fitting is unnecessary and the LiDAR path's own global fitter
# (denser, better-tested) takes over. This is the number the brief's "enough
# views that the fused cloud is dense" fallback triggers on.
GLOBAL_FUSION_MIN_FRAMES = 12
GLOBAL_FUSION_MIN_POINTS = 20_000


@dataclass
class ViewPlane:
    """One plane hypothesis from one frame's own local points."""

    frame_index: int
    kind: str                    # floor | ceiling | wall
    normal_world: np.ndarray     # unit, world frame
    point_world: np.ndarray      # a point on the plane, world frame
    inliers: int
    extent_world: Optional[Tuple[np.ndarray, np.ndarray]] = None   # (min, max) along-wall, world

    @property
    def offset(self) -> float:
        return float(np.dot(self.point_world, self.normal_world))


def _view_floor_ceiling_wall(
    local_points: np.ndarray, local_conf: np.ndarray, pose_R: np.ndarray, pose_t: np.ndarray,
    frame_index: int,
) -> List[ViewPlane]:
    """One frame's own floor / ceiling / wall hypotheses, in world coordinates."""
    if len(local_points) < MIN_POINTS_PER_VIEW_PLANE:
        return []

    down = ASSUMED_CAMERA_DOWN
    with quiet_fp():
        height = local_points @ down          # larger = lower (down is +)
    lo, hi = np.percentile(height, [2, 98])
    span = hi - lo
    if span <= 0:
        return []

    planes: List[ViewPlane] = []

    floor_mask = height >= hi - FLOOR_BAND_FRACTION * span
    if floor_mask.sum() >= MIN_POINTS_PER_VIEW_PLANE:
        floor_local = float(np.median(height[floor_mask]))
        point_local = floor_local * down
        planes.append(ViewPlane(
            frame_index=frame_index, kind="floor",
            normal_world=pose_R @ down, point_world=pose_R @ point_local + pose_t,
            inliers=int(floor_mask.sum()),
        ))

    ceiling_mask = height <= lo + CEILING_BAND_FRACTION * span
    if ceiling_mask.sum() >= MIN_POINTS_PER_VIEW_PLANE:
        ceiling_local = float(np.median(height[ceiling_mask]))
        point_local = ceiling_local * down
        planes.append(ViewPlane(
            frame_index=frame_index, kind="ceiling",
            normal_world=pose_R @ down, point_world=pose_R @ point_local + pose_t,
            inliers=int(ceiling_mask.sum()),
        ))

    normals = estimate_normals(local_points)
    with quiet_fp():
        vertical = np.abs(normals @ down) < VERTICAL_NORMAL_MAX
    if vertical.sum() < MIN_POINTS_PER_VIEW_PLANE:
        return planes

    wall_points = local_points[vertical]
    wall_normals = normals[vertical]
    # Cluster wall points by normal direction (mod pi -- a plane's two faces
    # share a direction) into distinct walls this one frame can see.
    angle = np.arctan2(wall_normals[:, 2], wall_normals[:, 0]) % np.pi
    order = np.argsort(angle)
    sorted_angle = angle[order]
    breaks = np.flatnonzero(np.diff(sorted_angle) > np.radians(20))
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks + 1, [len(sorted_angle)]])

    for start, end in zip(starts, ends):
        idx = order[start:end]
        if len(idx) < MIN_POINTS_PER_VIEW_PLANE:
            continue
        cluster_points = wall_points[idx]
        cluster_normals = wall_normals[idx]
        mean_normal = cluster_normals.mean(axis=0)
        norm = np.linalg.norm(mean_normal)
        if norm < 1e-6:
            continue
        mean_normal /= norm
        with quiet_fp():
            offset_local = float(np.median(cluster_points @ mean_normal))
        point_local = offset_local * mean_normal

        with quiet_fp():
            world_points = cluster_points @ pose_R.T + pose_t
        # Extent along the wall: project onto the direction perpendicular to
        # the normal and to "down", i.e. the wall's own horizontal direction.
        along = np.cross(down, mean_normal)
        along_norm = np.linalg.norm(along)
        if along_norm < 1e-6:
            continue
        along /= along_norm
        with quiet_fp():
            proj = cluster_points @ along
        extent_local = (proj.min(), proj.max())
        extent_world = (
            world_points[np.argmin(proj)],
            world_points[np.argmax(proj)],
        )

        planes.append(ViewPlane(
            frame_index=frame_index, kind="wall",
            normal_world=pose_R @ mean_normal, point_world=pose_R @ point_local + pose_t,
            inliers=len(idx), extent_world=extent_world,
        ))

    return planes


def _cluster_wall_planes(planes: Sequence[ViewPlane]) -> List[List[ViewPlane]]:
    """Group per-view wall hypotheses that describe the same physical wall."""
    walls = [p for p in planes if p.kind == "wall"]
    clusters: List[List[ViewPlane]] = []
    for plane in walls:
        placed = False
        for cluster in clusters:
            rep = cluster[0]
            cos = abs(float(np.dot(plane.normal_world, rep.normal_world)))
            if cos < WALL_NORMAL_CLUSTER_COS:
                continue
            if abs(plane.offset - rep.offset) > WALL_OFFSET_CLUSTER_M:
                continue
            cluster.append(plane)
            placed = True
            break
        if not placed:
            clusters.append([plane])
    return clusters


def _merge_floor_or_ceiling(planes: Sequence[ViewPlane]) -> Optional[Plane]:
    matches = [p for p in planes]
    if not matches:
        return None
    weights = np.array([p.inliers for p in matches], dtype=float)
    normal = np.average([p.normal_world for p in matches], axis=0, weights=weights)
    normal = normal / np.linalg.norm(normal)
    offset = float(np.average([p.offset for p in matches], weights=weights))
    return Plane(normal=normal, d=-offset, inliers=int(weights.sum()), total=int(weights.sum()))


def _wall_line_from_cluster(
    cluster: Sequence[ViewPlane], frame: FloorFrame,
) -> Optional[Tuple[WallLine, Dict[str, Any]]]:
    """A merged world-frame wall plane -> a WallLine in the room's floor frame."""
    weights = np.array([p.inliers for p in cluster], dtype=float)
    normal = np.average([p.normal_world for p in cluster], axis=0, weights=weights)
    normal = normal / np.linalg.norm(normal)
    offset = float(np.average([p.offset for p in cluster], weights=weights))

    n2 = np.array([float(normal @ frame.e1), float(normal @ frame.e2)])
    n2 /= max(np.linalg.norm(n2), 1e-9)
    axis = 0 if abs(n2[0]) >= abs(n2[1]) else 1

    point_on_plane = normal * offset
    uv = frame.project(point_on_plane[None, :])[0]
    coord = uv[axis]

    extents = []
    for plane in cluster:
        if plane.extent_world is None:
            continue
        a_uv = frame.project(plane.extent_world[0][None, :])[0]
        b_uv = frame.project(plane.extent_world[1][None, :])[0]
        extents.append(a_uv[1 - axis])
        extents.append(b_uv[1 - axis])
    if len(extents) < 2:
        return None
    span = (min(extents), max(extents))
    if span[1] - span[0] < 0.3:
        return None

    frames_seen = sorted({p.frame_index for p in cluster})
    agreement = 1.0
    if len(cluster) > 1:
        offsets = np.array([p.offset for p in cluster])
        agreement = float(max(0.0, 1.0 - np.std(offsets) / 0.15))

    line = WallLine(axis=axis, coord=float(coord), extent=span, inliers=int(weights.sum()), top_height_m=0.0)
    detail = {"views": frames_seen, "view_count": len(frames_seen), "agreement": round(agreement, 3)}
    return line, detail


@dataclass
class PhotoLayoutResult(LayoutResult):
    layout_method: str = "per_view_merge"
    view_plane_counts: Dict[str, Any] = field(default_factory=dict)
    floor_to_ceiling_span: Optional[float] = None
    wall_confidence: Dict[int, float] = field(default_factory=dict)


def extract_layout_photo(
    reconstruction: ReconstructionResult,
    seed: int = 0,
) -> PhotoLayoutResult:
    """Room polygon from a sparse, multi-view photo reconstruction.

    Chooses between the per-view merge path (few frames, sparse fused cloud)
    and the LiDAR path's own global RANSAC (enough frames that the fused
    cloud is dense) -- see ``GLOBAL_FUSION_MIN_FRAMES``.
    """
    n_frames = len(reconstruction.frame_points_local)
    n_points = len(reconstruction.points)

    if n_frames >= GLOBAL_FUSION_MIN_FRAMES and n_points >= GLOBAL_FUSION_MIN_POINTS:
        log.info(
            "%d frames / %d points: dense enough for the global fusion path", n_frames, n_points
        )
        floor = fit_floor(reconstruction.points)
        base = extract_layout(reconstruction.points, floor, np.array([p.t for p in reconstruction.poses]))
        return PhotoLayoutResult(
            **{f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values()},
            layout_method="global_fusion",
        )

    per_view: List[ViewPlane] = []
    for pose in reconstruction.poses:
        local = reconstruction.frame_points_local.get(pose.frame_index)
        conf = reconstruction.frame_confidence_local.get(pose.frame_index)
        if local is None or len(local) == 0:
            continue
        per_view.extend(_view_floor_ceiling_wall(local, conf, pose.R, pose.t, pose.frame_index))

    floors = [p for p in per_view if p.kind == "floor"]
    ceilings = [p for p in per_view if p.kind == "ceiling"]
    if not floors:
        raise ValueError("no frame produced a usable floor hypothesis")

    floor_plane = _merge_floor_or_ceiling(floors)
    ceiling_plane = _merge_floor_or_ceiling(ceilings)

    up = -floor_plane.normal / np.linalg.norm(floor_plane.normal)
    # ASSUMED_CAMERA_DOWN maps to world "down" per-frame via that frame's own
    # pose; floor_plane.normal already points toward lower height (see
    # _view_floor_ceiling_wall), so "up" is its negation.
    if float(np.dot(up, [0, 1, 0])) < 0:
        up = -up

    wall_clusters = _cluster_wall_planes(per_view)
    seed_axis = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed_axis, up))) > 0.9:
        seed_axis = np.array([0.0, 0.0, 1.0])
    a1 = seed_axis - np.dot(seed_axis, up) * up
    a1 /= np.linalg.norm(a1)
    a2 = np.cross(up, a1)

    wall_normals_2d = []
    for cluster in wall_clusters:
        weights = np.array([p.inliers for p in cluster], dtype=float)
        normal = np.average([p.normal_world for p in cluster], axis=0, weights=weights)
        normal /= max(np.linalg.norm(normal), 1e-9)
        wall_normals_2d.append([float(normal @ a1), float(normal @ a2)])
    wall_normals_2d = np.array(wall_normals_2d) if wall_normals_2d else np.empty((0, 2))
    lengths = np.linalg.norm(wall_normals_2d, axis=1, keepdims=True) if len(wall_normals_2d) else None
    if lengths is not None:
        wall_normals_2d = wall_normals_2d / np.maximum(lengths, 1e-9)

    rotation, manhattan_score = dominant_rotation(wall_normals_2d) if len(wall_normals_2d) else (0.0, 0.0)
    cos_r, sin_r = np.cos(rotation), np.sin(rotation)
    e1 = cos_r * a1 + sin_r * a2
    e2 = np.cross(up, e1)
    origin = floor_plane.normal * (-floor_plane.d / float(np.dot(floor_plane.normal, floor_plane.normal)))
    frame = FloorFrame(origin=origin, up=up, e1=e1, e2=e2, rotation_rad=float(rotation))

    lines: List[WallLine] = []
    view_details: Dict[int, Dict[str, Any]] = {}
    for i, cluster in enumerate(wall_clusters):
        built = _wall_line_from_cluster(cluster, frame)
        if built is None:
            continue
        line, detail = built
        lines.append(line)
        view_details[len(lines) - 1] = detail
    if not lines:
        raise ValueError("no wall cluster produced a usable line; too few views or too little overlap")

    all_world_points = np.concatenate([
        reconstruction.frame_points_local[p.frame_index] @ p.R.T + p.t
        for p in reconstruction.poses if p.frame_index in reconstruction.frame_points_local
    ])
    all_uv = frame.project(all_world_points)
    with quiet_fp():
        all_height = all_world_points @ up

    floor_height = float(np.dot(origin, up))
    heights_above_floor = all_height - floor_height
    floor_uv = frame.project(all_world_points[np.abs(heights_above_floor) <= 0.15])
    trajectory_uv = frame.project(np.array([p.t for p in reconstruction.poses]))

    polygon, walls, assembly_stats = assemble_polygon(lines, all_uv, floor_uv, trajectory_uv, margin=0.5)

    for index, wall in enumerate(walls):
        detail = view_details.get(index, {})
        wall.support_points = int(lines[index].inliers) if index < len(lines) else 0

    floor_to_ceiling_span = None
    if ceiling_plane is not None:
        # Height of the ceiling plane above the floor, along "up": a plane is
        # {p : normal . p = -d}, so any point on it is -d * normal / |normal|^2;
        # projecting that onto "up" and subtracting the floor's own height
        # gives the vertical span between the two, in the backbone's
        # scale-free units -- exactly what scale.py's ceiling-height cue needs.
        ceiling_point = ceiling_plane.normal * (-ceiling_plane.d)
        ceiling_height = float(np.dot(ceiling_point, up))
        floor_to_ceiling_span = abs(ceiling_height - floor_height)

    result = PhotoLayoutResult(
        frame=frame, polygon=polygon, walls=walls, lines=lines,
        floor_area_m2=float(polygon.area), perimeter_m=float(polygon.exterior.length),
        manhattan_snap=True, rotation_deg=float(np.degrees(rotation)),
        points_uv=all_uv, points_height=heights_above_floor,
        layout_method="per_view_merge",
        view_plane_counts={
            "floor_views": len(floors), "ceiling_views": len(ceilings),
            "wall_clusters": len(wall_clusters), "wall_lines": len(lines),
        },
        floor_to_ceiling_span=floor_to_ceiling_span,
        wall_confidence={i: view_details.get(i, {}).get("agreement", 0.5) for i in range(len(walls))},
        stats={
            "manhattan_score": round(float(manhattan_score), 4),
            "view_details": view_details,
            **assembly_stats,
        },
    )
    log.info(
        "photo layout (per-view merge): %.2f units^2, %d walls from %d wall cluster(s), "
        "%d/%d frames contributed floor/ceiling",
        result.floor_area_m2, len(walls), len(wall_clusters), len(floors), len(ceilings),
    )
    return result
