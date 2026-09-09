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
    densest_edge,
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

# A "wall" candidate closer to the camera than the floor point near your own
# feet cannot be the room's own bounding wall -- it is furniture. Found by
# testing against a real room with a wardrobe standing away from the true
# wall: the wardrobe's flat door is exactly as vertical-normal as a real wall
# and, because VGGT's per-pixel confidence favours near content, often has
# *more* inlier points than the true wall does, so nothing upstream of this
# check tells them apart. Requiring a wall to be meaningfully farther than the
# floor directly in front of you holds for any room bigger than you are tall,
# which is every room this tier is meant to run on.
WALL_MIN_RANGE_OVER_FLOOR = 1.2

# A wall's position is snapped to the outer edge of the pooled floor evidence
# (see _wall_line_from_cluster and _densest_edge) rather than trusted from one
# frame's own offset estimate. A raw percentile trusts a single far stray
# point (VGGT depth noise, or a glimpse through a doorway into another room)
# exactly as much as a genuinely dense wall -- found on demo_fourroom's
# office_b/c, 3-4 photos of a cluttered real scene where the pooled cloud is a
# scatter, not a clean ring, and a percentile alone routinely snapped a wall
# metres past anything real. ROOM_EDGE_BIN_M / ROOM_EDGE_MIN_BIN_FRACTION walk
# in from the extreme until they find a histogram bin that actually holds a
# working fraction of an even spread's share (1 / bin count along that axis),
# not merely a bin that holds *something*. ROOM_EDGE_MIN_FLOOR_POINTS guards
# against doing any of this off a pool too small to trust at all (falls back
# to the per-view estimate).
ROOM_EDGE_BIN_M = 0.05
ROOM_EDGE_MIN_BIN_FRACTION = 0.5
ROOM_EDGE_MIN_FLOOR_POINTS = 200

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
    floor_local: Optional[float] = None   # this frame's own floor distance, if found

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

        if floor_local is not None and abs(offset_local) <= WALL_MIN_RANGE_OVER_FLOOR * abs(floor_local):
            # Closer than (or barely past) the floor at your own feet: a
            # room's own bounding wall cannot be, so this is furniture (a
            # wardrobe door, a headboard, ...) wearing a wall-shaped normal,
            # not a wall -- see WALL_MIN_RANGE_OVER_FLOOR above.
            log.debug(
                "frame %d: dropping a wall candidate at %.3f units (floor is %.3f units away) "
                "-- too close to be this room's own wall", frame_index, abs(offset_local), abs(floor_local),
            )
            continue

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


def _densest_edge(axis_vals: np.ndarray, toward_high: bool) -> float:
    """This tier's wall-snapping edge -- see ROOM_EDGE_BIN_M above for why it is
    a histogram walk-in rather than a raw percentile. One implementation,
    shared with the arrangement's own axis fallback."""
    return densest_edge(
        axis_vals, toward_high,
        bin_m=ROOM_EDGE_BIN_M, min_bin_fraction=ROOM_EDGE_MIN_BIN_FRACTION,
    )


def _wall_line_from_cluster(
    cluster: Sequence[ViewPlane], frame: FloorFrame, floor_uv: Optional[np.ndarray] = None,
    camera_uv_by_frame: Optional[Dict[int, np.ndarray]] = None,
) -> Optional[Tuple[WallLine, Dict[str, Any]]]:
    """A merged world-frame wall plane -> a WallLine in the room's floor frame.

    The cluster's own position estimate is a per-view plane offset -- reliable
    for *which side of the room this wall is on* but not for *how far away it
    is*: VGGT's per-frame points are densest and most confident close to the
    camera, so a wall plane fit from one photo's own points tends to land
    short of the wall's true position, especially from the room's middle
    (see WALL_MIN_RANGE_OVER_FLOOR above for the furniture case this shares a
    cause with). ``floor_uv``, when given, is every frame's floor-height
    points pooled together -- far denser and, critically, not biased toward
    any one camera -- so the wall is snapped outward to the real edge of that
    pooled floor evidence along its own axis.

    Which edge (the axis's low or high side) is decided from the cluster's
    *own camera position(s)*, not from comparing its (biased) coordinate to
    a room-wide median: with several walls all biased toward the same
    centrally-placed cameras, more than one can land on the same side of a
    global median, which snaps them all to the same edge and silently
    inflates the room in the other direction. "Farther from this cluster's
    own camera(s), in the direction it already looks" has no such failure
    mode -- it only breaks if a camera saw straight through this wall, which
    a wall detection already rules out.
    """
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

    if floor_uv is not None and len(floor_uv) >= ROOM_EDGE_MIN_FLOOR_POINTS:
        camera_axis_val = None
        if camera_uv_by_frame is not None:
            paired = [(camera_uv_by_frame[p.frame_index][axis], p.inliers) for p in cluster
                      if p.frame_index in camera_uv_by_frame]
            if paired:
                cam_vals, cam_weights = zip(*paired)
                camera_axis_val = float(np.average(cam_vals, weights=cam_weights))
        reference = camera_axis_val if camera_axis_val is not None else float(np.median(floor_uv[:, axis]))
        coord = _densest_edge(floor_uv[:, axis], toward_high=coord >= reference)

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

    # ASSUMED_CAMERA_DOWN is +Y in each VGGT camera frame.  The merged floor
    # normal therefore points toward increasing camera-down height; its
    # negation is the room's true up axis.  Do not force this onto world +Y:
    # VGGT's world frame is camera-derived, and flipping it makes every point
    # above the floor (including door/window rays) appear below the floor.
    up = -floor_plane.normal / np.linalg.norm(floor_plane.normal)

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

    # Every frame's own points, pooled -- computed before the wall lines so
    # _wall_line_from_cluster can snap each wall out to this pooled evidence's
    # own outer edge instead of trusting a single (near-biased) per-view offset.
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
    camera_uv_by_frame = {p.frame_index: trajectory_uv[i] for i, p in enumerate(reconstruction.poses)}

    lines: List[WallLine] = []
    used_floor_fallback = False
    view_details: Dict[int, Dict[str, Any]] = {}
    for i, cluster in enumerate(wall_clusters):
        built = _wall_line_from_cluster(cluster, frame, floor_uv, camera_uv_by_frame)
        if built is None:
            continue
        line, detail = built
        lines.append(line)
        view_details[len(lines) - 1] = detail
    if not lines:
        # Two-photo corridors often contain a usable floor/trajectory extent
        # but no stable wall-plane cluster. Keep the room as a conservative
        # evidence-bounded rectangle instead of dropping it from the property.
        used_floor_fallback = True
        evidence = floor_uv if len(floor_uv) >= 8 else all_uv
        if len(evidence) < 8:
            raise ValueError("no wall cluster produced a usable line; too few views or too little overlap")
        u0, v0 = np.percentile(evidence, 5, axis=0)
        u1, v1 = np.percentile(evidence, 95, axis=0)
        if u1 - u0 < 0.8 or v1 - v0 < 0.8:
            u0, v0 = np.min(evidence, axis=0)
            u1, v1 = np.max(evidence, axis=0)
        if u1 - u0 < 0.5 or v1 - v0 < 0.5:
            raise ValueError("no wall cluster produced a usable line; floor evidence is too small")
        support = max(1, len(evidence) // 4)
        lines = [
            WallLine(axis=0, coord=float(u0), extent=(float(v0), float(v1)), inliers=support, top_height_m=0.0),
            WallLine(axis=0, coord=float(u1), extent=(float(v0), float(v1)), inliers=support, top_height_m=0.0),
            WallLine(axis=1, coord=float(v0), extent=(float(u0), float(u1)), inliers=support, top_height_m=0.0),
            WallLine(axis=1, coord=float(v1), extent=(float(u0), float(u1)), inliers=support, top_height_m=0.0),
        ]
        view_details = {
            index: {"views": sorted({p.frame_index for p in reconstruction.poses}),
                    "view_count": len(reconstruction.poses), "agreement": 0.2}
            for index in range(4)
        }
        log.warning(
            "no usable wall cluster; using a low-confidence rectangle from floor evidence"
        )

    polygon, walls, assembly_stats = assemble_polygon(lines, all_uv, floor_uv, trajectory_uv, margin=0.5)
    if used_floor_fallback:
        assembly_stats["axis_fallback"] = [0, 1]

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
