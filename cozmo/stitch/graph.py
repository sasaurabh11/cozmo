"""Room pose graph: matched rooms in, one connected, non-overlapping plan out.

    per-room reconstructions + match.py's correspondences
      -> lift 2D matches to 3D, project onto each room's own floor plane
      -> per-pair relative SE(2) (Procrustes on image matches, or a doorway
         alignment when there are too few of those)
      -> global pose graph (reusing the same optimiser as drift.py -- x, y,
         yaw per room, one room fixed as the reference frame)
      -> snap the whole property to a shared Manhattan frame
      -> push apart any rooms shapely says still overlap
      -> Adjacency objects, one per edge that made it this far

The two evidence sources are combined, not one chosen over the other:
keypoint matches give a precise transform when there are enough of them;
a matched doorway is what covers a room that shares a wall but no visible
scene with its neighbour (the case the brief calls the strongest signal, and
the only one that can work through a closed door).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import Polygon
from shapely.ops import unary_union

from ..geometry.layout import FloorFrame, WallSegment
from ..stitch.drift import _optimize_pose_graph, _wrap, _yaw_to_R
from ..stitch.match import RoomPairMatch

log = logging.getLogger("cozmo.stitch.graph")

# A relative transform from image matches needs at least this many inliers
# (after the RANSAC-lite consensus check below) to be trusted on its own.
MIN_TRANSFORM_INLIERS = 4
# An inlier's Procrustes residual, in metres, after the fitted transform.
INLIER_RESIDUAL_M = 0.30

EDGE_WEIGHT_PER_MATCH = 0.15
EDGE_WEIGHT_DOORWAY = 3.0
EDGE_WEIGHT_MAX = 8.0

OVERLAP_PUSH_STEP_M = 0.05
OVERLAP_MAX_ITERATIONS = 200
OVERLAP_AREA_TOLERANCE_M2 = 0.01


@dataclass
class RoomForStitch:
    """What graph.py needs from one already-reconstructed room.

    ``openings`` maps an opening id to ``(wall_index, width_m)`` -- the index
    into ``walls``, not a field on ``WallSegment`` itself (that dataclass is
    shared with the single-room LiDAR/photo paths and carries no opening
    linkage of its own), so a matched doorway can be turned back into the
    actual wall it sits on.
    """

    room_id: str
    frame: FloorFrame
    polygon: Polygon
    walls: List[WallSegment]
    # frame_index -> (N, 3) points in THIS room's own local/world frame (the
    # same frame `frame` and `polygon` are expressed in), for lifting 2D
    # keypoint matches to 3D.
    frame_points: Dict[int, np.ndarray]
    frame_K: Dict[int, np.ndarray]
    openings: Dict[str, Tuple[int, float]] = field(default_factory=dict)


@dataclass
class StitchEdge:
    room_a: str
    room_b: str
    relative_xy: np.ndarray        # room_b's origin, in room_a's 2D floor frame
    relative_yaw: float            # room_b's rotation relative to room_a
    confidence: float
    source: str                   # "keypoints" | "doorway" | "keypoints+doorway"
    via_opening_a: Optional[str] = None
    via_opening_b: Optional[str] = None
    wall_a: Optional[str] = None
    wall_b: Optional[str] = None
    inlier_count: int = 0


@dataclass
class StitchResult:
    room_ids: List[str]
    global_xy: Dict[str, np.ndarray]     # room_id -> (x, y) in the property frame
    global_yaw: Dict[str, float]
    edges_used: List[StitchEdge]
    edges_rejected: List[Tuple[str, str, str]]   # (room_a, room_b, reason)
    manhattan_rotation_rad: float
    overlap_resolved: bool
    overlap_iterations: int

    def transform_polygon(self, room_id: str, polygon: Polygon) -> Polygon:
        x, y = self.global_xy[room_id]
        yaw = self.global_yaw[room_id]
        c, s = np.cos(yaw), np.sin(yaw)
        # shapely affine_transform matrix: [a, b, d, e, xoff, yoff]
        return affine_transform(polygon, [c, -s, s, c, x, y])

    def transform_point(self, room_id: str, point_uv: Tuple[float, float]) -> Tuple[float, float]:
        x, y = self.global_xy[room_id]
        yaw = self.global_yaw[room_id]
        c, s = np.cos(yaw), np.sin(yaw)
        u, v = point_uv
        return (c * u - s * v + x, s * u + c * v + y)


def _nearest_point_3d(points: np.ndarray, K: np.ndarray, uv: Tuple[float, float]) -> Optional[np.ndarray]:
    """The room's own reconstructed 3D point whose reprojection is closest to
    a 2D pixel. Approximate -- there is no stored per-pixel index into the
    (confidence-filtered) reconstruction -- but with thousands of points per
    frame the nearest reprojection is a reasonable stand-in for the exact one.
    """
    if len(points) == 0:
        return None
    z = points[:, 2]
    valid = z > 1e-3
    if not np.any(valid):
        return None
    projected = points[valid, :2] / z[valid, None] * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    distances = np.linalg.norm(projected - np.array(uv), axis=1)
    best = np.argmin(distances)
    if distances[best] > 40:   # pixels; a poor reprojection match is not evidence
        return None
    return points[valid][best]


def _lift_matches_to_floor_frame(
    match: RoomPairMatch, room_a: RoomForStitch, room_b: RoomForStitch,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Matched 2D keypoints -> matched 2D points in each room's own floor frame."""
    pts_a, pts_b, weights = [], [], []
    for m in match.keypoint_matches:
        points_a = room_a.frame_points.get(m.frame_index_a)
        points_b = room_b.frame_points.get(m.frame_index_b)
        K_a = room_a.frame_K.get(m.frame_index_a)
        K_b = room_b.frame_K.get(m.frame_index_b)
        if points_a is None or points_b is None or K_a is None or K_b is None:
            continue
        p3_a = _nearest_point_3d(points_a, K_a, (m.u_a, m.v_a))
        p3_b = _nearest_point_3d(points_b, K_b, (m.u_b, m.v_b))
        if p3_a is None or p3_b is None:
            continue
        pts_a.append(room_a.frame.project(p3_a[None, :])[0])
        pts_b.append(room_b.frame.project(p3_b[None, :])[0])
        weights.append(m.score)
    if not pts_a:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
    return np.array(pts_a), np.array(pts_b), np.array(weights)


def _procrustes_2d(
    pts_a: np.ndarray, pts_b: np.ndarray, weights: np.ndarray,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Weighted rigid 2D transform mapping ``pts_b`` onto ``pts_a``: finds
    (t, yaw) minimising sum(weight * |R(yaw) @ pts_b + t - pts_a|^2).

    Closed form (Kabsch/Procrustes in 2D via SVD), same as every other
    point-set alignment in this codebase, just in two dimensions.
    """
    w = weights / weights.sum()
    mean_a = (pts_a * w[:, None]).sum(axis=0)
    mean_b = (pts_b * w[:, None]).sum(axis=0)
    a_centered = pts_a - mean_a
    b_centered = pts_b - mean_b
    H = (b_centered * w[:, None]).T @ a_centered
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    t = mean_a - R @ mean_b
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))

    predicted = (b_centered @ R.T) + mean_a
    residuals = np.linalg.norm(predicted - pts_a, axis=1)
    return t, yaw, residuals


def _ransac_procrustes(
    pts_a: np.ndarray, pts_b: np.ndarray, weights: np.ndarray, seed: int = 0,
) -> Optional[Tuple[np.ndarray, float, int]]:
    """Procrustes with a RANSAC-lite consensus pass: fit on all matches, drop
    residual outliers (a wrong keypoint match, not a wrong room alignment),
    refit on the survivors. Returns ``(t, yaw, inlier_count)`` or ``None``.
    """
    if len(pts_a) < 3:
        return None
    t, yaw, residuals = _procrustes_2d(pts_a, pts_b, weights)
    inliers = residuals <= INLIER_RESIDUAL_M
    if inliers.sum() < 3:
        return None
    t2, yaw2, residuals2 = _procrustes_2d(pts_a[inliers], pts_b[inliers], weights[inliers])
    final_inliers = int((residuals2 <= INLIER_RESIDUAL_M).sum())
    return t2, yaw2, final_inliers


def _doorway_transform(
    match: RoomPairMatch, room_a: RoomForStitch, room_b: RoomForStitch,
) -> Optional[Tuple[np.ndarray, float, str, str, str, str]]:
    """A relative transform from a matched doorway alone: align the opening's
    centre and require the two walls' outward normals to point at each other
    (rooms meet at a shared wall, so the door faces are antiparallel).
    """
    if not match.doorway_matches:
        return None
    doorway = match.doorway_matches[0]     # the widest-margin match; ties are rare

    def opening_frame(room: RoomForStitch, opening_id: str):
        entry = room.openings.get(opening_id)
        if entry is None:
            return None
        wall_index, _width = entry
        if not (0 <= wall_index < len(room.walls)):
            return None
        wall = room.walls[wall_index]
        mid = np.array([(wall.start[0] + wall.end[0]) / 2, (wall.start[1] + wall.end[1]) / 2])
        return mid, np.array(wall.normal), f"{room.room_id}_w{wall_index}"

    found_a = opening_frame(room_a, doorway.opening_id_a)
    found_b = opening_frame(room_b, doorway.opening_id_b)
    if found_a is None or found_b is None:
        return None
    mid_a, normal_a, wall_a = found_a
    mid_b, normal_b, wall_b = found_b

    # Rotate room B so its door normal points opposite room A's door normal,
    # then translate so the two door centres coincide.
    angle_a = np.arctan2(normal_a[1], normal_a[0])
    angle_b = np.arctan2(normal_b[1], normal_b[0])
    yaw = _wrap(angle_a + np.pi - angle_b)
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    t = mid_a - R @ mid_b
    return t, yaw, wall_a, wall_b, doorway.opening_id_a, doorway.opening_id_b


def build_edges(
    matches: Sequence[RoomPairMatch], rooms_by_id: Dict[str, RoomForStitch],
) -> Tuple[List[StitchEdge], List[Tuple[str, str, str]]]:
    edges: List[StitchEdge] = []
    rejected: List[Tuple[str, str, str]] = []

    for match in matches:
        room_a, room_b = rooms_by_id[match.room_a], rooms_by_id[match.room_b]
        keypoint_result = None
        if match.keypoint_matches:
            pts_a, pts_b, weights = _lift_matches_to_floor_frame(match, room_a, room_b)
            if len(pts_a) >= 3:
                keypoint_result = _ransac_procrustes(pts_a, pts_b, weights)

        doorway_result = _doorway_transform(match, room_a, room_b)

        if keypoint_result and keypoint_result[2] >= MIN_TRANSFORM_INLIERS:
            t, yaw, inliers = keypoint_result
            confidence = min(EDGE_WEIGHT_MAX, EDGE_WEIGHT_PER_MATCH * inliers)
            source = "keypoints"
            via_a = via_b = wall_a = wall_b = None
            if doorway_result:
                source = "keypoints+doorway"
                _, _, wall_a, wall_b, via_a, via_b = doorway_result
                confidence += EDGE_WEIGHT_DOORWAY
            edges.append(StitchEdge(
                match.room_a, match.room_b, t, yaw, min(confidence, EDGE_WEIGHT_MAX * 2),
                source, via_a, via_b, wall_a, wall_b, inliers,
            ))
        elif doorway_result:
            t, yaw, wall_a, wall_b, via_a, via_b = doorway_result
            edges.append(StitchEdge(
                match.room_a, match.room_b, t, yaw, EDGE_WEIGHT_DOORWAY,
                "doorway", via_a, via_b, wall_a, wall_b, 0,
            ))
        else:
            reason = (
                f"only {len(match.keypoint_matches)} keypoint match(es), no doorway match"
                if match.keypoint_matches else
                f"dinov2 similarity {match.dinov2_similarity:.2f}, no doorway match"
            )
            rejected.append((match.room_a, match.room_b, reason))

    return edges, rejected


def _dominant_wall_angle(rooms_by_id: Dict[str, RoomForStitch], global_yaw: Dict[str, float]) -> float:
    """Circular mean (mod 90 deg) of every room's own wall directions, each
    carried into the global frame by that room's solved yaw -- the same
    Manhattan-clustering idea single-room layout already uses, applied once
    across the whole assembled property.
    """
    angles, weights = [], []
    for room_id, room in rooms_by_id.items():
        yaw = global_yaw[room_id]
        for wall in room.walls:
            direction = np.array(wall.end) - np.array(wall.start)
            length = np.linalg.norm(direction)
            if length < 1e-6:
                continue
            angle = np.arctan2(direction[1], direction[0]) + yaw
            angles.append(angle)
            weights.append(length)
    if not angles:
        return 0.0
    angles_arr, weights_arr = np.array(angles), np.array(weights)
    resultant = np.sum(weights_arr * np.exp(4j * angles_arr))
    return float(np.angle(resultant) / 4.0)


def _resolve_overlaps(
    polygons: Dict[str, Polygon], anchor_id: str,
) -> Tuple[Dict[str, Tuple[float, float]], bool, int]:
    """Push rooms apart along their centroid axis until no two overlap.

    The anchor room never moves (it defines the property's origin); every
    other room can be nudged. A push large enough to need >200 steps is
    treated as unresolvable and reported as such rather than looped forever.
    """
    shifts = {room_id: (0.0, 0.0) for room_id in polygons}
    current = {room_id: poly for room_id, poly in polygons.items()}
    ids = [r for r in polygons if r != anchor_id]

    for iteration in range(OVERLAP_MAX_ITERATIONS):
        worst_pair, worst_area = None, OVERLAP_AREA_TOLERANCE_M2
        for i, a in enumerate(ids + [anchor_id]):
            for b in ids:
                if a == b:
                    continue
                key = tuple(sorted((a, b)))
                overlap = current[a].intersection(current[b]).area
                if overlap > worst_area:
                    worst_area, worst_pair = overlap, (a, b)
        if worst_pair is None:
            return shifts, True, iteration

        a, b = worst_pair
        mover = b if b != anchor_id else a
        other = a if mover == b else b
        ca, cb = current[mover].centroid, current[other].centroid
        direction = np.array([ca.x - cb.x, ca.y - cb.y])
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-6 else np.array([1.0, 0.0])
        dx, dy = direction * OVERLAP_PUSH_STEP_M
        sx, sy = shifts[mover]
        shifts[mover] = (sx + dx, sy + dy)
        current[mover] = affine_transform(polygons[mover], [1, 0, 0, 1, sx + dx, sy + dy])

    return shifts, False, OVERLAP_MAX_ITERATIONS


def build_stitch_graph(
    rooms: Sequence[RoomForStitch], matches: Sequence[RoomPairMatch],
    anchor_room_id: Optional[str] = None,
) -> StitchResult:
    """Match edges -> global pose graph -> Manhattan snap -> overlap fix."""
    rooms_by_id = {r.room_id: r for r in rooms}
    room_ids = [r.room_id for r in rooms]
    anchor = anchor_room_id or room_ids[0]
    anchor_index = room_ids.index(anchor)

    edges, rejected = build_edges(matches, rooms_by_id)
    if not edges:
        log.warning("no room pair produced a usable transform; every room stays at the origin")

    xyz0 = np.zeros((len(room_ids), 3))
    yaw0 = np.zeros(len(room_ids))
    graph_edges = []
    for edge in edges:
        ia, ib = room_ids.index(edge.room_a), room_ids.index(edge.room_b)
        rel_xyz = np.array([edge.relative_xy[0], edge.relative_xy[1], 0.0])
        graph_edges.append((ia, ib, rel_xyz, edge.relative_yaw, edge.confidence))

    if graph_edges and len(room_ids) > 1:
        if anchor_index != 0:
            # _optimize_pose_graph always fixes node 0; relabel so the anchor is node 0.
            order = [anchor_index] + [i for i in range(len(room_ids)) if i != anchor_index]
            remap = {old: new for new, old in enumerate(order)}
            graph_edges = [(remap[a], remap[b], r, y, w) for a, b, r, y, w in graph_edges]
            opt_xyz, opt_yaw = _optimize_pose_graph(xyz0[order], yaw0[order], graph_edges)
            inverse = {new: old for old, new in remap.items()}
            final_xyz = np.zeros_like(opt_xyz)
            final_yaw = np.zeros_like(opt_yaw)
            for new_idx, old_idx in inverse.items():
                final_xyz[old_idx] = opt_xyz[new_idx]
                final_yaw[old_idx] = opt_yaw[new_idx]
            opt_xyz, opt_yaw = final_xyz, final_yaw
        else:
            opt_xyz, opt_yaw = _optimize_pose_graph(xyz0, yaw0, graph_edges)
    else:
        opt_xyz, opt_yaw = xyz0, yaw0

    global_yaw = {room_id: float(opt_yaw[i]) for i, room_id in enumerate(room_ids)}
    global_xy = {room_id: opt_xyz[i, :2].copy() for i, room_id in enumerate(room_ids)}

    manhattan_rotation = _dominant_wall_angle(rooms_by_id, global_yaw)
    # Snap the whole property by the same amount, so every room's own already-
    # axis-aligned polygon stays axis-aligned in the shared frame too.
    snap = -_wrap(manhattan_rotation)
    if abs(snap) > 1e-6:
        c, s = np.cos(snap), np.sin(snap)
        for room_id in room_ids:
            global_yaw[room_id] = _wrap(global_yaw[room_id] + snap)
            x, y = global_xy[room_id]
            global_xy[room_id] = np.array([c * x - s * y, s * x + c * y])

    polygons = {
        room_id: affine_transform(
            rooms_by_id[room_id].polygon,
            [np.cos(global_yaw[room_id]), -np.sin(global_yaw[room_id]),
             np.sin(global_yaw[room_id]), np.cos(global_yaw[room_id]),
             global_xy[room_id][0], global_xy[room_id][1]],
        )
        for room_id in room_ids
    }
    shifts, resolved, iterations = _resolve_overlaps(polygons, anchor)
    for room_id, (dx, dy) in shifts.items():
        global_xy[room_id] = global_xy[room_id] + np.array([dx, dy])

    return StitchResult(
        room_ids=room_ids, global_xy=global_xy, global_yaw=global_yaw,
        edges_used=edges, edges_rejected=rejected,
        manhattan_rotation_rad=float(manhattan_rotation),
        overlap_resolved=resolved, overlap_iterations=iterations,
    )
