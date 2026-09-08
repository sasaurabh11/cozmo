"""Room layout: point cloud in, wall polygon out.

The pipeline, in order:

1. **Floor frame.** The fitted floor plane gives an up axis and a 2D basis.
   Heights are signed distances from that plane, not raw ``y`` -- the sample
   capture's floor is tilted 0.7 deg, which is 11 cm of error across a 9 m room
   if you use ``y`` directly.
2. **Dominant directions.** Point normals in the wall band are histogrammed to
   find the room's own axes. Rooms are assumed rectilinear, but *not* aligned to
   the capture's world axes: the sample room sits 23 deg off ARKit's frame, and
   taking the pose frame as given inflates the footprint.
3. **Wall lines by RANSAC**, locked to those two directions, fitted and removed
   one at a time.
4. **Cell arrangement.** Those lines cut the plane into cells; a cell is part of
   the room when the interior evidence (floor points, and the camera path, which
   is interior by definition) covers enough of it. The union of kept cells is the
   room polygon.

Step 4 is why the output is a polygon and not a bounding box: an L-shaped room
is just a different set of kept cells.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
from scipy import ndimage
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

from ._numeric import quiet_fp
from .planes import Plane

log = logging.getLogger("cozmo.geometry.layout")

# Wall band: high enough to clear skirting and most floor clutter, low enough to
# stay under the height where a handheld capture stops seeing walls.
WALL_BAND_MIN_M = 0.35
WALL_BAND_MAX_M = 1.80
FLOOR_BAND_M = 0.10

NORMAL_RADIUS_M = 0.12
NORMAL_MAX_NN = 30
# |n . up| below this counts as a vertical surface (within ~70 deg of horizontal).
VERTICAL_NORMAL_MAX = 0.35

LINE_TOLERANCE_M = 0.05
LINE_MIN_INLIERS = 150
LINE_MIN_EXTENT_M = 0.60
LINE_MAX_COUNT = 14
LINE_RANSAC_SAMPLES = 256

GRID_RES_M = 0.05
TRAJECTORY_RADIUS_M = 0.35
CELL_COVERAGE_MIN = 0.30
CELL_MIN_AREA_M2 = 0.20
# Wall lines closer together than this are the same wall seen twice (front and
# back faces, or two RANSAC rounds splitting one surface).
LINE_MERGE_M = 0.15
# Points this far above the floor are used to measure how high a wall was
# actually observed -- the number the ceiling fallback extrapolates from.
WALL_TOP_MIN_HEIGHT_M = 0.50

POLYGON_SIMPLIFY_M = 0.02
# Morphological cleanup of the cell union. A closing fills the one-cell notches
# a doorway or a poorly-seen corner leaves in the arrangement; an opening then
# removes slivers narrower than a real alcove. Mitred joins keep corners square.
POLYGON_CLEAN_M = 0.12
COLLINEAR_TOLERANCE_M = 0.03
MIN_WALL_LENGTH_M = 0.25


@dataclass
class FloorFrame:
    """A metric 2D frame lying in the floor plane."""

    origin: np.ndarray      # a point on the floor plane
    up: np.ndarray          # unit normal, pointing up
    e1: np.ndarray          # in-plane axis, aligned to the room's own direction
    e2: np.ndarray          # in-plane axis, e1 x up
    rotation_rad: float     # rotation of e1 away from the capture's world axes

    def project(self, points: np.ndarray) -> np.ndarray:
        """World points -> (N, 2) coordinates in the floor plane."""
        rel = points - self.origin
        with quiet_fp():
            return np.stack([rel @ self.e1, rel @ self.e2], axis=1)

    def height(self, points: np.ndarray, plane: Plane) -> np.ndarray:
        """Signed height above the floor plane."""
        with quiet_fp():
            return points @ plane.normal + plane.d

    def to_world(self, uv: np.ndarray, height: float = 0.0) -> np.ndarray:
        """(N, 2) plane coordinates -> world points at a given height."""
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        return self.origin + uv[:, :1] * self.e1 + uv[:, 1:2] * self.e2 + height * self.up


@dataclass
class WallLine:
    """An infinite line in the floor plane, locked to one of the room's axes."""

    axis: int               # 0: constant u (runs along v); 1: constant v
    coord: float
    extent: Tuple[float, float]
    inliers: int
    top_height_m: float

    @property
    def length(self) -> float:
        return self.extent[1] - self.extent[0]


@dataclass
class WallSegment:
    """One edge of the room polygon."""

    start: Tuple[float, float]
    end: Tuple[float, float]
    length_m: float
    normal: Tuple[float, float]      # outward-ish 2D normal in the floor frame
    top_height_m: Optional[float] = None
    support_points: int = 0

    @property
    def direction(self) -> np.ndarray:
        d = np.array(self.end) - np.array(self.start)
        norm = np.linalg.norm(d)
        return d / norm if norm else d


@dataclass
class LayoutResult:
    frame: FloorFrame
    polygon: Polygon
    walls: List[WallSegment]
    lines: List[WallLine]
    floor_area_m2: float
    perimeter_m: float
    manhattan_snap: bool
    rotation_deg: float
    wall_band: np.ndarray = field(repr=False, default_factory=lambda: np.empty((0, 3)))
    wall_band_uv: np.ndarray = field(repr=False, default_factory=lambda: np.empty((0, 2)))
    wall_band_height: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    # The whole cloud in floor-frame coordinates. Opening detection needs this
    # rather than the wall band: a door runs from the floor to about 2 m, and the
    # band starts at 0.35 m and stops at 1.80 m, so a door seen through the band
    # is a hole that starts too high to be a door and ends before its head.
    points_uv: np.ndarray = field(repr=False, default_factory=lambda: np.empty((0, 2)))
    points_height: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def wall_top_heights(self) -> List[float]:
        return [w.top_height_m for w in self.walls if w.top_height_m is not None]

    def summary(self) -> Dict[str, Any]:
        return {
            "floor_area_m2": round(self.floor_area_m2, 4),
            "perimeter_m": round(self.perimeter_m, 4),
            "wall_count": len(self.walls),
            "line_count": len(self.lines),
            "manhattan_snap": self.manhattan_snap,
            "rotation_deg": round(self.rotation_deg, 3),
            "wall_lengths_m": [round(w.length_m, 3) for w in self.walls],
            **self.stats,
        }


# --------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------


def estimate_normals(points: np.ndarray) -> np.ndarray:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS_M, max_nn=NORMAL_MAX_NN)
    )
    return np.asarray(cloud.normals)


def dominant_rotation(normals_2d: np.ndarray) -> Tuple[float, float]:
    """Room rotation from wall normals, and how strongly the room agrees on it.

    Directions 90 deg apart are the same wall family, so angles are averaged
    after multiplying by 4: the resultant length is a Manhattan-ness score, and
    a low score means the room does not have two dominant axes.
    """
    if len(normals_2d) == 0:
        return 0.0, 0.0
    theta = np.arctan2(normals_2d[:, 1], normals_2d[:, 0])
    resultant = np.mean(np.exp(4j * theta))
    return float(np.angle(resultant) / 4.0), float(np.abs(resultant))


def _fit_axis_lines(
    perpendicular: np.ndarray,
    along: np.ndarray,
    heights: np.ndarray,
    axis: int,
    seed: int = 0,
) -> List[WallLine]:
    """RANSAC lines at constant ``perpendicular``, fitted and removed one by one."""
    lines: List[WallLine] = []
    remaining = np.ones(len(perpendicular), dtype=bool)
    rng = np.random.default_rng(seed)

    for _ in range(LINE_MAX_COUNT):
        idx = np.flatnonzero(remaining)
        if len(idx) < LINE_MIN_INLIERS:
            break

        samples = rng.choice(idx, size=min(LINE_RANSAC_SAMPLES, len(idx)), replace=False)
        best_count, best_coord = 0, None
        for s in samples:
            count = int(np.count_nonzero(np.abs(perpendicular[idx] - perpendicular[s]) <= LINE_TOLERANCE_M))
            if count > best_count:
                best_count, best_coord = count, float(perpendicular[s])
        if best_coord is None or best_count < LINE_MIN_INLIERS:
            break

        inliers = idx[np.abs(perpendicular[idx] - best_coord) <= LINE_TOLERANCE_M]
        refined = float(np.median(perpendicular[inliers]))          # least-noise refit
        inliers = idx[np.abs(perpendicular[idx] - refined) <= LINE_TOLERANCE_M]
        if len(inliers) < LINE_MIN_INLIERS:
            remaining[inliers] = False
            continue

        span = (float(along[inliers].min()), float(along[inliers].max()))
        if span[1] - span[0] >= LINE_MIN_EXTENT_M:
            lines.append(WallLine(
                axis=axis, coord=refined, extent=span, inliers=int(len(inliers)),
                top_height_m=float(np.percentile(heights[inliers], 98)),
            ))
        remaining[inliers] = False

    return lines


def _merge_lines(lines: Sequence[WallLine]) -> List[WallLine]:
    """Collapse lines that describe the same wall.

    A wall seen from both sides, or split across two RANSAC rounds, otherwise
    cuts the arrangement into slivers that no cell can fill.
    """
    merged: List[WallLine] = []
    for axis in (0, 1):
        group = sorted((l for l in lines if l.axis == axis), key=lambda l: l.coord)
        cluster: List[WallLine] = []
        for line in group:
            if cluster and line.coord - cluster[-1].coord > LINE_MERGE_M:
                merged.append(_collapse(cluster))
                cluster = []
            cluster.append(line)
        if cluster:
            merged.append(_collapse(cluster))
    return merged


def _collapse(cluster: Sequence[WallLine]) -> WallLine:
    weights = np.array([l.inliers for l in cluster], dtype=float)
    coords = np.array([l.coord for l in cluster], dtype=float)
    return WallLine(
        axis=cluster[0].axis,
        coord=float(np.average(coords, weights=weights)),
        extent=(min(l.extent[0] for l in cluster), max(l.extent[1] for l in cluster)),
        inliers=int(weights.sum()),
        top_height_m=max(l.top_height_m for l in cluster),
    )


def _interior_mask(
    floor_uv: np.ndarray, trajectory_uv: np.ndarray, bounds: Tuple[float, float, float, float]
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Binary grid of where the room's interior is.

    Floor points are direct evidence; the camera path is interior by definition
    and fills in the parts of the floor a handheld capture never looks at.
    """
    u0, v0, u1, v1 = bounds
    width = max(1, int(np.ceil((u1 - u0) / GRID_RES_M)))
    height = max(1, int(np.ceil((v1 - v0) / GRID_RES_M)))
    mask = np.zeros((height, width), dtype=bool)

    def stamp(points: np.ndarray, radius_cells: int) -> None:
        if len(points) == 0:
            return
        cu = np.clip(((points[:, 0] - u0) / GRID_RES_M).astype(int), 0, width - 1)
        cv = np.clip(((points[:, 1] - v0) / GRID_RES_M).astype(int), 0, height - 1)
        if radius_cells <= 0:
            mask[cv, cu] = True
            return
        for du in range(-radius_cells, radius_cells + 1):
            for dv in range(-radius_cells, radius_cells + 1):
                if du * du + dv * dv > radius_cells * radius_cells:
                    continue
                mask[np.clip(cv + dv, 0, height - 1), np.clip(cu + du, 0, width - 1)] = True

    stamp(floor_uv, 0)
    stamp(trajectory_uv, int(round(TRAJECTORY_RADIUS_M / GRID_RES_M)))

    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5)))
    mask = ndimage.binary_fill_holes(mask)
    return mask, (u0, v0)


def _cells_from_lines(
    lines: Sequence[WallLine], bounds: Tuple[float, float, float, float]
) -> Tuple[List[float], List[float]]:
    """Cut lines for the arrangement, one axis at a time.

    Only fitted wall lines are used. Falling back to the data extent would let
    the room's outer edge sit wherever the point cloud happened to stop, which
    on the synthetic fixture stretched a 3.60 m wall to 3.89 m: the extra came
    from cells between the real wall and the edge of the data, not from the room.
    The extent is used only on an axis with too few lines to bound anything.
    """
    u0, v0, u1, v1 = bounds
    us = sorted({l.coord for l in lines if l.axis == 0 and u0 <= l.coord <= u1})
    vs = sorted({l.coord for l in lines if l.axis == 1 and v0 <= l.coord <= v1})
    if len(us) < 2:
        log.warning("only %d wall line(s) across u; falling back to the data extent", len(us))
        us = sorted({u0, u1, *us})
    if len(vs) < 2:
        log.warning("only %d wall line(s) across v; falling back to the data extent", len(vs))
        vs = sorted({v0, v1, *vs})
    return us, vs


def _merge_collinear(ring: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Drop vertices that sit on the straight line between their neighbours."""
    pts = [tuple(p) for p in ring]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        return pts

    kept: List[Tuple[float, float]] = []
    for i, current in enumerate(pts):
        prev_pt = np.array(pts[i - 1])
        next_pt = np.array(pts[(i + 1) % len(pts)])
        cur = np.array(current)
        a, b = cur - prev_pt, next_pt - cur
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            continue
        ua, ub = a / na, b / nb
        cross = abs(float(ua[0] * ub[1] - ua[1] * ub[0]))
        if cross * min(na, nb) > COLLINEAR_TOLERANCE_M:
            kept.append(current)
    return kept if len(kept) >= 3 else pts


def _walls_from_polygon(polygon: Polygon, lines: Sequence[WallLine]) -> List[WallSegment]:
    ring = _merge_collinear(list(polygon.exterior.coords))
    centroid = np.array(polygon.centroid.coords[0])

    walls: List[WallSegment] = []
    for i, start in enumerate(ring):
        end = ring[(i + 1) % len(ring)]
        seg = np.array(end) - np.array(start)
        length = float(np.linalg.norm(seg))
        if length < MIN_WALL_LENGTH_M:
            continue
        direction = seg / length
        normal = np.array([-direction[1], direction[0]])
        midpoint = (np.array(start) + np.array(end)) / 2
        if np.dot(normal, midpoint - centroid) < 0:      # point away from the room
            normal = -normal

        # Attribute the nearest fitted line's observed top height to this wall.
        top, support = None, 0
        best = None
        for line in lines:
            on_axis = abs(direction[0]) < 0.3 if line.axis == 0 else abs(direction[1]) < 0.3
            if not on_axis:
                continue
            distance = abs(midpoint[line.axis] - line.coord)
            if best is None or distance < best[0]:
                best = (distance, line)
        if best is not None and best[0] <= 0.25:
            top, support = best[1].top_height_m, best[1].inliers

        walls.append(WallSegment(
            start=(float(start[0]), float(start[1])), end=(float(end[0]), float(end[1])),
            length_m=length, normal=(float(normal[0]), float(normal[1])),
            top_height_m=top, support_points=support,
        ))
    return walls


def _trajectory_count(geom: Polygon, trajectory_uv: np.ndarray) -> int:
    """How many camera positions fall inside a candidate region."""
    if len(trajectory_uv) == 0:
        return 0
    u0, v0, u1, v1 = geom.bounds
    inside = (
        (trajectory_uv[:, 0] >= u0) & (trajectory_uv[:, 0] <= u1)
        & (trajectory_uv[:, 1] >= v0) & (trajectory_uv[:, 1] <= v1)
    )
    return int(inside.sum())


def extract_layout(
    points: np.ndarray,
    floor: Plane,
    trajectory: np.ndarray,
    manhattan_snap: bool = True,
    seed: int = 0,
) -> LayoutResult:
    """Fit the room polygon.

    ``manhattan_snap=False`` is the drift ablation: the room's own axes are not
    estimated and the capture's world axes are used as they came out of ARKit.
    """
    with quiet_fp():
        heights = points @ floor.normal + floor.d

    band_mask = (heights >= WALL_BAND_MIN_M) & (heights <= WALL_BAND_MAX_M)
    band = points[band_mask]
    band_heights = heights[band_mask]
    if len(band) < 500:
        raise ValueError(f"only {len(band)} points in the wall band; cannot fit a layout")

    normals = estimate_normals(band)
    up = floor.normal / np.linalg.norm(floor.normal)
    with quiet_fp():
        vertical = np.abs(normals @ up) < VERTICAL_NORMAL_MAX
    if vertical.sum() < 200:
        raise ValueError("no vertical surfaces found; the capture has no usable walls")

    # A provisional in-plane basis, before the room's own rotation is known.
    seed_axis = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed_axis, up))) > 0.9:
        seed_axis = np.array([0.0, 0.0, 1.0])
    a1 = seed_axis - np.dot(seed_axis, up) * up
    a1 /= np.linalg.norm(a1)
    a2 = np.cross(up, a1)

    with quiet_fp():
        normals_2d = np.stack([normals[vertical] @ a1, normals[vertical] @ a2], axis=1)
    lengths = np.linalg.norm(normals_2d, axis=1, keepdims=True)
    normals_2d = normals_2d / np.maximum(lengths, 1e-9)

    if manhattan_snap:
        rotation, manhattan_score = dominant_rotation(normals_2d)
    else:
        # The ablation arm: no correction, so the pose frame's axes are used
        # exactly as ARKit produced them.
        rotation, manhattan_score = 0.0, float(np.abs(np.mean(np.exp(4j * np.arctan2(
            normals_2d[:, 1], normals_2d[:, 0])))))

    cos_r, sin_r = np.cos(rotation), np.sin(rotation)
    e1 = cos_r * a1 + sin_r * a2
    e2 = np.cross(up, e1)
    origin = -floor.d * floor.normal          # the floor plane's closest point to the world origin
    frame = FloorFrame(origin=origin, up=up, e1=e1, e2=e2, rotation_rad=rotation)

    band_uv = frame.project(band)
    wall_uv = band_uv[vertical]
    wall_heights = band_heights[vertical]
    with quiet_fp():
        wall_normals_2d = np.stack([normals[vertical] @ e1, normals[vertical] @ e2], axis=1)

    # Split walls by which room axis they face, then fit lines along each.
    faces_u = np.abs(wall_normals_2d[:, 0]) >= np.abs(wall_normals_2d[:, 1])
    lines = _fit_axis_lines(wall_uv[faces_u, 0], wall_uv[faces_u, 1], wall_heights[faces_u], axis=0, seed=seed)
    lines += _fit_axis_lines(wall_uv[~faces_u, 1], wall_uv[~faces_u, 0], wall_heights[~faces_u], axis=1, seed=seed + 1)
    if not lines:
        if manhattan_snap:
            raise ValueError("no wall lines survived RANSAC")
        # The ablation arm, on a room that is not aligned to the capture's world
        # axes: with no estimate of the room's own orientation there is nothing
        # for an axis-locked line fitter to lock onto, so the layout degrades to
        # the extent of the evidence in the pose frame. That is the honest
        # answer for "poses used as-is", and it is the number the drift gate is
        # asking about -- raising here would leave the ablation with no number
        # at all.
        log.warning(
            "no wall lines in the pose frame; the uncorrected layout falls back to "
            "the bounding extent of the observed points"
        )
    else:
        lines = _merge_lines(lines)

    # Line fitting works inside the wall band, but how high a wall was *seen*
    # has to be measured over the full cloud -- the band's own ceiling would
    # otherwise be reported as the wall's top, and the ceiling fallback would
    # inherit that as its answer.
    all_uv = frame.project(points)
    for line in lines:
        near = (
            (np.abs(all_uv[:, line.axis] - line.coord) <= LINE_TOLERANCE_M)
            & (all_uv[:, 1 - line.axis] >= line.extent[0])
            & (all_uv[:, 1 - line.axis] <= line.extent[1])
            & (heights >= WALL_TOP_MIN_HEIGHT_M)
        )
        if np.count_nonzero(near) >= 50:
            line.top_height_m = float(np.percentile(heights[near], 98))

    floor_uv = frame.project(points[np.abs(heights) <= FLOOR_BAND_M])
    trajectory_uv = frame.project(trajectory) if len(trajectory) else np.empty((0, 2))

    margin = 0.2
    extent_uv = np.vstack([band_uv, floor_uv, trajectory_uv]) if len(floor_uv) else band_uv
    bounds = (
        float(extent_uv[:, 0].min()) - margin, float(extent_uv[:, 1].min()) - margin,
        float(extent_uv[:, 0].max()) + margin, float(extent_uv[:, 1].max()) + margin,
    )
    mask, (u0, v0) = _interior_mask(floor_uv, trajectory_uv, bounds)

    us, vs = _cells_from_lines(lines, bounds)
    kept = []
    for i in range(len(us) - 1):
        for j in range(len(vs) - 1):
            cell = box(us[i], vs[j], us[i + 1], vs[j + 1])
            if cell.area < CELL_MIN_AREA_M2:
                continue
            cu0 = int((us[i] - u0) / GRID_RES_M); cu1 = int(np.ceil((us[i + 1] - u0) / GRID_RES_M))
            cv0 = int((vs[j] - v0) / GRID_RES_M); cv1 = int(np.ceil((vs[j + 1] - v0) / GRID_RES_M))
            patch = mask[max(cv0, 0):cv1, max(cu0, 0):cu1]
            walked = bool(len(trajectory_uv)) and bool(np.any(
                (trajectory_uv[:, 0] >= us[i]) & (trajectory_uv[:, 0] < us[i + 1])
                & (trajectory_uv[:, 1] >= vs[j]) & (trajectory_uv[:, 1] < vs[j + 1])
            ))
            # A cell the operator physically stood in is interior whatever the
            # floor evidence says: it is the one place we know is not a wall.
            if walked or (patch.size and patch.mean() >= CELL_COVERAGE_MIN):
                kept.append(cell)

    if not kept:
        raise ValueError("no cell of the wall arrangement is supported by interior evidence")

    merged = unary_union(kept).buffer(0)
    if isinstance(merged, MultiPolygon):
        # The component the operator actually walked, not merely the biggest:
        # a large well-scanned area seen through a doorway is not this room.
        merged = max(merged.geoms, key=lambda g: _trajectory_count(g, trajectory_uv) or g.area * 1e-6)

    cleaned = (
        merged.buffer(POLYGON_CLEAN_M, join_style=2)
        .buffer(-2 * POLYGON_CLEAN_M, join_style=2)
        .buffer(POLYGON_CLEAN_M, join_style=2)
    )
    if isinstance(cleaned, MultiPolygon):
        cleaned = max(cleaned.geoms, key=lambda g: g.area)
    if cleaned.is_empty or cleaned.area < 0.5 * merged.area:
        # The cleanup ate the room: keep the raw union rather than a fiction.
        log.warning("polygon cleanup removed too much area; keeping the raw cell union")
        cleaned = merged
    polygon = Polygon(cleaned.exterior).simplify(POLYGON_SIMPLIFY_M)

    walls = _walls_from_polygon(polygon, lines)
    result = LayoutResult(
        frame=frame,
        polygon=polygon,
        walls=walls,
        lines=lines,
        floor_area_m2=float(polygon.area),
        perimeter_m=float(polygon.exterior.length),
        manhattan_snap=manhattan_snap,
        rotation_deg=float(np.degrees(rotation)),
        wall_band=band,
        wall_band_uv=band_uv,
        wall_band_height=band_heights,
        points_uv=all_uv,
        points_height=heights,
        stats={
            "manhattan_score": round(manhattan_score, 4),
            "wall_band_points": int(len(band)),
            "vertical_points": int(vertical.sum()),
            "cells_kept": len(kept),
            "bounds_uv": [round(b, 3) for b in bounds],
        },
    )
    log.info(
        "layout: %.2f m2, %d walls, rotation %.1f deg, manhattan score %.2f",
        result.floor_area_m2, len(walls), result.rotation_deg, manhattan_score,
    )
    return result
