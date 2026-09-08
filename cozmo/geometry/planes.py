"""Floor and ceiling planes.

The floor is easy: it is the largest horizontal plane and every capture has one
under the operator's feet. The ceiling is the hard case and the interesting one.

A handheld walkthrough held at chest height, pointed at walls and furniture,
often never sees the ceiling at all. In the sample capture only 3.7% of points
sit above camera height. A pipeline that fits a ceiling plane to that 3.7%
returns a confident number derived from noise, a light fitting and the top of a
door frame. So there are two paths, and which one ran is recorded in the plan:

``measured_plane``      a real ceiling plane with real support
``wall_extrapolation``  no ceiling; the walls' observed tops are extrapolated
                        upward and the interval widens substantially

The second path is a lower bound dressed as an estimate, and the interval says
so: it is deliberately asymmetric, because the ceiling can only be at or above
where the walls stopped being observed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
from scipy import ndimage

from ._numeric import quiet_fp

log = logging.getLogger("cozmo.geometry.planes")

UP = np.array([0.0, 1.0, 0.0])

# A plane counts as horizontal when its normal is within ~18 deg of vertical.
HORIZONTAL_COS = 0.95

FLOOR_DISTANCE_M = 0.02
FLOOR_SEED_BIN_M = 0.02
# The floor seed is the lowest height band holding this fraction of the densest
# band's points -- low enough to ignore thin noise below the floor, high enough
# that furniture cannot outvote the floor it stands on.
FLOOR_SEED_MIN_FRACTION = 0.20
FLOOR_SEED_BAND_M = 0.08

# A ceiling must clear this height above the floor to be believable at all.
CEILING_MIN_ABOVE_FLOOR_M = 1.90
CEILING_DISTANCE_M = 0.04
# ... and needs this much support, as a fraction of all points, to be trusted.
CEILING_MIN_SUPPORT_RATIO = 0.02
CEILING_MIN_SUPPORT_POINTS = 400
# A ceiling must have this fraction of its cells fully surrounded by other
# ceiling cells. A ring of wall-tops has almost none.
CEILING_MIN_INTERIOR = 0.15

# Fallback interval widths (crude placeholders pending calibration).
CEILING_CI_MEASURED_M = 0.03
CEILING_CI_TIGHT_CONSENSUS = (-0.05, 0.35)
CEILING_CI_LOOSE_CONSENSUS = (-0.05, 0.80)
WALL_TOP_CONSENSUS_SPREAD_M = 0.15


@dataclass
class Plane:
    """A fitted plane: ``normal . p + d = 0``, normal unit length."""

    normal: np.ndarray
    d: float
    inliers: int
    total: int
    inlier_indices: Optional[np.ndarray] = None

    @property
    def inlier_ratio(self) -> float:
        return self.inliers / self.total if self.total else 0.0

    @property
    def is_horizontal(self) -> bool:
        return abs(float(np.dot(self.normal, UP))) >= HORIZONTAL_COS

    @property
    def height(self) -> float:
        """Signed height where the plane crosses the vertical axis."""
        ny = float(self.normal[1])
        if abs(ny) < 1e-6:
            raise ValueError("vertical plane has no single height")
        return -self.d / ny

    def distance_to(self, points: np.ndarray) -> np.ndarray:
        return np.abs(points @ self.normal + self.d)

    def summary(self) -> Dict[str, Any]:
        return {
            "normal": [round(float(v), 4) for v in self.normal],
            "height_m": round(self.height, 4) if self.is_horizontal else None,
            "inliers": self.inliers,
            "inlier_ratio": round(self.inlier_ratio, 4),
        }


@dataclass
class CeilingEstimate:
    """Ceiling height above the floor, and how honestly it was arrived at."""

    height_above_floor_m: float
    ci_95: Tuple[float, float]
    method: str                       # measured_plane | wall_extrapolation | prior
    support_points: int = 0
    wall_top_spread_m: Optional[float] = None
    note: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def measured(self) -> bool:
        return self.method == "measured_plane"

    def summary(self) -> Dict[str, Any]:
        return {
            "height_above_floor_m": round(self.height_above_floor_m, 4),
            "ci_95_m": [round(v, 4) for v in self.ci_95],
            "method": self.method,
            "support_points": self.support_points,
            "wall_top_spread_m": (
                round(self.wall_top_spread_m, 4) if self.wall_top_spread_m is not None else None
            ),
            "note": self.note,
            **self.stats,
        }


def _segment(points: np.ndarray, distance_m: float, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """One RANSAC plane segmentation. Returns (model, inlier indices)."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    model, inliers = cloud.segment_plane(
        distance_threshold=distance_m, ransac_n=3, num_iterations=1000
    )
    return np.asarray(model, dtype=float), np.asarray(inliers, dtype=int)


def _plane_from_model(model: np.ndarray, inliers: Optional[np.ndarray], total: int) -> Plane:
    a, b, c, d = model
    normal = np.array([a, b, c], dtype=float)
    norm = np.linalg.norm(normal)
    if norm == 0:
        raise ValueError("degenerate plane model")
    return Plane(
        normal=normal / norm,
        d=float(d) / norm,
        inliers=0 if inliers is None else len(inliers),
        total=total,
        inlier_indices=inliers,
    )


def fit_floor(points: np.ndarray, distance_m: float = FLOOR_DISTANCE_M) -> Plane:
    """RANSAC floor plane, seeded by the height histogram.

    Running RANSAC on the raw cloud does not reliably return the floor: in a
    furnished room the largest single plane is often a wall, and on the sample
    capture unseeded RANSAC settled 6-9 cm above the real floor. So the vertical
    histogram picks the seed -- the *lowest* height band with substantial support,
    not the densest, since a bed or a desk can out-populate the floor it stands
    on -- and RANSAC then fits the plane properly within that band, which is what
    recovers the tilt.
    """
    if len(points) < 100:
        raise ValueError(f"too few points to fit a floor plane: {len(points)}")

    heights = points[:, 1]
    edges = np.arange(heights.min(), heights.max() + FLOOR_SEED_BIN_M, FLOOR_SEED_BIN_M)
    counts, _ = np.histogram(heights, bins=edges)
    if counts.max() == 0:
        raise ValueError("degenerate height histogram")

    supported = np.flatnonzero(counts >= FLOOR_SEED_MIN_FRACTION * counts.max())
    seed_bin = int(supported[0]) if len(supported) else int(np.argmax(counts))
    seed_height = float((edges[seed_bin] + edges[seed_bin + 1]) / 2)

    band = points[np.abs(heights - seed_height) <= FLOOR_SEED_BAND_M]
    plane: Optional[Plane] = None
    if len(band) >= 100:
        model, inliers = _segment(band, distance_m)
        if len(inliers) >= 50:
            candidate = _plane_from_model(model, None, len(points))
            if candidate.is_horizontal:
                plane = candidate

    if plane is None:
        log.warning("floor: RANSAC gave no horizontal plane in the seed band; using the band height")
        plane = Plane(normal=UP.copy(), d=-seed_height, inliers=0, total=len(points))

    # Orient upward so height and side tests are unambiguous, then recount
    # inliers against the whole cloud rather than the seed band.
    if float(np.dot(plane.normal, UP)) < 0:
        plane.normal = -plane.normal
        plane.d = -plane.d
    with quiet_fp():
        residual = points @ plane.normal + plane.d
    inlier_indices = np.flatnonzero(np.abs(residual) <= distance_m)
    plane.inliers = int(len(inlier_indices))
    plane.total = len(points)
    plane.inlier_indices = inlier_indices

    tilt_deg = float(np.degrees(np.arccos(np.clip(abs(np.dot(plane.normal, UP)), -1, 1))))
    log.info(
        "floor plane at y=%.3f m, tilt %.2f deg (%d inliers, %.1f%%)",
        plane.height, tilt_deg, plane.inliers, 100 * plane.inlier_ratio,
    )
    return plane


def _patchiness(points: np.ndarray, cell_m: float = 0.15) -> Tuple[float, float]:
    """Is this horizontal point set a surface, or the outline of one?

    A ceiling covers area; the ring of points where walls stop only outlines it.
    Coverage alone cannot tell them apart -- a partly-seen ceiling covers little
    of its own footprint, and a ring covers a surprising amount of it. What does
    separate them is erosion: a one-cell-wide ring has no cell whose neighbours
    are all occupied, while any real patch of ceiling does.

    Returns ``(fill, interior_fraction)``.
    """
    if len(points) < 10:
        return 0.0, 0.0
    xz = points[:, [0, 2]]
    span = xz.max(axis=0) - xz.min(axis=0)
    if np.any(span < cell_m):
        return 0.0, 0.0

    cells = np.floor((xz - xz.min(axis=0)) / cell_m).astype(int)
    shape = (int(np.ceil(span[1] / cell_m)) + 1, int(np.ceil(span[0] / cell_m)) + 1)
    grid = np.zeros(shape, dtype=bool)
    grid[np.clip(cells[:, 1], 0, shape[0] - 1), np.clip(cells[:, 0], 0, shape[1] - 1)] = True

    occupied = int(grid.sum())
    fill = occupied / grid.size if grid.size else 0.0
    eroded = int(ndimage.binary_erosion(grid, structure=np.ones((3, 3))).sum())
    return fill, (eroded / occupied if occupied else 0.0)


def fit_ceiling(
    points: np.ndarray,
    floor_height: float,
    camera_height: float,
    distance_m: float = CEILING_DISTANCE_M,
) -> Optional[Plane]:
    """Try for a real ceiling plane. Returns None when the support is not there.

    Seeded from the height histogram for the same reason the floor is: run plain
    RANSAC on everything above camera height and it returns a wall, because in a
    room whose ceiling was barely seen there is far more wall up there than
    ceiling.
    """
    threshold = max(camera_height + 0.15, floor_height + CEILING_MIN_ABOVE_FLOOR_M)
    above = points[points[:, 1] > threshold]
    required = max(CEILING_MIN_SUPPORT_POINTS, int(CEILING_MIN_SUPPORT_RATIO * len(points)))
    if len(above) < required:
        log.info("ceiling: only %d points above %.2f m, need %d", len(above), threshold, required)
        return None

    heights = above[:, 1]
    edges = np.arange(heights.min(), heights.max() + FLOOR_SEED_BIN_M, FLOOR_SEED_BIN_M)
    counts, _ = np.histogram(heights, bins=edges)
    if counts.max() == 0:
        return None
    supported = np.flatnonzero(counts >= FLOOR_SEED_MIN_FRACTION * counts.max())
    seed_bin = int(supported[-1])                       # the highest, not the densest
    seed_height = float((edges[seed_bin] + edges[seed_bin + 1]) / 2)

    band = above[np.abs(heights - seed_height) <= FLOOR_SEED_BAND_M]
    if len(band) < 50:
        return None
    model, inliers = _segment(band, distance_m)
    if len(inliers) < 50:
        return None
    plane = _plane_from_model(model, None, len(points))
    if not plane.is_horizontal:
        log.info("ceiling: best plane in the top band is not horizontal")
        return None
    if float(np.dot(plane.normal, UP)) < 0:
        plane.normal = -plane.normal
        plane.d = -plane.d

    with quiet_fp():
        residual = points @ plane.normal + plane.d
    inlier_indices = np.flatnonzero(np.abs(residual) <= distance_m)
    plane.inliers = int(len(inlier_indices))
    plane.inlier_indices = inlier_indices
    plane.total = len(points)

    if plane.inliers < required:
        log.info("ceiling: plane has %d inliers, need %d", plane.inliers, required)
        return None
    if plane.height - floor_height < CEILING_MIN_ABOVE_FLOOR_M:
        log.info("ceiling: candidate at %.2f m above floor is too low", plane.height - floor_height)
        return None

    fill, interior = _patchiness(points[inlier_indices])
    if interior < CEILING_MIN_INTERIOR:
        log.info(
            "ceiling: candidate at %.2f m is an outline, not a surface "
            "(interior fraction %.2f, need %.2f); this is the top edge of the walls",
            plane.height - floor_height, interior, CEILING_MIN_INTERIOR,
        )
        return None

    log.info("ceiling plane at %.3f m above floor (%d inliers, fill %.0f%%, interior %.2f)",
             plane.height - floor_height, plane.inliers, 100 * fill, interior)
    return plane


def estimate_ceiling(
    points: np.ndarray,
    floor: Plane,
    camera_height: float,
    wall_top_heights: Optional[Sequence[float]] = None,
    prior_height_m: float = 2.44,
) -> CeilingEstimate:
    """Measure the ceiling if it is there; otherwise extrapolate the walls up."""
    floor_height = floor.height

    plane = fit_ceiling(points, floor_height, camera_height)
    if plane is not None:
        height = plane.height - floor_height
        return CeilingEstimate(
            height_above_floor_m=height,
            ci_95=(height - CEILING_CI_MEASURED_M, height + CEILING_CI_MEASURED_M),
            method="measured_plane",
            support_points=plane.inliers,
            note="Ceiling plane fitted directly; interval is the plane-fit placeholder.",
            stats={"plane": plane.summary()},
        )

    tops = [float(t) for t in (wall_top_heights or []) if np.isfinite(t)]
    if tops:
        consensus = float(np.median(tops))
        highest = float(np.max(tops))
        spread = float(highest - np.min(tops)) if len(tops) > 1 else 0.0
        agree = spread <= WALL_TOP_CONSENSUS_SPREAD_M and len(tops) >= 3
        stats = {
            "wall_top_consensus_m": round(consensus, 4),
            "highest_wall_point_m": round(highest, 4),
            "wall_tops_m": [round(t, 3) for t in tops],
        }

        if agree and consensus >= CEILING_MIN_ABOVE_FLOOR_M:
            # Walls stop together, high up: that agreement is the wall/ceiling
            # junction, and extrapolating to it is a real measurement.
            lo_off, hi_off = CEILING_CI_TIGHT_CONSENSUS
            return CeilingEstimate(
                height_above_floor_m=consensus + 0.05,
                ci_95=(consensus + lo_off, consensus + hi_off),
                method="wall_extrapolation",
                support_points=len(tops),
                wall_top_spread_m=spread,
                note=(
                    f"No ceiling plane, but walls stop together at {consensus:.2f} m "
                    f"(spread {spread:.2f} m), read as the wall/ceiling junction."
                ),
                stats=stats,
            )

        # Walls stop at inconsistent heights, or too low to be a ceiling. This is
        # where the scan ended, not where the room does. The only defensible
        # claim is a lower bound: the ceiling is at least as high as the highest
        # point actually seen on a wall. The interval runs from that bound to a
        # structural prior, and the estimate is the prior, not the observation --
        # reporting 1.7 m here because the operator never looked up would be a
        # confident answer to a question the capture did not ask.
        lower = max(highest, CEILING_MIN_ABOVE_FLOOR_M * 0.9)
        upper = max(prior_height_m + 0.55, lower + 0.35)
        height = min(max(prior_height_m, lower + 0.05), upper)
        return CeilingEstimate(
            height_above_floor_m=height,
            ci_95=(lower, upper),
            method="scan_cutoff_prior",
            support_points=len(tops),
            wall_top_spread_m=spread,
            note=(
                f"No ceiling plane, and wall tops disagree by {spread:.2f} m "
                f"(median {consensus:.2f} m, highest {highest:.2f} m): the capture stopped "
                f"below the ceiling. Reported height is a structural prior bounded below by "
                f"the highest observed wall point; treat it as an interval, not a measurement."
            ),
            stats=stats,
        )

    log.warning("no ceiling and no wall tops; falling back to a structural prior")
    return CeilingEstimate(
        height_above_floor_m=prior_height_m,
        ci_95=(prior_height_m - 0.30, prior_height_m + 0.30),
        method="prior",
        support_points=0,
        note=(
            "Neither a ceiling plane nor wall tops were available. This is a prior, "
            "not a measurement, and should not be reported as one."
        ),
    )
