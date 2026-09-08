"""Doors and windows as holes in wall planes.

For each wall the points near its plane are binned into a wall-local occupancy
grid: ``u`` along the wall, ``v`` up from the floor. Walls return depth; doorways
and glazing do not, so an opening is where the wall stops returning.

The trap is that "no points here" has two causes -- an opening, and a piece of
wall nobody pointed the phone at. Treating the empty cells as 2D blobs walks
straight into it: on the test fixture the doorway's empty region merged with the
unobserved strip of wall above it, and the merged region was too tall to be a
door, so a perfectly visible doorway went undetected.

So detection works one column at a time, which matches how openings are actually
shaped:

* a **door** is a run of adjacent columns that are empty from the floor up,
  with wall on both sides of the run;
* a **window** is a run of adjacent columns each holding a gap that is closed
  both below and above, at a consistent height, with unbroken wall either side.

A column with no returns at all is unobserved, not empty: it takes part in
nothing and is reported in the wall's observation fraction instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._numeric import quiet_fp

log = logging.getLogger("cozmo.geometry.openings")

CELL_M = 0.05
WALL_BAND_M = 0.12          # perpendicular distance that counts as "on this wall"
MIN_POINTS_PER_CELL = 1
MIN_COLUMN_POINTS = 3       # fewer than this and the column is unobserved
# A column's first wall is the first *sustained* run of occupied cells, not the
# first occupied cell. Floor points bleeding into the wall band put isolated
# cells inside a doorway, which otherwise chop one 0.85 m door into three
# fragments too narrow to be a door.
SUSTAINED_WALL_CELLS = 3
# Columns this far apart still belong to the same opening: a stray return in the
# middle of a doorway must not split it.
RUN_GAP_TOLERANCE = 2
MIN_WALL_OBSERVATION = 0.30 # fraction of columns that must be observed at all
MIN_WALL_POINTS = 200

DOOR_WIDTH_RANGE = (0.55, 1.40)
DOOR_MIN_CLEARANCE_M = 1.40
WINDOW_WIDTH_RANGE = (0.35, 2.60)
WINDOW_MIN_HEIGHT_M = 0.40
WINDOW_MIN_SILL_M = 0.25
WINDOW_SILL_TOLERANCE_M = 0.20

# A run must be flanked by wall on both sides, within this many columns.
FLANK_SEARCH_COLUMNS = 6
FLANK_MAX_CLEARANCE_M = 0.35


@dataclass
class OpeningDetection:
    wall_index: int
    kind: str                       # "door" | "window"
    width_m: float
    height_m: float
    offset_along_wall_m: float      # from the wall's start point
    sill_height_m: float
    confidence: float
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WallGrid:
    occupied: np.ndarray            # (rows, cols) bool, row 0 at the floor
    observed: np.ndarray            # (cols,) bool
    length_m: float
    points: int

    @property
    def columns(self) -> int:
        return self.occupied.shape[1]

    @property
    def observation_fraction(self) -> float:
        return float(self.observed.mean()) if self.observed.size else 0.0


def build_wall_grid(
    uv: np.ndarray,
    heights: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    ceiling_m: float,
) -> WallGrid:
    """Occupancy grid for one wall: rows are height, columns run along the wall."""
    seg = end - start
    length = float(np.linalg.norm(seg))
    if length <= 0:
        raise ValueError("degenerate wall segment")
    direction = seg / length
    normal = np.array([-direction[1], direction[0]])

    rel = uv - start
    with quiet_fp():
        along = rel @ direction
        across = np.abs(rel @ normal)

    near = (
        (across <= WALL_BAND_M) & (along >= 0) & (along <= length)
        & (heights >= 0) & (heights <= ceiling_m)
    )
    columns = max(1, int(np.ceil(length / CELL_M)))
    rows = max(1, int(np.ceil(ceiling_m / CELL_M)))
    grid = np.zeros((rows, columns), dtype=np.int32)
    if np.any(near):
        cu = np.clip((along[near] / CELL_M).astype(int), 0, columns - 1)
        cv = np.clip((heights[near] / CELL_M).astype(int), 0, rows - 1)
        np.add.at(grid, (cv, cu), 1)

    occupied = grid >= MIN_POINTS_PER_CELL
    observed = occupied.sum(axis=0) >= MIN_COLUMN_POINTS
    return WallGrid(occupied=occupied, observed=observed, length_m=length, points=int(near.sum()))


def _runs(flags: np.ndarray, gap_tolerance: int = 0) -> List[Tuple[int, int]]:
    """Contiguous True runs as [start, end) index pairs.

    ``gap_tolerance`` bridges short False stretches, so a single spurious cell
    does not split one opening into several.
    """
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(flags)))

    if gap_tolerance <= 0 or len(runs) < 2:
        return runs

    merged = [runs[0]]
    for current in runs[1:]:
        previous = merged[-1]
        if current[0] - previous[1] <= gap_tolerance:
            merged[-1] = (previous[0], current[1])
        else:
            merged.append(current)
    return merged


def _first_sustained(column: np.ndarray) -> Optional[int]:
    """Row of the first run of ``SUSTAINED_WALL_CELLS`` occupied cells."""
    run_length = 0
    for row, occupied in enumerate(column):
        run_length = run_length + 1 if occupied else 0
        if run_length >= SUSTAINED_WALL_CELLS:
            return row - SUSTAINED_WALL_CELLS + 1
    return None


def _clearance_cells(grid: WallGrid) -> np.ndarray:
    """Per column, how many empty cells sit between the floor and the first wall."""
    rows = grid.occupied.shape[0]
    clearance = np.full(grid.columns, rows, dtype=int)
    for column in range(grid.columns):
        if not grid.observed[column]:
            continue
        first = _first_sustained(grid.occupied[:, column])
        clearance[column] = rows if first is None else int(first)
    return clearance


def _column_gaps(grid: WallGrid, column: int) -> List[Tuple[int, int]]:
    """Empty runs closed above and below by wall, in one column."""
    occupied_rows = np.flatnonzero(grid.occupied[:, column])
    if len(occupied_rows) < 2:
        return []
    interior = np.zeros(grid.occupied.shape[0], dtype=bool)
    interior[occupied_rows.min():occupied_rows.max() + 1] = True
    empty = interior & ~grid.occupied[:, column]
    return _runs(empty)


def _flanked(clearance: np.ndarray, observed: np.ndarray, start: int, end: int) -> bool:
    """Is this run bounded by wall that reaches the floor on both sides?"""
    limit = FLANK_MAX_CLEARANCE_M / CELL_M

    def solid(indices: Sequence[int]) -> bool:
        for index in indices:
            if 0 <= index < len(clearance) and observed[index]:
                return clearance[index] <= limit
        return False

    left = solid(range(start - 1, start - 1 - FLANK_SEARCH_COLUMNS, -1))
    right = solid(range(end, end + FLANK_SEARCH_COLUMNS))
    return left and right


def _detect_doors(grid: WallGrid, wall_index: int) -> List[OpeningDetection]:
    clearance = _clearance_cells(grid)
    threshold = DOOR_MIN_CLEARANCE_M / CELL_M
    candidate = grid.observed & (clearance >= threshold)

    detections: List[OpeningDetection] = []
    for start, end in _runs(candidate, gap_tolerance=RUN_GAP_TOLERANCE):
        width = (end - start) * CELL_M
        if not (DOOR_WIDTH_RANGE[0] <= width <= DOOR_WIDTH_RANGE[1]):
            continue
        if not _flanked(clearance, grid.observed, start, end):
            continue
        height = float(np.median(clearance[start:end])) * CELL_M
        # Confidence follows how consistent the run's own height is: a real
        # doorway has a flat head, missing data does not.
        spread = float(np.ptp(clearance[start:end])) * CELL_M
        detections.append(OpeningDetection(
            wall_index=wall_index, kind="door",
            width_m=width, height_m=height,
            offset_along_wall_m=start * CELL_M, sill_height_m=0.0,
            confidence=round(min(0.95, max(0.3, 1.0 - spread)), 3),
            stats={"head_spread_m": round(spread, 3), "columns": end - start},
        ))
    return detections


def _detect_windows(grid: WallGrid, wall_index: int) -> List[OpeningDetection]:
    sills: List[Optional[int]] = []
    heads: List[Optional[int]] = []
    for column in range(grid.columns):
        best: Optional[Tuple[int, int]] = None
        if grid.observed[column]:
            for low, high in _column_gaps(grid, column):
                if (high - low) * CELL_M < WINDOW_MIN_HEIGHT_M:
                    continue
                if low * CELL_M < WINDOW_MIN_SILL_M:
                    continue
                if best is None or (high - low) > (best[1] - best[0]):
                    best = (low, high)
        sills.append(best[0] if best else None)
        heads.append(best[1] if best else None)

    has_gap = np.array([s is not None for s in sills])
    clearance = _clearance_cells(grid)

    detections: List[OpeningDetection] = []
    for start, end in _runs(has_gap, gap_tolerance=RUN_GAP_TOLERANCE):
        run_sills = np.array([sills[c] for c in range(start, end) if sills[c] is not None], dtype=float)
        run_heads = np.array([heads[c] for c in range(start, end) if heads[c] is not None], dtype=float)
        if len(run_sills) < 2:
            continue
        # One opening, not several stacked: the sill has to hold steady across it.
        if float(np.ptp(run_sills)) * CELL_M > WINDOW_SILL_TOLERANCE_M:
            continue
        width = (end - start) * CELL_M
        if not (WINDOW_WIDTH_RANGE[0] <= width <= WINDOW_WIDTH_RANGE[1]):
            continue
        # Wall either side, and no gap there -- otherwise this is a run that
        # simply ran out of observation.
        if not _flanked(clearance, grid.observed, start, end):
            continue
        if (start > 0 and has_gap[start - 1]) or (end < len(has_gap) and has_gap[end]):
            continue

        sill = float(np.median(run_sills)) * CELL_M
        head = float(np.median(run_heads)) * CELL_M
        spread = float(np.ptp(run_sills)) * CELL_M
        detections.append(OpeningDetection(
            wall_index=wall_index, kind="window",
            width_m=width, height_m=head - sill,
            offset_along_wall_m=start * CELL_M, sill_height_m=sill,
            confidence=round(min(0.95, max(0.3, 1.0 - 2 * spread)), 3),
            stats={"sill_spread_m": round(spread, 3), "columns": end - start},
        ))
    return detections


def wall_observation_fractions(
    wall_uv: np.ndarray,
    wall_heights: np.ndarray,
    walls: Sequence[Any],
    ceiling_height_m: float,
) -> List[float]:
    """Fraction of each wall's columns that returned anything at all.

    Reported per wall in the plan because it bounds what any claim about that
    wall can mean: "no openings found" in a wall observed across 12% of its
    length is not evidence that the wall is solid.
    """
    fractions: List[float] = []
    for wall in walls:
        try:
            grid = build_wall_grid(
                wall_uv, wall_heights, np.array(wall.start, dtype=float),
                np.array(wall.end, dtype=float), ceiling_height_m,
            )
        except ValueError:
            fractions.append(0.0)
            continue
        fractions.append(grid.observation_fraction)
    return fractions


def detect_openings(
    wall_uv: np.ndarray,
    wall_heights: np.ndarray,
    walls: Sequence[Any],
    ceiling_height_m: float,
) -> List[OpeningDetection]:
    """Find doors and windows in every wall of a layout."""
    detections: List[OpeningDetection] = []

    for index, wall in enumerate(walls):
        try:
            grid = build_wall_grid(
                wall_uv, wall_heights, np.array(wall.start, dtype=float),
                np.array(wall.end, dtype=float), ceiling_height_m,
            )
        except ValueError:
            continue

        if grid.points < MIN_WALL_POINTS:
            continue
        if grid.observation_fraction < MIN_WALL_OBSERVATION:
            log.info(
                "wall %d observed in only %.0f%% of its columns; not claiming openings in it",
                index, 100 * grid.observation_fraction,
            )
            continue

        detections.extend(_detect_doors(grid, index))
        detections.extend(_detect_windows(grid, index))

    log.info("openings: %d detected across %d walls", len(detections), len(walls))
    return detections
