"""Plan rendering: plan.png and plan.svg.

Deliberately plain. The drawing exists so a person can see whether the geometry
is right, so it shows the things that would reveal it being wrong: wall lengths
with their intervals, openings as gaps in the wall rather than symbols on top of
it, a scale bar, and the capture's own axis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

log = logging.getLogger("cozmo.geometry.render")

WALL_LINEWIDTH = 3.0
OPENING_LINEWIDTH = 1.2
DOOR_COLOR = "#c0392b"
WINDOW_COLOR = "#2980b9"
WALL_COLOR = "#222222"
FLOOR_COLOR = "#f2f2f0"


def _nice_scale_length(span: float) -> float:
    for candidate in (5.0, 2.0, 1.0, 0.5):
        if span > candidate * 2.5:
            return candidate
    return 0.5


def render_plan(
    walls: Sequence[Any],
    openings: Sequence[Any],
    out_dir: Path,
    title: str = "",
    subtitle: str = "",
    trajectory_uv: Optional[np.ndarray] = None,
    rotation_deg: float = 0.0,
    basename: str = "plan",
) -> List[Path]:
    """Draw the plan. ``walls`` carry (start, end, length_m); openings index them."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_aspect("equal")

    ring = [np.array(w.start) for w in walls]
    if ring:
        polygon = np.array(ring + [ring[0]])
        ax.fill(polygon[:, 0], polygon[:, 1], color=FLOOR_COLOR, zorder=0)

    by_wall: Dict[int, List[Any]] = {}
    for opening in openings:
        by_wall.setdefault(opening.wall_index, []).append(opening)

    for index, wall in enumerate(walls):
        start, end = np.array(wall.start, dtype=float), np.array(wall.end, dtype=float)
        seg = end - start
        length = float(np.linalg.norm(seg))
        if length <= 0:
            continue
        direction = seg / length

        # Walk the wall, drawing solid runs and leaving gaps where openings are.
        spans = sorted(
            (max(0.0, o.offset_along_wall_m), min(length, o.offset_along_wall_m + o.width_m), o)
            for o in by_wall.get(index, [])
        )
        cursor = 0.0
        for span_start, span_end, opening in spans:
            if span_start > cursor:
                a, b = start + direction * cursor, start + direction * span_start
                ax.plot([a[0], b[0]], [a[1], b[1]], color=WALL_COLOR, lw=WALL_LINEWIDTH,
                        solid_capstyle="butt", zorder=3)
            a, b = start + direction * span_start, start + direction * span_end
            colour = DOOR_COLOR if opening.kind == "door" else WINDOW_COLOR
            ax.plot([a[0], b[0]], [a[1], b[1]], color=colour, lw=OPENING_LINEWIDTH,
                    linestyle=(0, (2, 2)), solid_capstyle="butt", zorder=4)
            cursor = max(cursor, span_end)
        if cursor < length:
            a, b = start + direction * cursor, end
            ax.plot([a[0], b[0]], [a[1], b[1]], color=WALL_COLOR, lw=WALL_LINEWIDTH,
                    solid_capstyle="butt", zorder=3)

        midpoint = (start + end) / 2
        normal = np.array([-direction[1], direction[0]])
        label = getattr(wall, "label", None) or f"{length:.2f} m"
        ax.annotate(
            label, xy=midpoint + normal * 0.18, ha="center", va="center", fontsize=8,
            rotation=np.degrees(np.arctan2(direction[1], direction[0])) % 180 - 0,
            rotation_mode="anchor", color="#333333", zorder=5,
        )

    if trajectory_uv is not None and len(trajectory_uv):
        ax.plot(trajectory_uv[:, 0], trajectory_uv[:, 1], color="#7f8c8d", lw=0.7, alpha=0.7,
                zorder=2, label="capture path")

    xs = [p[0] for w in walls for p in (w.start, w.end)]
    ys = [p[1] for w in walls for p in (w.start, w.end)]
    if not xs:
        xs, ys = [0.0, 1.0], [0.0, 1.0]
    pad = 0.6
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    # Scale bar, bottom left.
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    bar = _nice_scale_length(span)
    bx, by = min(xs) - pad * 0.5, min(ys) - pad * 0.6
    ax.plot([bx, bx + bar], [by, by], color="#000000", lw=2.5, solid_capstyle="butt", zorder=6)
    ax.annotate(f"{bar:g} m", xy=(bx + bar / 2, by + 0.08), ha="center", fontsize=8, zorder=6)

    # Capture-frame axis. ARKit has no compass, so this is not magnetic north
    # and does not pretend to be.
    ax_x, ax_y = max(xs) + pad * 0.35, max(ys) - 0.4
    angle = np.radians(-rotation_deg + 90.0)
    ax.annotate(
        "", xy=(ax_x + 0.35 * np.cos(angle), ax_y + 0.35 * np.sin(angle)), xytext=(ax_x, ax_y),
        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.2), zorder=6,
    )
    ax.annotate("capture +Z\n(not magnetic N)", xy=(ax_x, ax_y - 0.28), ha="center",
                fontsize=6.5, color="#555555", zorder=6)

    handles = [
        Line2D([], [], color=WALL_COLOR, lw=WALL_LINEWIDTH, label="wall"),
        Line2D([], [], color=DOOR_COLOR, lw=OPENING_LINEWIDTH, linestyle=(0, (2, 2)), label="door"),
        Line2D([], [], color=WINDOW_COLOR, lw=OPENING_LINEWIDTH, linestyle=(0, (2, 2)), label="window"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7, frameon=False)

    ax.set_title(title or "Floor plan", fontsize=12, loc="left")
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=7.5, color="#555555")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()

    written: List[Path] = []
    for suffix in ("png", "svg"):
        path = out_dir / f"{basename}.{suffix}"
        fig.savefig(path, dpi=150 if suffix == "png" else None, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    log.info("rendered %s", ", ".join(p.name for p in written))
    return written
