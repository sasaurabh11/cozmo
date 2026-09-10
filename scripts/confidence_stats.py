#!/usr/bin/env python3
"""Depth-confidence statistics per LiDAR capture -- the evidence behind
known_failure_modes.md.

    python scripts/confidence_stats.py [capture-dir ...]

The Stray Scanner writes a per-pixel depth confidence map beside every depth
frame (0 = no usable return, 1 = medium, 2 = high). It is the sensor's own
admission of where it failed, which makes it the right instrument for asking
what mirrors, glass and glossy surfaces actually did to a capture -- no extra
tooling, no labelling.

Prints, per capture: the mean/p90/max fraction of zero-confidence pixels, the
mean fraction of high-confidence pixels, and the worst frames by name so they
can be opened alongside their RGB.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

DEFAULT_CAPTURES = ("apartment_lidar", "scan_with_ceiling", "scan_floor_only")
MAX_FRAMES = 120          # even sampling; reading every frame changes nothing


def stats(capture: Path) -> None:
    files = sorted((capture / "confidence").glob("*.png"))
    if not files:
        print(f"{capture.name}: no confidence/ frames")
        return

    rows = []
    for path in files[:: max(1, len(files) // MAX_FRAMES)]:
        conf = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if conf is None:
            continue
        rows.append((path.stem, float((conf == 0).mean()), float((conf == 2).mean())))

    zero = np.array([r[1] for r in rows])
    high = np.array([r[2] for r in rows])
    worst = sorted(rows, key=lambda r: -r[1])[:5]

    print(
        f"{capture.name}: zero-conf mean {zero.mean()*100:.1f}%  "
        f"p90 {np.percentile(zero, 90)*100:.1f}%  max {zero.max()*100:.1f}%  "
        f"| high-conf mean {high.mean()*100:.1f}%  ({len(rows)} frames sampled)"
    )
    print("   worst:", ", ".join(f"frame {n} @ {z*100:.0f}%" for n, z, _ in worst))


def main() -> None:
    args = sys.argv[1:]
    captures = [Path(a) for a in args] or [Path("captures") / c for c in DEFAULT_CAPTURES]
    for capture in captures:
        if capture.is_dir():
            stats(capture)
        else:
            print(f"{capture}: not found (real captures are gitignored)")


if __name__ == "__main__":
    main()
