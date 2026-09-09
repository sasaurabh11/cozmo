"""Adapter between the verified Stray Scanner loader and the pipeline.

:mod:`cozmo.io.stray` is used as-is; nothing here modifies it. This module gives
the rest of the pipeline the interface it already expects (warnings, a summary
for the run manifest) and is where ingest-time checks live -- most importantly
the ceiling check, which decides whether the ceiling height in the output plan
was measured or inferred.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .stray import CONFIDENCE_HIGH, StrayCapture, quaternion_to_rotation, sanity_check

log = logging.getLogger("cozmo.io.lidar")

# Below this fraction of points above camera height, assume the ceiling was
# never captured. The threshold is the loader's own (sanity_check uses 0.15);
# it is restated here because the pipeline acts on it.
CEILING_FRACTION_THRESHOLD = 0.15

# A capture with fewer poses than this cannot support a layout.
MIN_FRAMES = 4


@dataclass
class LidarCapture:
    """A Stray capture plus what ingest concluded about it."""

    root: Path
    capture: StrayCapture
    sanity: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)

    @property
    def frame_count(self) -> int:
        return len(self.capture)

    @property
    def has_depth(self) -> bool:
        return self.frame_count > 0

    @property
    def ceiling_likely_captured(self) -> bool:
        return bool(self.sanity.get("ceiling_likely_captured", False))

    @property
    def camera_heights(self) -> np.ndarray:
        return self.capture.odometry["y"].to_numpy()

    def trajectory(self) -> np.ndarray:
        return self.capture.odometry[["x", "y", "z"]].to_numpy()


def load_lidar_capture(root: Path) -> LidarCapture:
    """Open a Stray capture and run the ingest checks."""
    root = Path(root)
    capture = StrayCapture(root)

    warnings_out: List[str] = []
    if len(capture) < MIN_FRAMES:
        warnings_out.append(f"only {len(capture)} frames; too few to reconstruct a layout")

    # The loader's own sanity pass fuses a sparse cloud; the spurious FP flags
    # Accelerate raises on float32 matmul are not data problems (every point
    # comes back finite), so they are silenced rather than surfaced as warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(all="ignore"):
            sanity = sanity_check(capture)

    fraction = float(sanity.get("fraction_above_camera_height", 0.0))
    if not sanity.get("ceiling_likely_captured", False):
        message = (
            f"CEILING LIKELY NOT CAPTURED: only {fraction:.1%} of points sit above camera "
            f"height (threshold {CEILING_FRACTION_THRESHOLD:.0%}). Ceiling height will be "
            f"inferred from wall extent, not measured, and its interval will be wide."
        )
        log.warning(message)
        warnings_out.append(message)

    gap = sanity.get("loop_closure_gap_m")
    if gap is not None and gap > 1.0:
        warnings_out.append(
            f"trajectory ends {gap:.2f} m from where it started; this walk is not a closed "
            f"loop, so loop closure cannot be used to bound drift"
        )

    max_step = sanity.get("max_step_m")
    if max_step is not None and max_step > 0.15:
        warnings_out.append(
            f"largest inter-frame motion is {max_step:.3f} m; tracking may have jumped"
        )

    return LidarCapture(root=root, capture=capture, sanity=sanity, warnings=warnings_out)


def iter_rgb_frames(
    lidar: LidarCapture,
    stride: int = 100,
    max_frames: Optional[int] = None,
    long_edge: Optional[int] = None,
):
    """Yield ``(index, rgb, R, t, K)`` for sampled frames.

    The video is walked sequentially with ``grab()`` and decoded only on the
    frames wanted: seeking per frame re-opens and re-seeks the stream, which on
    a 1715-frame HEVC file costs more than decoding the whole thing.

    ``K`` comes back scaled to the returned image, so a caller that downscales
    for detection still projects correctly -- the single most likely place to
    introduce a silent metric error.
    """
    import cv2

    capture = lidar.capture
    video_path = capture.root / "rgb.mp4"
    if not video_path.is_file():
        log.warning("no rgb.mp4 in %s; semantic detection cannot run", capture.root)
        return

    video = cv2.VideoCapture(str(video_path))
    try:
        wanted = list(range(0, len(capture), max(1, stride)))
        if max_frames is not None:
            wanted = wanted[:max_frames]
        wanted_set = set(wanted)
        emitted = 0

        for index in range(len(capture)):
            if index not in wanted_set:
                video.grab()
                continue
            ok, bgr = video.read()
            if not ok:
                break

            scale = 1.0
            if long_edge:
                longest = max(bgr.shape[0], bgr.shape[1])
                if longest > long_edge:
                    scale = long_edge / longest
                    bgr = cv2.resize(
                        bgr, (int(bgr.shape[1] * scale), int(bgr.shape[0] * scale))
                    )

            row = capture.odometry.iloc[index]
            R = quaternion_to_rotation(row.qx, row.qy, row.qz, row.qw)
            t = np.array([row.x, row.y, row.z], dtype=float)

            K = capture.rgb_K.copy()
            K[:2, :] *= scale

            yield index, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), R, t, K
            emitted += 1
        log.info("iterated %d RGB frame(s) at stride %d", emitted, stride)
    finally:
        video.release()


def summarize(lidar: LidarCapture) -> Dict[str, Any]:
    """Compact description for the run manifest."""
    sanity = lidar.sanity
    return {
        "frame_count": lidar.frame_count,
        "depth_frames": len(lidar.capture.depth_paths),
        "confidence_frames": len(lidar.capture.confidence_paths),
        "depth_resolution": list(lidar.capture.depth_size),
        "rgb_resolution": list(lidar.capture.rgb_size),
        "has_intrinsics": True,
        "depth_intrinsics_fx": round(float(lidar.capture.depth_K[0, 0]), 4),
        "min_confidence_used": CONFIDENCE_HIGH,
        "path_length_m": sanity.get("path_length_m"),
        "loop_closure_gap_m": sanity.get("loop_closure_gap_m"),
        "fraction_above_camera_height": sanity.get("fraction_above_camera_height"),
        "ceiling_likely_captured": sanity.get("ceiling_likely_captured"),
        "horizontal_planes_m": sanity.get("horizontal_planes_m"),
        "warnings": lidar.warnings,
    }
