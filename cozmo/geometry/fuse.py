"""Frame fusion: depth frames in, one world-space point cloud out.

Confidence masking happens here and only here. ARKit's level-0 depth is where
mirrors, glass, wet-look floors and dark surfaces put their garbage, so the
default keeps level 2 only; the level actually used is recorded in the manifest
because it is the single biggest lever on what the rest of the pipeline sees.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import open3d as o3d

from ..io.lidar import LidarCapture
from ..io.stray import CONFIDENCE_HIGH

log = logging.getLogger("cozmo.geometry.fuse")

DEFAULT_STRIDE = 10
DEFAULT_VOXEL_M = 0.03
DEFAULT_MIN_DEPTH_M = 0.2
# ARKit LiDAR degrades badly past ~5 m; the loader defaults to this too.
DEFAULT_MAX_DEPTH_M = 5.0
# Never fuse fewer than this many frames when the capture has them: a stride
# tuned for a 1700-frame walkthrough would otherwise use two frames of a short
# capture and fail for reasons that look like algorithm bugs.
MIN_FRAMES_FUSED = 24


@dataclass
class FusedCloud:
    """A fused cloud plus the settings and statistics behind it."""

    points: np.ndarray                     # (N, 3) float32, world frame
    trajectory: np.ndarray                 # (F, 3) camera positions, world frame
    stride: int
    voxel_size_m: float
    min_confidence: int
    frames_used: int
    points_before_downsample: int
    stats: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.points)

    @property
    def camera_height_median(self) -> float:
        return float(np.median(self.trajectory[:, 1])) if len(self.trajectory) else 0.0

    def to_open3d(self) -> o3d.geometry.PointCloud:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(self.points.astype(np.float64))
        return cloud

    def summary(self) -> Dict[str, Any]:
        return {
            "points": len(self.points),
            "points_before_downsample": self.points_before_downsample,
            "frames_used": self.frames_used,
            "stride": self.stride,
            "voxel_size_m": self.voxel_size_m,
            "min_confidence": self.min_confidence,
            **self.stats,
        }


def voxel_downsample(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    """Open3D voxel grid downsample. A no-op when voxel_size_m <= 0."""
    if voxel_size_m <= 0 or len(points) == 0:
        return points
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return np.asarray(cloud.voxel_down_sample(voxel_size_m).points, dtype=np.float32)


def fuse_capture(
    lidar: LidarCapture,
    stride: int = DEFAULT_STRIDE,
    voxel_size_m: float = DEFAULT_VOXEL_M,
    min_confidence: int = CONFIDENCE_HIGH,
    min_depth_m: float = DEFAULT_MIN_DEPTH_M,
    max_depth_m: float = DEFAULT_MAX_DEPTH_M,
) -> FusedCloud:
    """Unproject every ``stride``-th frame into the world frame and fuse."""
    capture = lidar.capture
    if len(capture) == 0:
        raise ValueError("capture contains no depth frames")

    requested_stride = max(1, stride)
    stride = min(requested_stride, max(1, len(capture) // MIN_FRAMES_FUSED))
    if stride != requested_stride:
        log.info("stride %d would use too few frames; using %d", requested_stride, stride)

    chunks = []
    frames_used = 0
    dropped_frames = 0

    # numpy on Accelerate raises spurious divide/overflow flags on float32
    # matmul; every point comes back finite, so the flags are noise, not data.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(all="ignore"):
            for index in range(0, len(capture), stride):
                frame = capture.frame(index)
                points = capture.unproject(
                    frame,
                    min_confidence=min_confidence,
                    min_depth_m=min_depth_m,
                    max_depth_m=max_depth_m,
                )
                if len(points) == 0:
                    dropped_frames += 1
                    continue
                finite = np.isfinite(points).all(axis=1)
                if not finite.all():
                    points = points[finite]
                if len(points):
                    chunks.append(points)
                    frames_used += 1
                else:
                    dropped_frames += 1

    if not chunks:
        raise ValueError(
            f"no points survived fusion (confidence >= {min_confidence}, "
            f"depth in [{min_depth_m}, {max_depth_m}] m)"
        )

    raw = np.concatenate(chunks)
    points = voxel_downsample(raw, voxel_size_m)
    trajectory = lidar.trajectory()

    heights = points[:, 1]
    camera_height = float(np.median(trajectory[:, 1])) if len(trajectory) else 0.0
    stats = {
        "dropped_frames": dropped_frames,
        "requested_stride": requested_stride,
        "extent_m": [round(float(np.ptp(points[:, i])), 3) for i in range(3)],
        "height_range_m": [round(float(heights.min()), 3), round(float(heights.max()), 3)],
        "camera_height_median_m": round(camera_height, 3),
        "fraction_above_camera": round(float((heights > camera_height).mean()), 4),
    }
    log.info(
        "fused %d frames -> %d points (%d before downsample)",
        frames_used, len(points), len(raw),
    )

    return FusedCloud(
        points=points,
        trajectory=trajectory,
        stride=stride,
        voxel_size_m=voxel_size_m,
        min_confidence=min_confidence,
        frames_used=frames_used,
        points_before_downsample=int(len(raw)),
        stats=stats,
    )
