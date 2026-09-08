"""Stray Scanner capture loader (LiDAR tier).

PLACEHOLDER -- this module is expected to be replaced by the existing loader.
Keep the public surface below (:class:`StrayCapture`, :func:`load_stray_capture`,
:func:`summarize`) and the rest of the pipeline keeps working unchanged.

Stray Scanner writes one directory per capture::

    <capture>/rgb.mp4              H.264 colour, one frame per pose row
    <capture>/camera_matrix.csv    3x3 intrinsics for the colour stream
    <capture>/odometry.csv         timestamp, frame, x, y, z, qx, qy, qz, qw
    <capture>/imu.csv              timestamp, ax, ay, az, gx, gy, gz
    <capture>/depth/000000.png     uint16 depth in millimetres (LiDAR)
    <capture>/confidence/000000.png  0/1/2 ARKit depth confidence

This implementation parses the metadata -- intrinsics, poses, frame inventory --
which is all the skeleton needs. Pixels are located, never decoded: no vision
code yet, and decoding here would pull opencv into a stage that has nothing to
do with it.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

DEPTH_SCALE_MM_TO_M = 1e-3

# ARKit confidence levels; the pipeline is expected to drop level 0 depth, which
# is where mirrors, glass and wet-look floors mostly land.
CONFIDENCE_LOW, CONFIDENCE_MEDIUM, CONFIDENCE_HIGH = 0, 1, 2


@dataclass
class Pose:
    """One ARKit odometry row. Translation in metres, rotation as a quaternion."""

    timestamp: float
    frame: int
    translation: Tuple[float, float, float]
    quaternion: Tuple[float, float, float, float]  # (qx, qy, qz, qw)


@dataclass
class StrayCapture:
    root: Path
    intrinsics: Optional[np.ndarray]  # 3x3, None when camera_matrix.csv is absent
    poses: List[Pose]
    depth_frames: List[Path]
    confidence_frames: List[Path]
    rgb_video: Optional[Path]
    imu_csv: Optional[Path]
    warnings: List[str] = field(default_factory=list)

    @property
    def frame_count(self) -> int:
        return len(self.poses)

    @property
    def has_depth(self) -> bool:
        return bool(self.depth_frames)

    def trajectory(self) -> np.ndarray:
        """(N, 3) translation track -- the input to any drift analysis."""
        if not self.poses:
            return np.zeros((0, 3), dtype=float)
        return np.array([p.translation for p in self.poses], dtype=float)

    def path_length_m(self) -> float:
        xyz = self.trajectory()
        if len(xyz) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())

    def loop_closure_gap_m(self) -> Optional[float]:
        """Distance between first and last pose. On a walkthrough that returns
        to its start this is raw, uncorrected drift -- the number the drift
        ablation has to move."""
        xyz = self.trajectory()
        if len(xyz) < 2:
            return None
        return float(np.linalg.norm(xyz[-1] - xyz[0]))


def _read_intrinsics(path: Path, warnings: List[str]) -> Optional[np.ndarray]:
    if not path.is_file():
        warnings.append("camera_matrix.csv missing; intrinsics unknown")
        return None
    rows = [
        [float(v) for v in row]
        for row in csv.reader(path.open())
        if row and not row[0].lstrip().startswith("#")
    ]
    matrix = np.array(rows, dtype=float)
    if matrix.shape != (3, 3):
        warnings.append(f"camera_matrix.csv is {matrix.shape}, expected (3, 3)")
        return None
    return matrix


def _read_odometry(path: Path, warnings: List[str]) -> List[Pose]:
    if not path.is_file():
        warnings.append("odometry.csv missing; capture has no poses")
        return []

    poses: List[Pose] = []
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        for index, row in enumerate(reader):
            if not row:
                continue
            try:
                values = [float(v) for v in row[:9]]
            except ValueError:
                # First row is Stray's header line.
                if index == 0:
                    continue
                warnings.append(f"odometry.csv row {index} unparseable; skipped")
                continue
            if len(values) < 9:
                warnings.append(f"odometry.csv row {index} has {len(values)} columns, expected 9")
                continue
            ts, frame, x, y, z, qx, qy, qz, qw = values
            poses.append(
                Pose(timestamp=ts, frame=int(frame), translation=(x, y, z), quaternion=(qx, qy, qz, qw))
            )
    return poses


def _frames_in(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.iterdir() if p.suffix.lower() == ".png"), key=lambda p: p.name)


def load_stray_capture(root: Path) -> StrayCapture:
    """Read a Stray Scanner capture directory's metadata."""
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"stray capture root is not a directory: {root}")

    warnings: List[str] = []
    intrinsics = _read_intrinsics(root / "camera_matrix.csv", warnings)
    poses = _read_odometry(root / "odometry.csv", warnings)
    depth_frames = _frames_in(root / "depth")
    confidence_frames = _frames_in(root / "confidence")

    if not depth_frames:
        warnings.append("no depth/*.png frames; this capture cannot serve the LiDAR tier")
    elif poses and len(depth_frames) != len(poses):
        warnings.append(
            f"{len(depth_frames)} depth frames vs {len(poses)} poses; frames will be matched by index"
        )
    if depth_frames and not confidence_frames:
        warnings.append("no confidence/*.png; low-confidence depth cannot be rejected")

    rgb = root / "rgb.mp4"
    imu = root / "imu.csv"
    return StrayCapture(
        root=root,
        intrinsics=intrinsics,
        poses=poses,
        depth_frames=depth_frames,
        confidence_frames=confidence_frames,
        rgb_video=rgb if rgb.is_file() else None,
        imu_csv=imu if imu.is_file() else None,
        warnings=warnings,
    )


def summarize(capture: StrayCapture) -> Dict[str, object]:
    """Compact description for the run manifest."""
    return {
        "frame_count": capture.frame_count,
        "depth_frames": len(capture.depth_frames),
        "confidence_frames": len(capture.confidence_frames),
        "has_intrinsics": capture.intrinsics is not None,
        "has_rgb_video": capture.rgb_video is not None,
        "path_length_m": round(capture.path_length_m(), 4),
        "loop_closure_gap_m": (
            round(capture.loop_closure_gap_m(), 4) if capture.loop_closure_gap_m() is not None else None
        ),
        "warnings": capture.warnings,
    }
