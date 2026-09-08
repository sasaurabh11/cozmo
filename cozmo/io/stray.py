"""
Stray Scanner capture loader.

Verified against a real capture: conventions below were determined empirically by
testing which produces a flat floor plane, not assumed from documentation.

Format:
    rgb.mp4            HEVC video, full resolution (e.g. 1920x1440)
    depth/NNNNNN.png   uint16, millimetres, low resolution (256x192)
    confidence/NNNNNN.png  uint8, 0=low 1=medium 2=high (ARKit confidence)
    odometry.csv       per-frame pose: x,y,z + quaternion qx,qy,qz,qw
    camera_matrix.csv  3x3 intrinsics for the RGB stream

Conventions (verified):
    - Camera frame is standard pinhole: x right, y down, z forward.
    - odometry quaternion is camera-to-world: P_world = R @ P_cam + t
    - World y is up. Origin is the phone's starting pose.
    - Depth intrinsics = RGB intrinsics scaled by (depth_width / rgb_width).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

DEPTH_SCALE_MM_TO_M = 1000.0
CONFIDENCE_HIGH = 2
CONFIDENCE_MEDIUM = 1


@dataclass
class Frame:
    """One synchronised RGB + depth + pose sample."""

    index: int
    timestamp: float
    depth_m: np.ndarray  # (H, W) float32, metres, 0 where invalid
    confidence: np.ndarray  # (H, W) uint8, 0/1/2
    R: np.ndarray  # (3, 3) camera-to-world rotation
    t: np.ndarray  # (3,) camera position in world
    rgb: np.ndarray | None = None  # (Hf, Wf, 3) BGR, only if requested


def quaternion_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a unit quaternion to a 3x3 rotation matrix."""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n == 0:
        raise ValueError("zero-norm quaternion")
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


class StrayCapture:
    """Reads a Stray Scanner capture directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not (self.root / "odometry.csv").exists():
            candidates = [d for d in self.root.iterdir() if (d / "odometry.csv").exists()]
            if len(candidates) == 1:
                self.root = candidates[0]
            else:
                raise FileNotFoundError(f"no odometry.csv under {self.root}")

        self.odometry = pd.read_csv(self.root / "odometry.csv", skipinitialspace=True)
        self.odometry.columns = [c.strip() for c in self.odometry.columns]

        self.rgb_K = np.loadtxt(self.root / "camera_matrix.csv", delimiter=",")

        self.depth_paths = sorted((self.root / "depth").glob("*.png"))
        self.confidence_paths = sorted((self.root / "confidence").glob("*.png"))
        if len(self.depth_paths) != len(self.confidence_paths):
            raise ValueError("depth and confidence frame counts differ")

        probe = cv2.imread(str(self.depth_paths[0]), cv2.IMREAD_UNCHANGED)
        self.depth_size = (probe.shape[1], probe.shape[0])  # (W, H)

        cap = cv2.VideoCapture(str(self.root / "rgb.mp4"))
        self.rgb_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        cap.release()

        # The scale trap: depth is much lower resolution than RGB.
        s = self.depth_size[0] / self.rgb_size[0]
        self.depth_K = self.rgb_K.copy()
        self.depth_K[:2, :] *= s

    def __len__(self) -> int:
        return len(self.depth_paths)

    def frame(self, i: int, with_rgb: bool = False) -> Frame:
        depth = cv2.imread(str(self.depth_paths[i]), cv2.IMREAD_UNCHANGED)
        depth_m = depth.astype(np.float32) / DEPTH_SCALE_MM_TO_M
        conf = cv2.imread(str(self.confidence_paths[i]), cv2.IMREAD_UNCHANGED)

        row = self.odometry.iloc[i]
        R = quaternion_to_rotation(row.qx, row.qy, row.qz, row.qw)
        t = np.array([row.x, row.y, row.z], dtype=np.float64)

        rgb = self._read_rgb(i) if with_rgb else None
        return Frame(i, float(row.timestamp), depth_m, conf, R, t, rgb)

    def _read_rgb(self, i: int) -> np.ndarray:
        cap = cv2.VideoCapture(str(self.root / "rgb.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise IOError(f"could not read RGB frame {i}")
        return img

    def unproject(
        self,
        frame: Frame,
        min_confidence: int = CONFIDENCE_HIGH,
        min_depth_m: float = 0.2,
        max_depth_m: float = 5.0,
    ) -> np.ndarray:
        """Turn one frame into world-space points. Returns (N, 3) float32."""
        fx, fy = self.depth_K[0, 0], self.depth_K[1, 1]
        cx, cy = self.depth_K[0, 2], self.depth_K[1, 2]

        h, w = frame.depth_m.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        mask = (
            (frame.confidence >= min_confidence)
            & (frame.depth_m > min_depth_m)
            & (frame.depth_m < max_depth_m)
        )
        if not mask.any():
            return np.empty((0, 3), dtype=np.float32)

        z = frame.depth_m[mask]
        x = (u[mask] - cx) / fx * z
        y = (v[mask] - cy) / fy * z
        pts_cam = np.stack([x, y, z], axis=1)

        return (pts_cam @ frame.R.T + frame.t).astype(np.float32)

    def point_cloud(self, stride: int = 10, **kwargs) -> np.ndarray:
        """Fuse every `stride`-th frame into one world-space cloud."""
        chunks = []
        for i in range(0, len(self), stride):
            pts = self.unproject(self.frame(i), **kwargs)
            if len(pts):
                chunks.append(pts)
        if not chunks:
            return np.empty((0, 3), dtype=np.float32)
        return np.concatenate(chunks)


def dominant_horizontal_planes(
    points: np.ndarray, bin_m: float = 0.02, min_fraction: float = 0.05
) -> list[tuple[float, int]]:
    """Find candidate floor/ceiling heights by histogramming the vertical axis.

    Returns [(height_m, point_count), ...] sorted by height, densest bins only.
    An empty or single-entry result usually means a surface was never captured.
    """
    y = points[:, 1]
    edges = np.arange(y.min(), y.max() + bin_m, bin_m)
    counts, _ = np.histogram(y, bins=edges)
    threshold = counts.max() * min_fraction

    planes = [
        ((edges[i] + edges[i + 1]) / 2, int(counts[i]))
        for i in range(len(counts))
        if counts[i] >= threshold
    ]
    return planes


def sanity_check(capture: StrayCapture) -> dict:
    """Run the checks that catch convention and scaling errors early."""
    cloud = capture.point_cloud(stride=20)
    planes = dominant_horizontal_planes(cloud, min_fraction=0.25)

    xyz = capture.odometry[["x", "y", "z"]].to_numpy()
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)

    above_camera = float((cloud[:, 1] > 0).mean()) if len(cloud) else 0.0

    return {
        "frames": len(capture),
        "depth_resolution": capture.depth_size,
        "rgb_resolution": capture.rgb_size,
        "points_sampled": len(cloud),
        "horizontal_planes_m": [round(h, 3) for h, _ in planes],
        "path_length_m": round(float(steps.sum()), 2),
        "loop_closure_gap_m": round(float(np.linalg.norm(xyz[-1] - xyz[0])), 3),
        "max_step_m": round(float(steps.max()), 4),
        "fraction_above_camera_height": round(above_camera, 4),
        "ceiling_likely_captured": above_camera > 0.15,
    }


if __name__ == "__main__":
    import json
    import sys

    cap = StrayCapture(sys.argv[1])
    print(json.dumps(sanity_check(cap), indent=2))
