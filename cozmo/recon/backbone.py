"""The reconstruction backbone: a Protocol, and one implementation.

``Reconstructor`` is deliberately narrow -- one method, one result shape -- so a
second backbone is a config change, not a rewrite. A backbone swap is exactly
the kind of thing the fix loop might ship: if VGGT turns out to be the
worst-performing gate, trying DUSt3R or MASt3R here means adding a class and a
registry entry, not touching scale.py, layout.py or the pipeline.

Every backbone returns points **in its own scale-free units**: recovering
metres is scale.py's job, deliberately kept separate, because conflating "what
the network saw" with "how big it thinks the room is" is exactly the kind of
silent coupling that makes a backbone swap expensive later.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from .frames import Frame

log = logging.getLogger("cozmo.recon.backbone")

DEFAULT_WEIGHTS_DIR = Path(os.environ.get("COZMO_WEIGHTS_DIR", "weights"))


class ReconstructorUnavailable(RuntimeError):
    """Weights are missing or the backbone failed to load."""


@dataclass
class FramePose:
    """Camera-to-world pose for one frame, in the backbone's own scale-free units."""

    frame_index: int
    R: np.ndarray             # (3, 3) camera-to-world rotation
    t: np.ndarray             # (3,) camera position, world frame
    K: np.ndarray             # (3, 3) intrinsics used/refined by the backbone


@dataclass
class ReconstructionResult:
    """What every backbone produces, regardless of how it got there."""

    points: np.ndarray                        # (N, 3) float32, scale-free world frame
    point_confidence: np.ndarray               # (N,) float32 in [0, 1]
    point_frame_index: np.ndarray              # (N,) int, which frame each point came from
    poses: List[FramePose]
    scale_is_metric: bool = False
    # Per-frame local point clouds, in CAMERA-LOCAL coordinates (not world), when
    # the backbone can supply per-pixel depth. This is what layout.py's
    # per-view plane fitting needs -- see the Plane-DUSt3R ordering in layout.py
    # -- and is optional because not every backbone provides dense per-view depth.
    frame_points_local: Dict[int, np.ndarray] = field(default_factory=dict)
    frame_confidence_local: Dict[int, np.ndarray] = field(default_factory=dict)
    backbone_name: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.points)

    def points_for_frame(self, frame_index: int, min_confidence: float = 0.0) -> np.ndarray:
        mask = (self.point_frame_index == frame_index) & (self.point_confidence >= min_confidence)
        return self.points[mask]


@runtime_checkable
class Reconstructor(Protocol):
    """What a backbone must provide. Swappable by config -- see get_reconstructor."""

    name: str

    def reconstruct(self, frames: Sequence[Frame]) -> ReconstructionResult: ...


# --------------------------------------------------------------------------
# VGGT
# --------------------------------------------------------------------------


RECON_VENV_PYTHON = Path(os.environ.get("COZMO_RECON_PYTHON", ".venv-recon/bin/python"))
RECON_TIMEOUT_S = 1800


class VGGTReconstructor:
    """Visual Geometry Grounded Transformer (facebookresearch/vggt).

    Single feed-forward pass over all frames at once: no per-pair matching, no
    bundle adjustment. Outputs, per frame: a depth map, a camera pose, and a
    point confidence map, all already aligned to a single world frame (the
    first frame's camera, by VGGT's own convention) -- which is exactly the
    "first image's frame as world frame" convention layout.py assumes.

    **Runs in its own interpreter.** The upstream ``vggt`` package pins
    ``numpy<2`` and needs Python >= 3.10; this repo is numpy 2.x / Python 3.9
    everywhere else (open3d and several geometry fixes need the numpy-2 API).
    Rather than force a numpy downgrade that would silently change behaviour in
    every other module, ``reconstruct()`` here dispatches to
    :mod:`cozmo.recon.worker` running under ``.venv-recon`` and reads the
    result back. The ``Reconstructor`` interface -- one call, one result -- does
    not change; the subprocess is an implementation detail of this one backbone.

    Weights load strictly from local disk (no hub access at inference time): a
    benchmark run must not depend on what a remote file server happened to
    serve that day.
    """

    name = "vggt"

    def __init__(
        self,
        weights_dir: Optional[Path] = None,
        device: Optional[str] = None,
        image_size: int = 518,
        _direct: bool = False,
    ) -> None:
        """``_direct=True`` is set only by worker.py, which *is* the subprocess
        and must actually load the model rather than dispatch to another one."""
        self.weights_dir = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        self.image_size = image_size
        self.device = device
        self._direct = _direct

        if not _direct:
            python = RECON_VENV_PYTHON
            if not python.is_file():
                raise ReconstructorUnavailable(
                    f"{python} not found. Run scripts/setup_recon_env.sh to build the "
                    f"VGGT virtualenv (numpy<2, Python >= 3.10, kept separate from the "
                    f"rest of the project's numpy 2.x stack)."
                )
            weights_file = self.weights_dir / "vggt-1b" / "model.safetensors"
            if not weights_file.is_file():
                raise ReconstructorUnavailable(
                    f"VGGT weights not found in {self.weights_dir / 'vggt-1b'}. "
                    f"Run scripts/fetch_weights.sh first."
                )
            log.info("VGGT will run under %s (weights: %s)", python, weights_file)
            return

        # _direct path: actually load the model. Only reached inside worker.py,
        # under .venv-recon, where torch/vggt/numpy<2 are the ones installed.
        try:
            import torch
        except ImportError as exc:
            raise ReconstructorUnavailable(f"torch not installed: {exc}") from exc
        self._torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available() else "cpu")

        weights_file = self.weights_dir / "vggt-1b" / "model.safetensors"
        if not weights_file.is_file():
            raise ReconstructorUnavailable(
                f"VGGT weights not found in {self.weights_dir / 'vggt-1b'}. "
                f"Run scripts/fetch_weights.sh first."
            )

        try:
            from vggt.models.vggt import VGGT
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ReconstructorUnavailable(f"the vggt package is not importable: {exc}") from exc

        self.model = VGGT()
        state_dict = load_file(str(weights_file))
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            log.warning("VGGT checkpoint missing %d parameter tensor(s)", len(missing))
        self.model = self.model.to(self.device).eval()
        log.info("VGGT loaded on %s from %s", self.device, weights_file)

    def reconstruct(self, frames: Sequence[Frame]) -> ReconstructionResult:
        if len(frames) < 2:
            raise ValueError("VGGT needs at least 2 frames to triangulate a scale-free scene")
        if not self._direct:
            return self._reconstruct_subprocess(frames)
        return self._reconstruct_direct(frames)

    def _reconstruct_subprocess(self, frames: Sequence[Frame]) -> ReconstructionResult:
        import json
        import subprocess
        import sys as _sys
        import tempfile

        with tempfile.TemporaryDirectory(prefix="cozmo-vggt-") as scratch:
            request = {
                "frame_paths": [str(f.path) for f in frames],
                "weights_dir": str(self.weights_dir),
                "image_size": self.image_size,
            }
            request_path = Path(scratch) / "request.json"
            response_prefix = Path(scratch) / "response"
            request_path.write_text(json.dumps(request))

            command = [str(RECON_VENV_PYTHON), "-m", "cozmo.recon.worker",
                       str(request_path), str(response_prefix)]
            log.info("running VGGT under %s", RECON_VENV_PYTHON)
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=RECON_TIMEOUT_S, check=False,
                cwd=str(Path(__file__).resolve().parents[2]),
            )

            response_json = Path(str(response_prefix) + ".json")
            if not response_json.is_file():
                raise ReconstructorUnavailable(
                    f"VGGT subprocess produced no response (exit {completed.returncode}): "
                    f"{(completed.stderr or '').strip()[-1500:]}"
                )
            payload = json.loads(response_json.read_text())
            if not payload.get("ok"):
                raise ReconstructorUnavailable(
                    f"VGGT subprocess failed: {payload.get('error')}\n{payload.get('traceback', '')}"
                )

            archive = np.load(str(response_prefix) + ".npz")
            poses = [
                FramePose(frame_index=p["frame_index"], R=np.array(p["R"]), t=np.array(p["t"]),
                         K=np.array(p["K"]))
                for p in payload["poses"]
            ]
            frame_points_local = {i: archive[f"local_points_{i}"] for i in payload["local_frame_indices"]}
            frame_confidence_local = {i: archive[f"local_conf_{i}"] for i in payload["local_frame_indices"]}

            return ReconstructionResult(
                points=archive["points"], point_confidence=archive["point_confidence"],
                point_frame_index=archive["point_frame_index"], poses=poses,
                scale_is_metric=bool(payload["scale_is_metric"]),
                frame_points_local=frame_points_local, frame_confidence_local=frame_confidence_local,
                backbone_name=payload["backbone_name"], stats=payload["stats"],
            )

    def _reconstruct_direct(self, frames: Sequence[Frame]) -> ReconstructionResult:
        torch = self._torch
        from vggt.utils.load_fn import load_and_preprocess_images_square
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        # VGGT's own preprocessing expects file paths; frames are already
        # decoded, so its images are written to a scratch dir once per call.
        # (Kept simple: photo-tier folders are 2-8 images, not a hot path.)
        import tempfile
        import cv2

        with tempfile.TemporaryDirectory(prefix="cozmo-vggt-") as scratch:
            paths = []
            for frame in frames:
                path = Path(scratch) / f"{frame.index:03d}.png"
                cv2.imwrite(str(path), cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR))
                paths.append(str(path))

            images, original_sizes = load_and_preprocess_images_square(paths, self.image_size)
            images = images.to(self.device)

            with torch.no_grad():
                dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
                with torch.autocast(device_type=self.device if self.device != "mps" else "cpu",
                                     dtype=dtype, enabled=self.device == "cuda"):
                    predictions = self.model(images.unsqueeze(0))

        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images.shape[-2:]
        )
        depth = predictions["depth"]              # (1, F, H, W, 1)
        depth_conf = predictions["depth_conf"]     # (1, F, H, W)
        world_points = predictions["world_points"] # (1, F, H, W, 3)

        extrinsic = extrinsic[0].cpu().numpy()     # (F, 3, 4), world-to-camera
        intrinsic = intrinsic[0].cpu().numpy()     # (F, 3, 3)
        depth = depth[0, ..., 0].cpu().numpy()
        depth_conf = depth_conf[0].cpu().numpy()
        world_points = world_points[0].cpu().numpy()

        all_points, all_conf, all_frame_idx = [], [], []
        poses: List[FramePose] = []
        frame_points_local: Dict[int, np.ndarray] = {}
        frame_confidence_local: Dict[int, np.ndarray] = {}

        conf_threshold = float(np.percentile(depth_conf, 50))  # median: keep the confident half

        for i, frame in enumerate(frames):
            Rw2c, tw2c = extrinsic[i][:, :3], extrinsic[i][:, 3]
            R = Rw2c.T                             # camera-to-world
            t = -R @ tw2c
            poses.append(FramePose(frame_index=frame.index, R=R, t=t, K=intrinsic[i]))

            conf = depth_conf[i].reshape(-1)
            pts = world_points[i].reshape(-1, 3)
            keep = conf >= conf_threshold
            all_points.append(pts[keep])
            all_conf.append(conf[keep] / max(conf.max(), 1e-6))
            all_frame_idx.append(np.full(keep.sum(), frame.index))

            # Camera-local points, for layout.py's per-view plane fitting: undo
            # the world transform VGGT already applied.
            local = (pts[keep] - t) @ R
            frame_points_local[frame.index] = local.astype(np.float32)
            frame_confidence_local[frame.index] = conf[keep].astype(np.float32)

        return ReconstructionResult(
            points=np.concatenate(all_points).astype(np.float32),
            point_confidence=np.concatenate(all_conf).astype(np.float32),
            point_frame_index=np.concatenate(all_frame_idx).astype(np.int32),
            poses=poses,
            scale_is_metric=False,
            frame_points_local=frame_points_local,
            frame_confidence_local=frame_confidence_local,
            backbone_name=self.name,
            stats={
                "device": self.device,
                "image_size": self.image_size,
                "confidence_threshold": round(conf_threshold, 4),
                "frames": len(frames),
            },
        )


# --------------------------------------------------------------------------
# Registry: how a second backbone gets swapped in by config
# --------------------------------------------------------------------------

class StubReconstructor:
    """A trivial backbone: one point per frame, on a small ring, no model.

    Exists so CLI/pipeline plumbing (tier dispatch, the output contract, the
    manifest, "there is no --tier flag") can be exercised fast and without
    weights -- the same reason ``build_stub_plan`` exists for the LiDAR/video
    tiers. Never picked by default; select it explicitly with
    ``--backbone stub`` or ``backbone_name="stub"``. Real reconstruction
    accuracy is exercised separately (opt-in, see tests/test_recon.py).
    """

    name = "stub"

    def __init__(self, weights_dir: Optional[Path] = None, **_: Any) -> None:
        pass

    def reconstruct(self, frames: Sequence[Frame]) -> ReconstructionResult:
        if len(frames) < 2:
            raise ValueError("need at least 2 frames")
        n = len(frames)
        rng = np.random.default_rng(0)
        # A trivial 3 x 3 x 2.5 box, world y up, floor at world y=0 -- just
        # enough vertical structure (floor, ceiling, one wall) for the per-view
        # plane fitter to find something. Not a room; only meant to exercise
        # the CLI/manifest/schema plumbing without needing real weights.
        floor = np.stack([rng.uniform(-1.5, 1.5, 60), np.zeros(60), rng.uniform(-1.5, 1.5, 60)], axis=1)
        ceiling = np.stack([rng.uniform(-1.5, 1.5, 30), np.full(30, 2.5), rng.uniform(-1.5, 1.5, 30)], axis=1)
        wall = np.stack([rng.uniform(-1.5, 1.5, 60), rng.uniform(0.0, 2.5, 60), np.full(60, 1.5)], axis=1)
        cloud = np.concatenate([floor, ceiling, wall])

        angles = np.linspace(-0.3, 0.3, n)
        poses = []
        points, conf, frame_idx = [], [], []
        local_points, local_conf = {}, {}
        for i, frame in enumerate(frames):
            t = np.array([0.2 * i, 1.2, -1.0 + 0.1 * i])
            R = np.eye(3)
            poses.append(FramePose(frame_index=frame.index, R=R, t=t, K=frame.K.copy()))
            points.append(cloud)
            conf.append(np.full(len(cloud), 0.9))
            frame_idx.append(np.full(len(cloud), frame.index))
            # R is identity for every stub camera, so camera-local = world - t.
            local_points[frame.index] = (cloud - t).astype(np.float32)
            local_conf[frame.index] = np.full(len(cloud), 0.9, dtype=np.float32)
        return ReconstructionResult(
            points=np.concatenate(points).astype(np.float32),
            point_confidence=np.concatenate(conf).astype(np.float32),
            point_frame_index=np.concatenate(frame_idx).astype(np.int32),
            poses=poses, scale_is_metric=False,
            frame_points_local=local_points, frame_confidence_local=local_conf,
            backbone_name=self.name, stats={"stub": True, "frames": n},
        )


BACKBONES: Dict[str, type] = {
    "vggt": VGGTReconstructor,
    "stub": StubReconstructor,
}


def get_reconstructor(name: str = "vggt", **kwargs: Any) -> Reconstructor:
    """Config-driven backbone selection.

    A fix-loop entry that reads "the worst gate is X, the fix is swapping the
    backbone" changes one string here (or the ``--backbone`` CLI flag) and
    nothing else -- scale.py and layout.py only depend on ReconstructionResult.
    """
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone '{name}'; available: {sorted(BACKBONES)}")
    return BACKBONES[name](**kwargs)
