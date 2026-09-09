"""A metric monocular depth model: ZoeDepth (Intel/zoedepth-nyu).

Trained on NYU-Depth-v2 -- indoor rooms, output directly in metres -- which is
why it is the choice here over an outdoor-oriented metric model. Used by
scale.py's strongest cue: align this to the backbone's own scale-free depth by
a least-squares ratio and the result is a scale factor.

Loaded strictly from local disk, same convention as every other model in this
repo: a benchmark run must not depend on a remote file server.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("cozmo.recon.metric_depth")

DEFAULT_WEIGHTS_DIR = Path(os.environ.get("COZMO_WEIGHTS_DIR", "weights"))
ZOEDEPTH_DIR = "zoedepth-nyu"

_model = None
_processor = None
_device = None


class MetricDepthUnavailable(RuntimeError):
    """Weights are missing or the model failed to load."""


def _load(weights_dir: Optional[Path] = None) -> None:
    global _model, _processor, _device
    if _model is not None:
        return

    try:
        import torch
        from transformers import ZoeDepthForDepthEstimation, ZoeDepthImageProcessor
    except ImportError as exc:
        raise MetricDepthUnavailable(f"torch/transformers not installed: {exc}") from exc

    directory = Path(weights_dir or DEFAULT_WEIGHTS_DIR) / ZOEDEPTH_DIR
    if not (directory / "model.safetensors").is_file():
        raise MetricDepthUnavailable(
            f"ZoeDepth weights not found in {directory}. Run scripts/fetch_weights.sh first."
        )

    _device = "mps" if torch.backends.mps.is_available() else "cpu"
    _processor = ZoeDepthImageProcessor.from_pretrained(directory, local_files_only=True)
    _model = ZoeDepthForDepthEstimation.from_pretrained(
        directory, local_files_only=True
    ).to(_device).eval()
    log.info("ZoeDepth loaded on %s from %s", _device, directory)


def estimate_metric_depth(image: np.ndarray, weights_dir: Optional[Path] = None) -> Optional[np.ndarray]:
    """RGB image (H, W, 3) uint8 -> metric depth map (H, W) float32, in metres."""
    _load(weights_dir)
    import torch

    inputs = _processor(images=image, return_tensors="pt").to(_device)
    with torch.no_grad():
        outputs = _model(**inputs)

    depth = _processor.post_process_depth_estimation(
        outputs, source_sizes=[(image.shape[0], image.shape[1])]
    )[0]["predicted_depth"]
    return depth.cpu().numpy().astype(np.float32)
