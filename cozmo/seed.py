"""Global determinism.

Every entry point calls :func:`set_global_seeds` before touching anything that
could be stochastic. The seed used is recorded in run_manifest.json so a run can
be replayed bit-for-bit -- the repeatability gate is meaningless if our own
process is a source of variance.
"""

from __future__ import annotations

import os
import random
from typing import Dict

DEFAULT_SEED = 20260908


def set_global_seeds(seed: int = DEFAULT_SEED) -> Dict[str, object]:
    """Seed every RNG we might touch. Returns a record for the run manifest."""
    record: Dict[str, object] = {"seed": seed, "seeded": ["random"], "unavailable": []}

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        record["unavailable"].append("numpy")
    else:
        np.random.seed(seed)
        record["seeded"].append("numpy")

    try:
        import open3d as o3d
    except ImportError:
        record["unavailable"].append("open3d")
    else:
        # Open3D's RANSAC (segment_plane) draws from its own global RNG, not
        # numpy's. Without this, two runs over the same capture fit slightly
        # different floor planes and every dimension downstream moves -- which
        # is precisely the failure the repeatability gate exists to catch, and
        # it would have been ours rather than the sensor's.
        o3d.utility.random.seed(seed)
        record["seeded"].append("open3d")

    try:
        import torch
    except ImportError:
        # torch is optional until a learned model lands in the pipeline.
        record["unavailable"].append("torch")
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        record["seeded"].append("torch")

    return record
