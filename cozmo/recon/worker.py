"""VGGT reconstruction, run in its own interpreter.

VGGT's upstream package pins ``numpy<2`` and needs Python >= 3.10; the rest of
this repo runs numpy 2.x (open3d and several geometry fixes depend on the
numpy-2 API) on Python 3.9. Rather than force one numpy version on a codebase
where it would silently change behaviour elsewhere, VGGT gets its own
virtualenv (``.venv-recon``, built by ``scripts/fetch_weights.sh`` /
``scripts/setup_recon_env.sh``) and this script is what runs inside it.

Same shape as :mod:`cozmo.semantics.worker` and for a related reason: two
incompatible dependency stacks, kept apart by a process boundary instead of by
hoping they never collide in one environment.

    python -m cozmo.recon.worker request.json response_prefix

Writes ``<response_prefix>.json`` (scalars, poses, stats) and
``<response_prefix>.npz`` (points, confidence, per-frame local clouds).
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def main(argv: List[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} request.json response_prefix", file=sys.stderr)
        return 2

    request = json.loads(Path(argv[1]).read_text())
    response_prefix = Path(argv[2])

    try:
        from .backbone import VGGTReconstructor

        reconstructor = VGGTReconstructor(
            weights_dir=request.get("weights_dir"),
            image_size=int(request.get("image_size", 518)),
            _direct=True,   # bypass the subprocess dispatch -- we ARE the subprocess
        )

        # Frames are re-loaded here rather than passed as arrays: this process
        # has its own numpy build, and pickling arrays across a numpy major
        # version boundary is exactly the kind of silent corruption a process
        # boundary is supposed to prevent.
        from .frames import load_frame

        frames = [load_frame(Path(p), i) for i, p in enumerate(request["frame_paths"])]
        result = reconstructor.reconstruct(frames)

        frame_indices = sorted(result.frame_points_local)
        np.savez(
            str(response_prefix) + ".npz",
            points=result.points,
            point_confidence=result.point_confidence,
            point_frame_index=result.point_frame_index,
            **{f"local_points_{i}": result.frame_points_local[i] for i in frame_indices},
            **{f"local_conf_{i}": result.frame_confidence_local[i] for i in frame_indices},
        )
        payload: Dict[str, Any] = {
            "ok": True,
            "backbone_name": result.backbone_name,
            "scale_is_metric": result.scale_is_metric,
            "stats": result.stats,
            "local_frame_indices": frame_indices,
            "poses": [
                {"frame_index": p.frame_index, "R": p.R.tolist(), "t": p.t.tolist(), "K": p.K.tolist()}
                for p in result.poses
            ],
        }
    except Exception as exc:  # noqa: BLE001 - the parent needs the reason, not just failure
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}

    Path(str(response_prefix) + ".json").write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
