"""Room matching, run in its own process.

    python -m cozmo.stitch.worker request.json response.json

Same reason as :mod:`cozmo.semantics.worker`: torch (DINOv2, SuperPoint,
LightGlue) and open3d cannot share a process on this platform, and the caller
(``cozmo.pipeline.run``) already has open3d loaded for the LiDAR path.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

from .match import MatcherUnavailable, RoomFrames, RoomMatcher


def main(argv: List[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} request.json response.json", file=sys.stderr)
        return 2

    request = json.loads(Path(argv[1]).read_text())
    response_path = Path(argv[2])

    try:
        rooms = [
            RoomFrames(
                room_id=r["room_id"], frame_paths=r["frame_paths"],
                frame_indices=r["frame_indices"], openings=r.get("openings", []),
            )
            for r in request["rooms"]
        ]
        matcher = RoomMatcher(weights_dir=request.get("weights_dir"))
        matches = matcher.match_rooms(rooms)
        payload: Dict[str, Any] = {"ok": True, "matches": [m.as_dict() for m in matches]}
    except MatcherUnavailable as exc:
        payload = {"ok": False, "unavailable": True, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - the parent needs the reason, not just failure
        payload = {"ok": False, "unavailable": False, "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()}

    response_path.write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
