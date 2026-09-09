"""The semantic stage as a separate process.

    python -m cozmo.semantics.worker request.json response.json

Why a subprocess and not a function call: open3d (geometry) and torch
(detection) each bundle their own OpenMP runtime, and on macOS a process that
loads both either aborts with ``OMP: Error #179: pthread_mutex_init failed`` or
deadlocks inside the first inference call -- reproducibly, at 0% CPU, with no
error. Setting ``KMP_DUPLICATE_LIB_OK`` papers over it and is documented as
unsafe. Running the two stages in separate processes is the honest fix: it costs
one interpreter start, and it also means a detector that segfaults on a bad
frame loses the damage findings rather than the whole run.

Everything crossing the boundary is JSON. Nothing in this module, or anything it
imports, may import ``cozmo.geometry``.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ..io.lidar import load_lidar_capture
from .project import SurfacePlane
from .stage import run_semantics


def surface_to_dict(surface: SurfacePlane) -> Dict[str, Any]:
    return {
        "surface_id": surface.surface_id,
        "room_id": surface.room_id,
        "kind": surface.kind,
        "origin": [float(v) for v in surface.origin],
        "normal": [float(v) for v in surface.normal],
        "u_axis": [float(v) for v in surface.u_axis],
        "v_axis": [float(v) for v in surface.v_axis],
        "u_extent": float(surface.u_extent),
        "v_extent": float(surface.v_extent),
        "area_m2": float(surface.area_m2),
        "wall_id": surface.wall_id,
        "has_opening": bool(surface.has_opening),
        "is_exterior": surface.is_exterior,
    }


def surface_from_dict(data: Dict[str, Any]) -> SurfacePlane:
    return SurfacePlane(
        surface_id=data["surface_id"], room_id=data["room_id"], kind=data["kind"],
        origin=np.array(data["origin"], dtype=float),
        normal=np.array(data["normal"], dtype=float),
        u_axis=np.array(data["u_axis"], dtype=float),
        v_axis=np.array(data["v_axis"], dtype=float),
        u_extent=float(data["u_extent"]), v_extent=float(data["v_extent"]),
        area_m2=float(data["area_m2"]), wall_id=data.get("wall_id"),
        has_opening=bool(data.get("has_opening", False)),
        is_exterior=data.get("is_exterior"),
    )


class _Opening:
    """The subset of a geometric opening detection that crosses the boundary."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.wall_index = int(data["wall_index"])
        self.kind = str(data["kind"])
        self.width_m = float(data["width_m"])
        self.height_m = float(data["height_m"])
        self.offset_along_wall_m = float(data["offset_along_wall_m"])
        self.sill_height_m = float(data.get("sill_height_m", 0.0))
        self.confidence = float(data.get("confidence", 0.0))


def main(argv: List[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} request.json response.json", file=sys.stderr)
        return 2

    request = json.loads(Path(argv[1]).read_text())
    response_path = Path(argv[2])

    try:
        lidar = load_lidar_capture(Path(request["capture_root"]))
        result = run_semantics(
            lidar_capture=lidar,
            surfaces=[surface_from_dict(s) for s in request["surfaces"]],
            ceiling_height_m=float(request["ceiling_height_m"]),
            geometric_openings=[_Opening(o) for o in request["geometric_openings"]],
            room_id=request["room_id"],
            wall_count=int(request["wall_count"]),
            weights_dir=request.get("weights_dir"),
            frame_stride=int(request.get("frame_stride", 60)),
            max_frames=int(request.get("max_frames", 40)),
        )
        payload = {
            "ok": True,
            "available": result.available,
            "warnings": result.warnings,
            "details": result.details,
            "damage": [region.model_dump(mode="json") for region in result.damage],
            "concealed_flags": [flag.model_dump(mode="json") for flag in result.concealed_flags],
            "scope": [item.model_dump(mode="json") for item in result.scope],
            "openings": [opening.as_dict() for opening in result.openings],
        }
    except Exception as exc:  # noqa: BLE001 - the parent must see why, not just that
        payload = {
            "ok": False,
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "warnings": [], "details": {}, "damage": [], "concealed_flags": [],
            "scope": [], "openings": [],
        }

    response_path.write_text(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
