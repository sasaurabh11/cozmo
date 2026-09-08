"""Generate synthetic Stray Scanner captures with exactly known geometry.

    python tests/fixtures/synthesize.py

The real capture is 43 MB and lives outside the repo, so the LiDAR path needs a
fixture small enough to commit and exact enough to assert against. These are
ray-traced from a box room with a door and a window cut into it: every number
the pipeline should recover is known to the millimetre, and the ground truth CSV
is written from the same constants.

Two captures, because the ceiling logic has two paths worth testing:

    synthetic_room       ceiling observed  -> measured_plane
    synthetic_no_ceiling identical room, ceiling returns dropped, as when an
                         operator never points the phone up -> scan_cutoff_prior
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CAPTURES = HERE / "captures"

# ---- the room, in metres --------------------------------------------------
ROOM_WIDTH = 3.60      # along x
ROOM_DEPTH = 2.80      # along z
CEILING_HEIGHT = 2.50

# (face, u_min, u_max, v_min, v_max) with u along the face and v up from floor.
DOOR = {"face": "z_min", "u0": 1.10, "u1": 1.95, "v0": 0.00, "v1": 2.03}
WINDOW = {"face": "x_max", "u0": 0.85, "u1": 1.95, "v0": 0.90, "v1": 2.10}

# A real room sits at an arbitrary angle to ARKit's world axes -- the sample
# capture is 23 deg off -- so the fixture is too. An axis-aligned fixture would
# let the layout stage pass without ever estimating the room's own orientation,
# and the drift ablation would have nothing to correct.
ROOM_YAW_DEG = 23.0

DEPTH_W, DEPTH_H = 128, 96
RGB_W, RGB_H = 960, 720
DEPTH_FOCAL = 92.0                    # pixels, ~70 deg horizontal field of view
FRAME_COUNT = 16
CAMERA_HEIGHT = 1.40
MAX_RANGE_M = 5.0


def _rotation(yaw: float, pitch: float) -> np.ndarray:
    """Camera-to-world rotation: camera x right, y down, z forward; world y up."""
    forward = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward], axis=1)


def _quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Rotation matrix -> (qx, qy, qz, qw)."""
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s; qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s; qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s; qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s; qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s; qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s; qz = 0.25 * s
    return float(qx), float(qy), float(qz), float(qw)


def _in_opening(opening: Dict[str, float], u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return (u >= opening["u0"]) & (u <= opening["u1"]) & (v >= opening["v0"]) & (v <= opening["v1"])


def _room_to_world() -> np.ndarray:
    """Rotation taking room coordinates to world coordinates (yaw about up)."""
    yaw = np.radians(ROOM_YAW_DEG)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def render_depth(
    origin: np.ndarray, R: np.ndarray, drop_ceiling: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Ray-trace one depth frame of the room interior. Returns (depth_m, valid)."""
    us, vs = np.meshgrid(np.arange(DEPTH_W), np.arange(DEPTH_H))
    # z stays 1 and the direction is deliberately NOT normalised: Stray depth is
    # the camera-frame z of the hit, not the distance along the ray, so the ray
    # parameter t has to equal that z. Normalising here inflates every dimension
    # by 1/cos(angle from the optical axis) -- about 8% at the frame edge.
    dirs_cam = np.stack([
        (us - DEPTH_W / 2) / DEPTH_FOCAL,
        (vs - DEPTH_H / 2) / DEPTH_FOCAL,
        np.ones_like(us, dtype=float),
    ], axis=-1)
    dirs_world = dirs_cam @ R.T                 # world-space ray directions
    # Trace in the room's own frame; the camera lives in the world frame.
    room_from_world = _room_to_world().T
    dirs = dirs_world @ room_from_world.T
    origin = room_from_world @ origin

    # Slab method: distance to leaving the box through each face.
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([ROOM_WIDTH, CEILING_HEIGHT, ROOM_DEPTH])
    with np.errstate(divide="ignore", invalid="ignore"):
        t_lo = (lo - origin) / dirs
        t_hi = (hi - origin) / dirs
        t_exit = np.where(dirs > 0, t_hi, t_lo)
    t_exit = np.where(np.isfinite(t_exit), t_exit, np.inf)
    t_exit = np.where(t_exit > 0, t_exit, np.inf)

    axis = np.argmin(t_exit, axis=-1)
    t = np.min(t_exit, axis=-1)
    hit = origin + t[..., None] * dirs

    valid = np.isfinite(t) & (t > 0.2) & (t < MAX_RANGE_M)

    # Ceiling hits, when the operator never looked up.
    if drop_ceiling:
        valid &= ~((axis == 1) & (hit[..., 1] > CEILING_HEIGHT - 0.05))

    # Cut the openings out of their faces: nothing comes back through a doorway.
    door_face = (axis == 2) & (hit[..., 2] < 0.05)
    valid &= ~(door_face & _in_opening(DOOR, hit[..., 0], hit[..., 1]))
    window_face = (axis == 0) & (hit[..., 0] > ROOM_WIDTH - 0.05)
    valid &= ~(window_face & _in_opening(WINDOW, hit[..., 2], hit[..., 1]))

    depth = np.where(valid, t, 0.0)
    return depth.astype(np.float32), valid


def camera_path() -> List[Tuple[np.ndarray, float, float]]:
    """A slow turn near the middle of the room, sweeping every wall."""
    poses = []
    room_to_world = _room_to_world()
    for i in range(FRAME_COUNT):
        fraction = i / FRAME_COUNT
        yaw = 2 * np.pi * fraction + np.radians(ROOM_YAW_DEG)
        # A small orbit, so the walk is not a single pivot point.
        origin = room_to_world @ np.array([
            ROOM_WIDTH / 2 + 0.35 * np.cos(2 * np.pi * fraction),
            CAMERA_HEIGHT,
            ROOM_DEPTH / 2 + 0.35 * np.sin(2 * np.pi * fraction),
        ])
        # Sweep well above and below the horizon: at 1.4 m with a 55 deg vertical
        # field of view, a level phone only grazes a 2.5 m ceiling, and the
        # capture would not exercise the measured-ceiling path at all.
        pitch = 0.45 * np.sin(4 * np.pi * fraction)
        poses.append((origin, yaw, pitch))
    return poses


def write_capture(name: str, drop_ceiling: bool) -> Path:
    root = CAPTURES / name
    (root / "depth").mkdir(parents=True, exist_ok=True)
    (root / "confidence").mkdir(parents=True, exist_ok=True)

    rows = ["timestamp, frame, x, y, z, qx, qy, qz, qw"]
    for index, (origin, yaw, pitch) in enumerate(camera_path()):
        R = _rotation(yaw, pitch)
        depth, valid = render_depth(origin, R, drop_ceiling)

        cv2.imwrite(str(root / "depth" / f"{index:06d}.png"),
                    np.clip(depth * 1000.0, 0, 65535).astype(np.uint16))
        confidence = np.where(valid, 2, 0).astype(np.uint8)
        cv2.imwrite(str(root / "confidence" / f"{index:06d}.png"), confidence)

        qx, qy, qz, qw = _quaternion(R)
        rows.append(
            f"{index * 0.1:.6f}, {index:06d}, {origin[0]:.6f}, {origin[1]:.6f}, "
            f"{origin[2]:.6f}, {qx:.8f}, {qy:.8f}, {qz:.8f}, {qw:.8f}"
        )
    (root / "odometry.csv").write_text("\n".join(rows) + "\n")

    # The loader reads intrinsics for the RGB stream and scales them down to the
    # depth resolution, so they are written the same way round here.
    scale = RGB_W / DEPTH_W
    focal = DEPTH_FOCAL * scale
    (root / "camera_matrix.csv").write_text(
        f"{focal:.4f}, 0.0, {RGB_W / 2:.4f}\n"
        f"0.0, {focal:.4f}, {RGB_H / 2:.4f}\n"
        f"0.0, 0.0, 1.0\n"
    )

    writer = cv2.VideoWriter(
        str(root / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (RGB_W, RGB_H)
    )
    for _ in range(FRAME_COUNT):
        writer.write(np.full((RGB_H, RGB_W, 3), 200, dtype=np.uint8))
    writer.release()

    (root / "capture.json").write_text(json.dumps({
        "capture_id": name,
        "tier": "lidar",
        "space_id": "synthetic_room",
        "device": {"model": "synthetic", "has_lidar": True},
        "operator": "fixture",
        "declared_rooms": ["test_room"],
        "notes": (
            f"Ray-traced {ROOM_WIDTH} x {ROOM_DEPTH} m room, ceiling {CEILING_HEIGHT} m. "
            + ("Ceiling returns dropped." if drop_ceiling else "Ceiling observed.")
        ),
    }, indent=2) + "\n")
    return root


def write_ground_truth() -> Path:
    """Exact ground truth, from the same constants the renderer used."""
    path = HERE / "benchmark" / "ground_truth_synthetic.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["room,element,element_id,dimension,value_m,method,notes"]
    for index, length in enumerate([ROOM_WIDTH, ROOM_DEPTH, ROOM_WIDTH, ROOM_DEPTH]):
        lines.append(f"test_room,wall,test_room_w{index},length,{length:.3f},synthetic,exact by construction")
    lines.append(f"test_room,room,test_room,ceiling_height,{CEILING_HEIGHT:.3f},synthetic,exact by construction")
    lines.append(f"test_room,room,test_room,floor_area,{ROOM_WIDTH * ROOM_DEPTH:.3f},synthetic,w x d")
    # Opening ids follow the reconstruction's convention (<wall_id>_op<n>)
    # because the scorer matches ground truth to plans by id. That is a real
    # limitation: ground truth measured by a person in a real room has no way to
    # know these ids, and openings should be matched by position along the wall
    # instead. Until that lands, a change in wall ordering will show up here as
    # a missed opening -- visibly wrong rather than silently wrong.
    lines.append(f"test_room,opening,test_room_w2_op0,width,{DOOR['u1'] - DOOR['u0']:.3f},synthetic,doorway cut in the z_min wall")
    lines.append(f"test_room,opening,test_room_w3_op0,width,{WINDOW['u1'] - WINDOW['u0']:.3f},synthetic,window cut in the x_max wall")
    lines.append(f",property,property,footprint_area,{ROOM_WIDTH * ROOM_DEPTH:.3f},synthetic,single room")
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    print("ceiling observed :", write_capture("synthetic_room", drop_ceiling=False))
    print("ceiling dropped  :", write_capture("synthetic_no_ceiling", drop_ceiling=True))
    print("ground truth     :", write_ground_truth())


if __name__ == "__main__":
    main()
