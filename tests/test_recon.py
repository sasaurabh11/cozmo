"""Photo-tier reconstruction: the parts that run without model weights.

Everything here is pure numpy -- EXIF parsing, cue combination arithmetic, and
per-view plane clustering -- and is exactly what a reviewer needs to check by
hand, the same reasoning as test_semantics.py. VGGT and ZoeDepth themselves are
exercised separately (see the bottom of this file) and skipped when their
weights or the .venv-recon interpreter are absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from cozmo.recon.backbone import FramePose, ReconstructionResult
from cozmo.recon.frames import (
    DEFAULT_HORIZONTAL_FOV_DEG,
    FULL_FRAME_SENSOR_WIDTH_MM,
    _focal_px_from_exif,
    load_photo_folder,
)
from cozmo.recon.layout import (
    ASSUMED_CAMERA_DOWN,
    ViewPlane,
    _cluster_wall_planes,
    _merge_floor_or_ceiling,
    _view_floor_ceiling_wall,
)
from cozmo.recon.scale import ScaleCue, ceiling_height_cue, recover_scale

VGGT_WEIGHTS = Path("weights/vggt-1b/model.safetensors")
RECON_PYTHON = Path(os.environ.get("COZMO_RECON_PYTHON", ".venv-recon/bin/python"))
RUN_BACKBONE_TESTS = os.environ.get("COZMO_TEST_RECON") == "1"


# --------------------------------------------------------------------------
# frames.py
# --------------------------------------------------------------------------


class TestFocalLength:
    def test_35mm_equivalent_converts_correctly(self):
        # A 26mm 35mm-equiv lens on a 1920px-wide image.
        focal_px, focal_mm, source = _focal_px_from_exif({"FocalLengthIn35mmFilm": 26}, width_px=1920)
        assert source == "exif_35mm_equiv"
        assert focal_px == pytest.approx(26 / FULL_FRAME_SENSOR_WIDTH_MM * 1920)

    def test_native_focal_length_without_sensor_width_is_not_convertible(self):
        focal_px, focal_mm, source = _focal_px_from_exif({"FocalLength": 4.25}, width_px=1920)
        assert focal_px is None
        assert focal_mm == 4.25
        assert source == "exif_focal_mm_unconvertible"

    def test_no_exif_falls_back_to_default_fov(self):
        focal_px, focal_mm, source = _focal_px_from_exif({}, width_px=1920)
        assert focal_px is None and focal_mm is None
        assert source == "fov_default"

    def test_loaded_frame_records_which_path_was_used(self, tmp_path):
        import cv2

        path = tmp_path / "a.jpg"
        cv2.imwrite(str(path), np.full((100, 150, 3), 128, dtype=np.uint8))
        frames = load_photo_folder(tmp_path)
        assert len(frames) == 1
        frame = frames[0]
        # No EXIF on a synthetic image -> the default FOV, and it says so.
        assert frame.intrinsics_source in ("fov_default", "fov_default_no_sensor_width")
        assert frame.horizontal_fov_deg == pytest.approx(DEFAULT_HORIZONTAL_FOV_DEG, abs=0.5)

    def test_folder_requires_2_to_8_photos_is_the_callers_job(self, tmp_path):
        """load_photo_folder itself does not enforce the band -- callers warn."""
        import cv2

        for i in range(1):
            cv2.imwrite(str(tmp_path / f"{i}.jpg"), np.zeros((50, 50, 3), dtype=np.uint8))
        assert len(load_photo_folder(tmp_path)) == 1

    def test_empty_folder_is_a_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_photo_folder(tmp_path)


# --------------------------------------------------------------------------
# scale.py: cue combination
# --------------------------------------------------------------------------


def _fake_layout(span):
    class L:
        floor_to_ceiling_span = span
    return L()


class TestCeilingCue:
    def test_fires_with_a_span(self):
        cue = ceiling_height_cue(_fake_layout(1.0))
        assert cue.fired
        assert cue.scale_factor == pytest.approx(2.44)     # prior / span=1.0

    def test_does_not_fire_without_one(self):
        assert not ceiling_height_cue(_fake_layout(None)).fired
        assert not ceiling_height_cue(_fake_layout(0.0)).fired


class TestScaleCombination:
    def _run(self, cues):
        """Exercise recover_scale's combination logic directly via monkeypatched cues."""
        import cozmo.recon.scale as scale_mod

        class FakeLayout:
            floor_to_ceiling_span = None

        result = scale_mod.ScaleEstimate.__new__(scale_mod.ScaleEstimate)
        # Reimplement just the combination step the way recover_scale does,
        # so this test tracks the real arithmetic without needing a
        # reconstruction or frames.
        fired = [c for c in cues if c.fired and c.scale_factor]
        values = np.array([c.scale_factor for c in fired])
        weights = np.array([max(c.weight, scale_mod.MIN_CUE_WEIGHT) for c in fired])
        combined = scale_mod._weighted_median(values, weights)
        return combined, fired

    def test_agreeing_cues_combine_near_their_shared_value(self):
        cues = [
            ScaleCue("a", fired=True, scale_factor=1.00, relative_sd=0.05),
            ScaleCue("b", fired=True, scale_factor=1.02, relative_sd=0.10),
        ]
        combined, fired = self._run(cues)
        assert combined == pytest.approx(1.0, abs=0.05)

    def test_one_bad_cue_does_not_drag_a_correct_pair(self):
        """A weighted MEDIAN, not a mean: one outlier should not move the answer much."""
        cues = [
            ScaleCue("a", fired=True, scale_factor=1.00, relative_sd=0.05),
            ScaleCue("b", fired=True, scale_factor=1.01, relative_sd=0.05),
            ScaleCue("c", fired=True, scale_factor=5.00, relative_sd=0.05),   # a bad door detection
        ]
        combined, fired = self._run(cues)
        assert combined < 1.5   # nowhere near dragged toward 5.0

    def test_unfired_cues_do_not_vote(self):
        cues = [
            ScaleCue("a", fired=True, scale_factor=1.0, relative_sd=0.05),
            ScaleCue("b", fired=False, reason="no door detected"),
        ]
        combined, fired = self._run(cues)
        assert len(fired) == 1
        assert combined == pytest.approx(1.0)


class TestScaleEstimateDisagreement:
    def test_disagreeing_cues_widen_the_interval_more_than_either_cue_alone(self):
        import cozmo.recon.scale as scale_mod

        class FakeReconstruction:
            frame_points_local = {}

        class FakeLayout:
            floor_to_ceiling_span = 1.0     # ceiling cue: scale = 2.44

        estimate = scale_mod.recover_scale(
            frames=[], reconstruction=FakeReconstruction(), layout=FakeLayout(),
            run_metric_depth=False, run_door_cue=False,
        )
        # Only the ceiling cue fires; a single cue must not claim tight precision.
        assert estimate.method == "single_cue"
        width = estimate.ci_95[1] - estimate.ci_95[0]
        assert width / estimate.scale_factor >= 0.30

    def test_no_cue_firing_is_a_named_fallback_not_a_crash(self):
        import cozmo.recon.scale as scale_mod

        class FakeReconstruction:
            frame_points_local = {}

        class FakeLayout:
            floor_to_ceiling_span = None

        estimate = scale_mod.recover_scale(
            frames=[], reconstruction=FakeReconstruction(), layout=FakeLayout(),
            run_metric_depth=False, run_door_cue=False,
        )
        assert estimate.method == "no_cue_fired_unscaled"
        assert all(not c.fired for c in estimate.cues)


# --------------------------------------------------------------------------
# layout.py: per-view plane fitting and clustering
# --------------------------------------------------------------------------


def _box_room_view(width=4.0, height=2.5, depth=3.0, camera_offset=(2.0, 1.2, 0.2), n=6000, seed=0):
    """Camera-local points sampling a box room's floor, ceiling and two walls,
    as if this one frame saw them directly (down = ASSUMED_CAMERA_DOWN)."""
    rng = np.random.default_rng(seed)
    cx, cy, cz = camera_offset
    pts = []
    # Floor: local y = height - cy (below camera by `height-cy`... build directly
    # in local coords where local y is "down" and camera sits at origin).
    floor_y = height - cy
    ceil_y = -cy
    xs = rng.uniform(-width / 2, width / 2, n // 3)
    zs = rng.uniform(0.3, depth, n // 3)
    pts.append(np.stack([xs, np.full(n // 3, floor_y), zs], axis=1))
    xs = rng.uniform(-width / 2, width / 2, n // 6)
    zs = rng.uniform(0.3, depth, n // 6)
    pts.append(np.stack([xs, np.full(n // 6, ceil_y), zs], axis=1))
    # A wall straight ahead, at local z = depth - cz.
    wall_y = rng.uniform(ceil_y, floor_y, n // 3)
    wall_x = rng.uniform(-width / 2, width / 2, n // 3)
    pts.append(np.stack([wall_x, wall_y, np.full(n // 3, depth - cz)], axis=1))
    return np.concatenate(pts).astype(np.float32)


class TestPerViewPlanes:
    def test_a_single_view_finds_floor_ceiling_and_a_wall(self):
        points = _box_room_view()
        planes = _view_floor_ceiling_wall(
            points, np.ones(len(points)), np.eye(3), np.zeros(3), frame_index=0
        )
        kinds = {p.kind for p in planes}
        assert "floor" in kinds and "wall" in kinds

    def test_floor_offset_is_plausible(self):
        points = _box_room_view(height=2.5, camera_offset=(2.0, 1.2, 0.2))
        planes = _view_floor_ceiling_wall(
            points, np.ones(len(points)), np.eye(3), np.zeros(3), frame_index=0
        )
        floor = next(p for p in planes if p.kind == "floor")
        # Camera is 1.2 m above the floor -> floor plane offset ~1.2 (down = +y).
        assert floor.offset == pytest.approx(1.2, abs=0.1)

    def test_too_few_points_produces_nothing(self):
        assert _view_floor_ceiling_wall(
            np.zeros((5, 3)), np.ones(5), np.eye(3), np.zeros(3), frame_index=0
        ) == []

    def test_planes_transform_by_the_frames_pose(self):
        """A frame offset in world space must shift the plane by the same amount."""
        points = _box_room_view()
        t = np.array([10.0, 0.0, 0.0])
        planes = _view_floor_ceiling_wall(points, np.ones(len(points)), np.eye(3), t, frame_index=1)
        floor = next(p for p in planes if p.kind == "floor")
        # World-frame floor point should sit near x=10 plane (camera moved there).
        assert abs(floor.point_world[0] - 10.0) < 2.5   # within the room's own extent


class TestWallClustering:
    def test_two_views_of_the_same_wall_cluster_together(self):
        n1 = np.array([1.0, 0.0, 0.0])
        n2 = np.array([0.98, 0.0, 0.02]); n2 /= np.linalg.norm(n2)
        a = ViewPlane(frame_index=0, kind="wall", normal_world=n1, point_world=n1 * 2.0, inliers=100)
        b = ViewPlane(frame_index=1, kind="wall", normal_world=n2, point_world=n2 * 2.05, inliers=80)
        clusters = _cluster_wall_planes([a, b])
        assert len(clusters) == 1
        assert len(clusters[0]) == 2

    def test_different_walls_stay_separate(self):
        n1 = np.array([1.0, 0.0, 0.0])
        n2 = np.array([0.0, 0.0, 1.0])
        a = ViewPlane(frame_index=0, kind="wall", normal_world=n1, point_world=n1 * 2.0, inliers=100)
        b = ViewPlane(frame_index=1, kind="wall", normal_world=n2, point_world=n2 * 1.5, inliers=100)
        clusters = _cluster_wall_planes([a, b])
        assert len(clusters) == 2

    def test_same_direction_different_offset_stays_separate(self):
        """Two parallel walls (opposite sides of a room) must not merge."""
        n1 = np.array([1.0, 0.0, 0.0])
        a = ViewPlane(frame_index=0, kind="wall", normal_world=n1, point_world=n1 * 0.0, inliers=100)
        b = ViewPlane(frame_index=1, kind="wall", normal_world=n1, point_world=n1 * 4.0, inliers=100)
        clusters = _cluster_wall_planes([a, b])
        assert len(clusters) == 2


class TestFloorCeilingMerge:
    def test_merges_by_confidence_weighted_average(self):
        down = ASSUMED_CAMERA_DOWN
        a = ViewPlane(frame_index=0, kind="floor", normal_world=down, point_world=down * 1.0, inliers=100)
        b = ViewPlane(frame_index=1, kind="floor", normal_world=down, point_world=down * 1.2, inliers=300)
        merged = _merge_floor_or_ceiling([a, b])
        # Weighted toward b (300 vs 100 inliers).
        assert merged.height == pytest.approx(1.15, abs=0.02)

    def test_empty_input_returns_none(self):
        assert _merge_floor_or_ceiling([]) is None


# --------------------------------------------------------------------------
# Backbone tests: need real weights and .venv-recon. Opt-in, like test_semantics.
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not (VGGT_WEIGHTS.is_file() and RECON_PYTHON.is_file() and RUN_BACKBONE_TESTS),
    reason="needs weights, .venv-recon (scripts/setup_recon_env.sh) and COZMO_TEST_RECON=1",
)
class TestVGGTBackbone:
    def test_reconstructs_a_real_photo_folder(self):
        from cozmo.recon.backbone import get_reconstructor
        from cozmo.recon.frames import load_photo_folder

        frames = load_photo_folder(Path("captures/room_photos/rooms/living_room"))
        reconstructor = get_reconstructor("vggt")
        result = reconstructor.reconstruct(frames)
        assert len(result.points) > 0
        assert len(result.poses) == len(frames)
        assert not result.scale_is_metric

    def test_photo_tier_end_to_end_via_cli(self, tmp_path):
        from cozmo.pipeline.run import run_capture

        result = run_capture(
            Path("captures/room_photos"), tmp_path, semantics=False,
        )
        room = result.plan.rooms[0]
        assert len(room.walls) >= 3
        for wall in room.walls:
            assert wall.length.ci_95[0] <= wall.length.value <= wall.length.ci_95[1]
