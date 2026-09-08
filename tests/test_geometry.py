"""Reconstruction against known geometry.

The fixtures are ray-traced from a box room, so every number here has an exact
answer and the assertions are tolerances, not snapshots. Where a tolerance is
loose the reason is stated: it is either the 5 cm occupancy grid or the voxel
downsample, and both are visible in the plan's own intervals.
"""

from __future__ import annotations

import numpy as np
import pytest

from cozmo.geometry.fuse import MIN_FRAMES_FUSED, fuse_capture, voxel_downsample
from cozmo.geometry.layout import dominant_rotation, extract_layout
from cozmo.geometry.openings import build_wall_grid, detect_openings, wall_observation_fractions
from cozmo.geometry.planes import _patchiness, estimate_ceiling, fit_ceiling, fit_floor
from cozmo.io.lidar import load_lidar_capture

CELL_M = 0.05


@pytest.fixture(scope="module")
def scene(request):
    """Fuse the ceiling-observed fixture once for the whole module."""
    root = request.config.rootpath / "tests" / "fixtures" / "captures" / "synthetic_room"
    lidar = load_lidar_capture(root)
    cloud = fuse_capture(lidar)
    floor = fit_floor(cloud.points)
    layout = extract_layout(cloud.points, floor, cloud.trajectory)
    return {"lidar": lidar, "cloud": cloud, "floor": floor, "layout": layout}


@pytest.fixture(scope="module")
def scene_no_ceiling(request):
    root = request.config.rootpath / "tests" / "fixtures" / "captures" / "synthetic_no_ceiling"
    lidar = load_lidar_capture(root)
    cloud = fuse_capture(lidar)
    floor = fit_floor(cloud.points)
    layout = extract_layout(cloud.points, floor, cloud.trajectory)
    return {"lidar": lidar, "cloud": cloud, "floor": floor, "layout": layout}


class TestFusion:
    def test_points_land_inside_the_room(self, scene, synthetic_truth):
        """Every fused point must sit on a surface of the box, not outside it.

        This is the test that catches a depth convention error: Stray depth is
        camera-frame z, and treating it as distance along the ray inflates the
        room by 1/cos(angle) -- 8% at the frame edge.

        The room is yawed relative to the world frame, so the check is made in
        room coordinates.
        """
        from tests.fixtures.synthesize import _room_to_world

        points = scene["cloud"].points @ _room_to_world()   # world -> room
        assert points[:, 0].min() == pytest.approx(0.0, abs=0.04)
        assert points[:, 0].max() == pytest.approx(synthetic_truth["width"], abs=0.04)
        assert points[:, 2].min() == pytest.approx(0.0, abs=0.04)
        assert points[:, 2].max() == pytest.approx(synthetic_truth["depth"], abs=0.04)
        assert points[:, 1].max() == pytest.approx(synthetic_truth["ceiling"], abs=0.04)

    def test_stride_is_raised_when_a_capture_is_short(self, scene):
        """A stride tuned for 1700 frames must not use two frames of sixteen."""
        assert scene["cloud"].frames_used == 16
        assert scene["cloud"].stats["requested_stride"] == 10
        assert scene["cloud"].stride == 1

    def test_confidence_mask_is_recorded(self, scene):
        assert scene["cloud"].min_confidence == 2

    def test_voxel_downsample_reduces_and_preserves_extent(self, scene):
        raw = scene["cloud"].points
        coarse = voxel_downsample(raw, 0.20)
        assert len(coarse) < len(raw)
        assert coarse[:, 0].max() == pytest.approx(raw[:, 0].max(), abs=0.25)

    def test_voxel_size_zero_is_a_no_op(self):
        points = np.random.default_rng(0).random((100, 3)).astype(np.float32)
        assert len(voxel_downsample(points, 0.0)) == 100


class TestFloor:
    def test_floor_is_found_at_zero(self, scene):
        assert scene["floor"].height == pytest.approx(0.0, abs=0.02)

    def test_floor_is_level(self, scene):
        tilt = np.degrees(np.arccos(abs(float(np.dot(scene["floor"].normal, [0, 1, 0])))))
        assert tilt < 1.0

    def test_normal_points_up(self, scene):
        assert float(np.dot(scene["floor"].normal, [0, 1, 0])) > 0


class TestCeiling:
    def test_measured_when_the_ceiling_was_seen(self, scene, synthetic_truth):
        estimate = estimate_ceiling(
            scene["cloud"].points, scene["floor"],
            camera_height=scene["cloud"].camera_height_median,
            wall_top_heights=scene["layout"].wall_top_heights,
        )
        assert estimate.method == "measured_plane"
        assert estimate.height_above_floor_m == pytest.approx(synthetic_truth["ceiling"], abs=0.03)
        assert estimate.ci_95[0] <= synthetic_truth["ceiling"] <= estimate.ci_95[1]

    def test_not_measured_when_the_ceiling_was_not_seen(self, scene_no_ceiling):
        assert fit_ceiling(
            scene_no_ceiling["cloud"].points, scene_no_ceiling["floor"].height,
            scene_no_ceiling["cloud"].camera_height_median,
        ) is None

    def test_fallback_is_honest_about_being_a_fallback(self, scene_no_ceiling, synthetic_truth):
        estimate = estimate_ceiling(
            scene_no_ceiling["cloud"].points, scene_no_ceiling["floor"],
            camera_height=scene_no_ceiling["cloud"].camera_height_median,
            wall_top_heights=scene_no_ceiling["layout"].wall_top_heights,
        )
        assert estimate.method != "measured_plane"
        assert not estimate.measured
        assert estimate.note
        # Wrong is acceptable here; a confidently wrong interval is not.
        assert estimate.ci_95[0] <= synthetic_truth["ceiling"] <= estimate.ci_95[1]

    def test_fallback_interval_is_wider_than_a_measurement(self, scene, scene_no_ceiling):
        def width(s):
            estimate = estimate_ceiling(
                s["cloud"].points, s["floor"], s["cloud"].camera_height_median,
                s["layout"].wall_top_heights,
            )
            return estimate.ci_95[1] - estimate.ci_95[0]

        assert width(scene_no_ceiling) > 3 * width(scene)

    def test_a_ring_of_wall_tops_is_not_a_surface(self):
        """The guard that stops a capture with no ceiling from reporting one."""
        angle = np.linspace(0, 2 * np.pi, 400)
        ring = np.stack([2 + 1.8 * np.cos(angle), np.full(400, 2.5), 1.4 + 1.3 * np.sin(angle)], axis=1)
        _fill, interior = _patchiness(ring)
        assert interior < 0.15

        grid = np.mgrid[0:36, 0:28].reshape(2, -1).T * 0.1
        surface = np.stack([grid[:, 0], np.full(len(grid), 2.5), grid[:, 1]], axis=1)
        _fill, interior = _patchiness(surface)
        assert interior > 0.5


class TestLayout:
    def test_wall_lengths_match_the_room(self, scene, synthetic_truth):
        lengths = sorted(w.length_m for w in scene["layout"].walls)
        assert len(lengths) == 4
        assert lengths[0] == pytest.approx(synthetic_truth["depth"], abs=0.05)
        assert lengths[1] == pytest.approx(synthetic_truth["depth"], abs=0.05)
        assert lengths[2] == pytest.approx(synthetic_truth["width"], abs=0.05)
        assert lengths[3] == pytest.approx(synthetic_truth["width"], abs=0.05)

    def test_floor_area_matches(self, scene, synthetic_truth):
        assert scene["layout"].floor_area_m2 == pytest.approx(synthetic_truth["floor_area"], rel=0.02)

    def test_perimeter_matches(self, scene, synthetic_truth):
        expected = 2 * (synthetic_truth["width"] + synthetic_truth["depth"])
        assert scene["layout"].perimeter_m == pytest.approx(expected, rel=0.02)

    def test_polygon_is_closed_and_simple(self, scene):
        polygon = scene["layout"].polygon
        assert polygon.is_valid and polygon.is_simple
        assert polygon.exterior.is_ring

    def test_dominant_rotation_recovers_a_known_angle(self):
        """Wall normals 90 deg apart are one family; the estimator must agree."""
        angle = np.radians(23.0)
        directions = np.concatenate([
            np.full(500, angle), np.full(500, angle + np.pi / 2),
            np.full(500, angle + np.pi), np.full(500, angle - np.pi / 2),
        ])
        normals = np.stack([np.cos(directions), np.sin(directions)], axis=1)
        rotation, score = dominant_rotation(normals)
        assert np.degrees(rotation) == pytest.approx(23.0, abs=0.5)
        assert score > 0.99

    def test_dominant_rotation_reports_weak_structure(self):
        """A round room has no dominant axes and the score has to say so."""
        angles = np.linspace(0, 2 * np.pi, 2000)
        normals = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        _rotation, score = dominant_rotation(normals)
        assert score < 0.1

    def test_manhattan_snap_off_inflates_the_footprint(self, scene, synthetic_truth):
        """The drift ablation has to move the number it claims to move."""
        raw = extract_layout(
            scene["cloud"].points, scene["floor"], scene["cloud"].trajectory,
            manhattan_snap=False,
        )
        assert raw.floor_area_m2 > scene["layout"].floor_area_m2
        assert scene["layout"].floor_area_m2 == pytest.approx(synthetic_truth["floor_area"], rel=0.02)

    def test_layout_is_deterministic(self, scene):
        again = extract_layout(scene["cloud"].points, scene["floor"], scene["cloud"].trajectory)
        assert [w.length_m for w in again.walls] == [w.length_m for w in scene["layout"].walls]


class TestOpenings:
    def test_door_and_window_are_both_found(self, scene, synthetic_truth):
        detections = detect_openings(
            scene["layout"].points_uv, scene["layout"].points_height,
            scene["layout"].walls, synthetic_truth["ceiling"],
        )
        kinds = sorted(d.kind for d in detections)
        assert kinds == ["door", "window"]

    def test_window_dimensions(self, scene, synthetic_truth):
        detections = detect_openings(
            scene["layout"].points_uv, scene["layout"].points_height,
            scene["layout"].walls, synthetic_truth["ceiling"],
        )
        window = next(d for d in detections if d.kind == "window")
        # One occupancy cell is 5 cm, so that is the floor on any of these.
        assert window.width_m == pytest.approx(synthetic_truth["window_width"], abs=2 * CELL_M)
        assert window.sill_height_m == pytest.approx(synthetic_truth["window_sill"], abs=2 * CELL_M)
        assert window.height_m == pytest.approx(synthetic_truth["window_height"], abs=2 * CELL_M)

    def test_door_reaches_the_floor(self, scene, synthetic_truth):
        detections = detect_openings(
            scene["layout"].points_uv, scene["layout"].points_height,
            scene["layout"].walls, synthetic_truth["ceiling"],
        )
        door = next(d for d in detections if d.kind == "door")
        assert door.sill_height_m == 0.0
        assert door.height_m > 1.4
        assert door.width_m == pytest.approx(synthetic_truth["door_width"], abs=3 * CELL_M)

    def test_openings_need_the_full_wall_height_not_the_wall_band(self, scene, synthetic_truth):
        """Detecting against the 0.35-1.80 m band loses the door entirely.

        The band's own lower edge becomes the doorway's sill, which is then too
        high for a door, and its upper edge cuts off the head.
        """
        banded = detect_openings(
            scene["layout"].wall_band_uv, scene["layout"].wall_band_height,
            scene["layout"].walls, synthetic_truth["ceiling"],
        )
        assert not any(d.kind == "door" for d in banded)

    def test_no_openings_claimed_in_an_unobserved_wall(self, scene, synthetic_truth):
        """Emptiness that was never looked at is missing data, not a doorway."""
        empty = np.empty((0, 2))
        assert detect_openings(empty, np.empty(0), scene["layout"].walls, 2.5) == []

    def test_observation_fraction_is_reported_per_wall(self, scene, synthetic_truth):
        fractions = wall_observation_fractions(
            scene["layout"].points_uv, scene["layout"].points_height,
            scene["layout"].walls, synthetic_truth["ceiling"],
        )
        assert len(fractions) == len(scene["layout"].walls)
        assert all(0.0 <= f <= 1.0 for f in fractions)
        assert max(fractions) > 0.8

    def test_wall_grid_orientation(self, scene, synthetic_truth):
        """Row 0 is the floor; the grid must not be built upside down."""
        wall = scene["layout"].walls[0]
        grid = build_wall_grid(
            scene["layout"].points_uv, scene["layout"].points_height,
            np.array(wall.start), np.array(wall.end), synthetic_truth["ceiling"],
        )
        assert grid.occupied.shape[0] == pytest.approx(synthetic_truth["ceiling"] / CELL_M, abs=1)
        assert grid.length_m == pytest.approx(wall.length_m, abs=1e-6)
