"""Multi-room stitching and drift correction.

Drift correction is exercised here with an engineered synthetic trajectory
(an exact, known loop closure) rather than only against the real capture,
since the real capture is gitignored (too large to commit) and this needs to
run in CI. See README for the real-capture numbers this was built against.
"""

from __future__ import annotations

import numpy as np
import pytest

from cozmo.stitch.drift import (
    KEYFRAME_STRIDE,
    correct_trajectory,
    find_loop_closures,
)


def _straight_walk_with_drift(n=200, growing_drift=True):
    """A walk down a hallway and back, on a return leg offset sideways from
    the outbound one -- a walker who has drifted, seen from above. The offset
    grows with distance from the turn-around point if ``growing_drift``
    (accumulated VIO error), or stays constant if not (an already-corrected
    trajectory, used to check the "no closures" path is well-conditioned too).
    Every return-leg pose sits near the outbound pose at the same x, so both
    are genuine, findable loop-closure pairs; the walk also ends near (x=0),
    close to where it started.
    """
    R = np.eye(3)
    poses = []
    half = n // 2
    for i in range(n):
        if i < half:
            t = np.array([i * 0.05, 0.0, 0.0])
        else:
            steps_back = i - half
            true_x = half * 0.05 - steps_back * 0.05
            offset = (steps_back * 0.002) if growing_drift else 0.03
            t = np.array([true_x, 0.0, offset])
        poses.append((i, R, t))
    return poses


class TestLoopClosureDetection:
    def test_finds_the_engineered_revisit(self):
        poses = _straight_walk_with_drift()
        positions = np.array([t for _, _, t in poses])
        closures = find_loop_closures(positions, min_gap=15)
        # The walk starts and ends near (0,0,*); at least one closure should
        # pair an early-walk keyframe with a late one.
        assert closures
        assert any(abs(i - j) > 50 for i, j, _ in closures)

    def test_no_closures_on_a_one_way_walk(self):
        poses = [(i, np.eye(3), np.array([i * 0.05, 0.0, 0.0])) for i in range(100)]
        positions = np.array([t for _, _, t in poses])
        assert find_loop_closures(positions, min_gap=15) == []

    def test_a_slow_walk_over_a_short_span_is_not_all_one_closure(self):
        """min_gap is what keeps a slow-moving trajectory (many nearby poses,
        all close in time) from being reported as one giant revisit -- every
        candidate pair here is excluded purely by being too close in index,
        regardless of how close in space they are."""
        positions = np.array([[i * 0.001, 0.0, 0.0] for i in range(14)])
        assert find_loop_closures(positions, radius_m=0.25, min_gap=15) == []


class TestTrajectoryCorrection:
    def test_disabled_is_a_stated_no_op(self):
        poses = _straight_walk_with_drift()
        result = correct_trajectory(poses, enabled=False)
        assert result.method == "poses_as_is"
        assert result.loop_closures_used == 0
        # apply() must hand back exactly what it was given.
        R, t = result.apply(50, poses[50][1], poses[50][2])
        assert np.array_equal(t, poses[50][2])

    def test_finds_and_uses_the_engineered_closure(self):
        poses = _straight_walk_with_drift()
        result = correct_trajectory(poses, enabled=True)
        assert result.method == "pose_graph"
        assert result.loop_closures_used > 0
        assert result.residual_closure_error_m is not None

    def test_correction_pulls_the_drifted_end_back_toward_the_start(self):
        """The whole point: the corrected end of the walk must land closer to
        the true (near-zero) endpoint than the raw, drifted one did."""
        poses = _straight_walk_with_drift()
        result = correct_trajectory(poses, enabled=True)
        raw_end = poses[-1][2]
        _, corrected_end = result.apply(poses[-1][0], poses[-1][1], raw_end)
        # True end is (0, 0, 0); raw end has drifted sideways measurably.
        assert np.linalg.norm(raw_end) > 0.15
        assert np.linalg.norm(corrected_end) < np.linalg.norm(raw_end)

    def test_correction_is_smooth_between_keyframes(self):
        """apply() interpolates the correction offset; consecutive raw frames
        must not produce wildly different corrected positions."""
        poses = _straight_walk_with_drift()
        result = correct_trajectory(poses, enabled=True)
        corrected = np.array([result.apply(i, R, t)[1] for i, R, t in poses])
        steps = np.linalg.norm(np.diff(corrected, axis=0), axis=1)
        raw = np.array([t for _, _, t in poses])
        raw_steps = np.linalg.norm(np.diff(raw, axis=0), axis=1)
        # No corrected step should be wildly larger than the raw walk's own
        # step size -- the bug this guards against inflated a 14 m path to
        # nearly 400 m by hard-cutting at keyframe boundaries.
        assert steps.max() < raw_steps.max() * 5

    def test_too_short_a_trajectory_falls_back_cleanly(self):
        poses = [(i, np.eye(3), np.array([0.0, 0.0, float(i)])) for i in range(2)]
        result = correct_trajectory(poses, enabled=True)
        assert result.method == "poses_as_is"
        assert "too short" in result.notes

    def test_as_dict_is_json_safe(self):
        poses = _straight_walk_with_drift()
        result = correct_trajectory(poses, enabled=True)
        payload = result.as_dict()
        assert payload["method"] == "pose_graph"
        assert isinstance(payload["loop_closures_used"], int)


# --------------------------------------------------------------------------
# graph.py: room pose graph, Manhattan snap, overlap resolution
# --------------------------------------------------------------------------

from shapely.geometry import Polygon as _Polygon

from cozmo.geometry.layout import FloorFrame, WallSegment
from cozmo.stitch.graph import RoomForStitch, build_stitch_graph
from cozmo.stitch.match import DoorwayMatch, RoomPairMatch


def _rect_room(room_id, width, depth, door_wall_index=1, door_width=0.9):
    """A width x depth rectangle, walls in order south/east/north/west."""
    corners = [(0, 0), (width, 0), (width, depth), (0, depth)]
    walls = []
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        d = np.array(b) - np.array(a)
        length = np.linalg.norm(d)
        normal = np.array([-d[1], d[0]]) / length
        walls.append(WallSegment(start=a, end=b, length_m=length, normal=tuple(normal)))
    frame = FloorFrame(
        origin=np.zeros(3), up=np.array([0, 1, 0]),
        e1=np.array([1, 0, 0]), e2=np.array([0, 0, 1]), rotation_rad=0.0,
    )
    polygon = _Polygon(corners)
    return RoomForStitch(
        room_id=room_id, frame=frame, polygon=polygon, walls=walls,
        frame_points={}, frame_K={},
        openings={f"{room_id}_door": (door_wall_index, door_width)},
    )


class TestStitchGraphDoorwayOnly:
    """Two rooms, no keypoint matches at all -- purely a matched doorway."""

    def _rooms_and_match(self, width_a=3.0, depth_a=3.0, width_b=2.5, depth_b=2.5):
        room_a = _rect_room("room_a", width_a, depth_a, door_wall_index=1, door_width=0.9)
        room_b = _rect_room("room_b", width_b, depth_b, door_wall_index=3, door_width=0.9)
        match = RoomPairMatch(
            room_a="room_a", room_b="room_b", dinov2_similarity=0.1,
            doorway_matches=[DoorwayMatch("room_a_door", "room_b_door", 0.9, 0.9)],
        )
        return [room_a, room_b], [match]

    def test_two_rooms_end_up_connected_and_non_overlapping(self):
        rooms, matches = self._rooms_and_match()
        result = build_stitch_graph(rooms, matches)
        assert len(result.edges_used) == 1
        assert result.edges_used[0].source == "doorway"

        polygons = {r.room_id: result.transform_polygon(r.room_id, r.polygon) for r in rooms}
        overlap = polygons["room_a"].intersection(polygons["room_b"]).area
        assert overlap < 0.02

    def test_anchor_room_is_not_moved(self):
        rooms, matches = self._rooms_and_match()
        result = build_stitch_graph(rooms, matches, anchor_room_id="room_a")
        assert result.global_xy["room_a"] == pytest.approx([0.0, 0.0], abs=1e-6)
        assert result.global_yaw["room_a"] == pytest.approx(0.0, abs=1e-6)

    def test_no_doorway_or_keypoints_is_a_rejected_edge(self):
        room_a = _rect_room("room_a", 3.0, 3.0)
        room_b = _rect_room("room_b", 2.0, 2.0)
        match = RoomPairMatch(room_a="room_a", room_b="room_b", dinov2_similarity=0.1)
        result = build_stitch_graph([room_a, room_b], [match])
        assert result.edges_used == []
        assert len(result.edges_rejected) == 1


class TestStitchGraphThreeRooms:
    def test_a_chain_of_three_rooms_stays_connected(self):
        a = _rect_room("a", 3.0, 3.0, door_wall_index=1, door_width=0.9)
        b = _rect_room("b", 3.0, 3.0, door_wall_index=3, door_width=0.9)
        # b's east wall (index 1) connects to c.
        b2 = _rect_room("b", 3.0, 3.0, door_wall_index=1, door_width=0.8)
        b.openings["b_door_west"] = (3, 0.9)
        b.openings["b_door_east"] = (1, 0.8)
        c = _rect_room("c", 2.0, 2.0, door_wall_index=3, door_width=0.8)

        matches = [
            RoomPairMatch("a", "b", 0.1, doorway_matches=[DoorwayMatch("a_door", "b_door_west", 0.9, 0.9)]),
            RoomPairMatch("b", "c", 0.1, doorway_matches=[DoorwayMatch("b_door_east", "c_door", 0.8, 0.8)]),
        ]
        result = build_stitch_graph([a, b, c], matches, anchor_room_id="a")
        assert len(result.edges_used) == 2
        polys = {r.room_id: result.transform_polygon(r.room_id, r.polygon) for r in [a, b, c]}
        assert polys["a"].intersection(polys["b"]).area < 0.02
        assert polys["b"].intersection(polys["c"]).area < 0.02
        assert polys["a"].intersection(polys["c"]).area < 0.02
