"""Schema round-trip and the invariants the contract leans on."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from cozmo.schema import (
    Adjacency, DamageClass, DamageRegion, Measurement, Opening, OpeningType, Plan,
    Point2D, Pose2D, Room, Surface, SurfaceKind, Tier, Unit, Wall,
)


def _measurement(value: float, half: float = 0.01) -> Measurement:
    return Measurement.symmetric(value, half)


def _room(room_id: str = "living_room") -> Room:
    ceiling = _measurement(2.45, 0.012)
    walls = [
        Wall(
            id=f"{room_id}_w{i}",
            start=Point2D(x=0.0, y=0.0),
            end=Point2D(x=4.0, y=0.0),
            length=_measurement(4.0),
            height=ceiling,
        )
        for i in range(2)
    ]
    return Room(
        id=room_id, name="Living Room", pose=Pose2D(x=0.0, y=0.0),
        walls=walls, ceiling_height=ceiling,
        floor_area=Measurement.relative(12.0, 0.02, Unit.SQUARE_METERS),
        perimeter=_measurement(14.0, 0.05),
    )


class TestMeasurement:
    def test_interval_must_bracket_value(self):
        with pytest.raises(ValidationError, match="does not bracket"):
            Measurement(value=2.5, ci_95=(2.6, 2.7))

    def test_interval_bounds_must_be_ordered(self):
        with pytest.raises(ValidationError, match="out of order"):
            Measurement(value=2.5, ci_95=(2.6, 2.4))

    def test_symmetric_and_relative_constructors(self):
        assert Measurement.symmetric(2.0, 0.05).ci_95 == (1.95, 2.05)
        assert Measurement.relative(2.0, 0.10).ci_95 == (1.8, 2.2)

    def test_half_width_and_containment(self):
        m = Measurement.symmetric(4.0, 0.02)
        assert m.half_width == pytest.approx(0.02)
        assert m.relative_half_width == pytest.approx(0.005)
        assert m.contains(3.99) and m.contains(4.02) and not m.contains(4.021)

    def test_zero_width_interval_is_allowed(self):
        # Counted quantities ("1 door") legitimately have no spread.
        m = Measurement(value=1.0, ci_95=(1.0, 1.0), unit=Unit.EACH)
        assert m.half_width == 0.0 and m.contains(1.0)


class TestPlanRoundTrip:
    def test_fixture_plan_round_trips_byte_for_byte(self, results_dir):
        path = results_dir / "cap_lidar_a" / "plan.json"
        original = path.read_text()
        plan = Plan.from_json(original)
        assert plan.to_json() + "\n" == original

    def test_round_trip_preserves_every_measurement(self, results_dir):
        plan = Plan.from_json((results_dir / "cap_lidar_a" / "plan.json").read_text())
        again = Plan.from_json(plan.to_json())
        assert again.model_dump() == plan.model_dump()
        assert again.rooms[0].walls[0].length.ci_95 == plan.rooms[0].walls[0].length.ci_95

    def test_tier_is_an_enum_not_a_string(self, results_dir):
        plan = Plan.from_json((results_dir / "cap_photo_a" / "plan.json").read_text())
        assert plan.tier is Tier.PHOTO
        assert json.loads(plan.to_json())["tier"] == "photo"

    def test_unknown_field_is_rejected(self, results_dir):
        raw = json.loads((results_dir / "cap_lidar_a" / "plan.json").read_text())
        raw["totally_new_key"] = 1
        with pytest.raises(ValidationError):
            Plan.model_validate(raw)


class TestReferentialIntegrity:
    def test_opening_must_sit_on_a_known_wall(self):
        room = _room()
        with pytest.raises(ValidationError, match="unknown wall"):
            Room(
                id=room.id, name=room.name, pose=room.pose, walls=room.walls,
                openings=[Opening(
                    id="op_x", wall_id="not_a_wall", type=OpeningType.DOOR,
                    width=_measurement(0.81), height=_measurement(2.03),
                    offset_along_wall=_measurement(1.0),
                )],
                ceiling_height=room.ceiling_height, floor_area=room.floor_area,
                perimeter=room.perimeter,
            )

    def test_adjacency_needs_two_distinct_rooms(self):
        with pytest.raises(ValidationError, match="distinct"):
            Adjacency(room_a_id="a", room_b_id="a")

    def test_damage_must_reference_a_known_surface(self, results_dir):
        raw = json.loads((results_dir / "cap_lidar_a" / "plan.json").read_text())
        raw["damage"][0]["surface_id"] = "nowhere_surface"
        with pytest.raises(ValidationError, match="unknown surface"):
            Plan.model_validate(raw)

    def test_scope_item_must_reference_known_damage(self, results_dir):
        raw = json.loads((results_dir / "cap_lidar_a" / "plan.json").read_text())
        raw["scope"][0]["damage_region_ids"] = ["dmg_does_not_exist"]
        with pytest.raises(ValidationError, match="unknown damage"):
            Plan.model_validate(raw)

    def test_wall_surface_requires_a_wall_id(self):
        with pytest.raises(ValidationError, match="must reference a wall_id"):
            Surface(
                id="s1", room_id="living_room", kind=SurfaceKind.WALL,
                area=Measurement.relative(9.8, 0.02, Unit.SQUARE_METERS),
            )

    def test_damage_polygon_needs_three_points(self):
        with pytest.raises(ValidationError, match="at least 3 points"):
            DamageRegion(
                id="d", room_id="r", surface_id="s", damage_class=DamageClass.WATER,
                polygon=[Point2D(x=0, y=0), Point2D(x=1, y=0)],
                area=Measurement.relative(1.0, 0.02, Unit.SQUARE_METERS),
                max_extent=_measurement(1.0), severity=0.5, confidence=0.5,
            )
