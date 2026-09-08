"""Regenerate the benchmark fixtures.

    python tests/fixtures/generate.py

The fixtures are committed, so this script is not needed to run the tests; it
exists so the numbers below can be read as intent rather than as magic JSON.
Every value is chosen to land on a specific side of a specific gate -- the point
of the fixture is that `cozmo benchmark` prints both PASS and FAIL rows and each
one is explainable:

    cap_lidar_a   clean LiDAR capture, everything passes
    cap_lidar_b   second capture of the same property: living-room walls wander
                  (repeatability FAIL, ceiling spread FAIL) and one window width
                  is out by 2.1 cm (opening FAIL) while the bedroom repeats fine
    cap_photo_a   photo tier: wall lengths inside +-8%, but a missed window, a
                  phantom closet, an inflated footprint and intervals far too
                  tight -- the "confident garbage on thin input" case
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from cozmo.schema import (
    Adjacency, ConcealedFlag, DamageClass, DamageRegion, DriftCorrection, DriftMethod,
    Measurement, Opening, OpeningType, Plan, Point2D, Pose2D, PropertyTotals, QualityReport,
    Room, ScaleInfo, ScaleSource, ScopeItem, Surface, SurfaceKind, Tier, Unit, Wall,
)

HERE = Path(__file__).resolve().parent
BENCH = HERE / "benchmark"
CAPTURES = HERE / "captures"

GENERATED_AT = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)

# --------------------------------------------------------------------------
# Ground truth: two rooms, laser-measured
# --------------------------------------------------------------------------

GROUND_TRUTH_ROWS: Sequence[Tuple[str, str, str, str, float, str, str]] = (
    ("living_room", "wall", "living_room_w0", "length", 4.000, "laser", "south wall"),
    ("living_room", "wall", "living_room_w1", "length", 3.000, "laser", "east wall"),
    ("living_room", "wall", "living_room_w2", "length", 4.000, "laser", "north wall"),
    ("living_room", "wall", "living_room_w3", "length", 3.000, "laser", "west wall"),
    ("living_room", "room", "living_room", "ceiling_height", 2.450, "laser", ""),
    ("living_room", "room", "living_room", "floor_area", 12.000, "derived", "w0 x w1"),
    ("living_room", "opening", "op_lr_door", "width", 0.810, "tape", "door to bedroom"),
    ("living_room", "opening", "op_lr_window", "width", 1.200, "tape", "south window"),
    ("bedroom", "wall", "bedroom_w0", "length", 3.500, "laser", ""),
    ("bedroom", "wall", "bedroom_w1", "length", 3.000, "laser", ""),
    ("bedroom", "wall", "bedroom_w2", "length", 3.500, "laser", ""),
    ("bedroom", "wall", "bedroom_w3", "length", 3.000, "laser", ""),
    ("bedroom", "room", "bedroom", "ceiling_height", 2.450, "laser", ""),
    ("bedroom", "room", "bedroom", "floor_area", 10.500, "derived", "w0 x w1"),
    ("bedroom", "opening", "op_bed_door", "width", 0.760, "tape", "door to living room"),
    ("", "property", "property", "footprint_area", 22.500, "derived", "sum of room areas"),
)


def write_ground_truth() -> Path:
    BENCH.mkdir(parents=True, exist_ok=True)
    path = BENCH / "ground_truth.csv"
    lines = ["room,element,element_id,dimension,value_m,method,notes"]
    for room, element, element_id, dimension, value, method, notes in GROUND_TRUTH_ROWS:
        lines.append(f"{room},{element},{element_id},{dimension},{value:.3f},{method},{notes}")
    path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------
# Plan construction
# --------------------------------------------------------------------------


class RoomSpec:
    """Measured wall lengths and openings for one room in one fixture plan."""

    def __init__(
        self,
        room_id: str,
        name: str,
        origin: Tuple[float, float],
        wall_lengths: Sequence[float],
        ceiling: float,
        floor_area: float,
        openings: Sequence[Tuple[str, int, OpeningType, float]],
    ) -> None:
        self.room_id = room_id
        self.name = name
        self.origin = origin
        self.wall_lengths = list(wall_lengths)
        self.ceiling = ceiling
        self.floor_area = floor_area
        self.openings = list(openings)


def build_room(spec: RoomSpec, wall_half: float, ceiling_half: float,
               opening_half: float, area_rel: float, with_surfaces: bool) -> Room:
    x0, y0 = spec.origin
    width, depth = spec.wall_lengths[0], spec.wall_lengths[1]
    corners = [(x0, y0), (x0 + width, y0), (x0 + width, y0 + depth), (x0, y0 + depth)]

    ceiling = Measurement.symmetric(spec.ceiling, ceiling_half)
    walls: List[Wall] = []
    surfaces: List[Surface] = []
    for index, length in enumerate(spec.wall_lengths):
        (ax, ay), (bx, by) = corners[index], corners[(index + 1) % 4]
        wall_id = f"{spec.room_id}_w{index}"
        walls.append(Wall(
            id=wall_id,
            start=Point2D(x=round(ax, 3), y=round(ay, 3)),
            end=Point2D(x=round(bx, 3), y=round(by, 3)),
            length=Measurement.symmetric(length, wall_half),
            height=ceiling,
            opening_ids=[o[0] for o in spec.openings if o[1] == index],
        ))
        if with_surfaces:
            surfaces.append(Surface(
                id=f"{wall_id}_surface", room_id=spec.room_id, kind=SurfaceKind.WALL,
                wall_id=wall_id,
                area=Measurement.relative(round(length * spec.ceiling, 3), area_rel, Unit.SQUARE_METERS),
            ))

    openings = [
        Opening(
            id=opening_id,
            wall_id=f"{spec.room_id}_w{wall_index}",
            type=opening_type,
            width=Measurement.symmetric(opening_width, opening_half),
            height=Measurement.symmetric(2.030, opening_half),
            offset_along_wall=Measurement.symmetric(0.900, 0.030),
        )
        for opening_id, wall_index, opening_type, opening_width in spec.openings
    ]

    return Room(
        id=spec.room_id, name=spec.name,
        pose=Pose2D(x=x0, y=y0, theta_rad=0.0),
        walls=walls, openings=openings, surfaces=surfaces,
        ceiling_height=ceiling,
        floor_area=Measurement.relative(spec.floor_area, area_rel, Unit.SQUARE_METERS),
        perimeter=Measurement.relative(round(sum(spec.wall_lengths), 3), area_rel / 2 or 0.01),
    )


def build_plan(
    capture_id: str,
    tier: Tier,
    rooms: Sequence[RoomSpec],
    footprint: float,
    wall_half: float,
    ceiling_half: float,
    opening_half: float,
    area_rel: float,
    confidence: float,
    scale_source: ScaleSource,
    warnings: Sequence[str],
    with_damage: bool = False,
) -> Plan:
    built = [
        build_room(spec, wall_half, ceiling_half, opening_half, area_rel, with_surfaces=True)
        for spec in rooms
    ]
    total = round(sum(r.floor_area.value for r in built), 3)

    damage: List[DamageRegion] = []
    flags: List[ConcealedFlag] = []
    scope: List[ScopeItem] = []
    if with_damage:
        damage = [
            DamageRegion(
                id="dmg_water_01", room_id="living_room", surface_id="living_room_w2_surface",
                damage_class=DamageClass.WATER,
                polygon=[Point2D(x=0.9, y=0.0), Point2D(x=1.7, y=0.0),
                         Point2D(x=1.7, y=0.5), Point2D(x=0.9, y=0.45)],
                area=Measurement.relative(0.38, area_rel, Unit.SQUARE_METERS),
                max_extent=Measurement.symmetric(0.80, 0.03),
                severity=0.55, confidence=0.86, evidence_frames=["frame_000101"],
            ),
            DamageRegion(
                id="dmg_impact_01", room_id="living_room", surface_id="living_room_w1_surface",
                damage_class=DamageClass.IMPACT,
                polygon=[Point2D(x=1.2, y=0.9), Point2D(x=1.4, y=0.9),
                         Point2D(x=1.4, y=1.05), Point2D(x=1.2, y=1.05)],
                area=Measurement.relative(0.03, area_rel, Unit.SQUARE_METERS),
                max_extent=Measurement.symmetric(0.20, 0.02),
                severity=0.35, confidence=0.81, evidence_frames=["frame_000188"],
            ),
        ]
        flags = [ConcealedFlag(
            id="cf_water_01", room_id="living_room", surface_id="living_room_w2_surface",
            rule_id="CD-WATER-01",
            rule_text="Water staining reaching the wall base implies moisture in the cavity behind it.",
            triggered_by_damage_ids=["dmg_water_01"], probability=0.66,
            recommended_action="Moisture-meter the lower 0.6 m of the wall.", inspection_priority=2,
        )]
        scope = [ScopeItem(
            id="scope_001", room_id="living_room", surface_id="living_room_w2_surface",
            damage_region_ids=["dmg_water_01"], code="DRY-RMV-2",
            description="Remove water-damaged drywall, lower course",
            quantity=Measurement.relative(1.60, area_rel, Unit.SQUARE_METERS),
        )]

    return Plan(
        capture_id=capture_id, tier=tier, pipeline_version="0.1.0-fixture",
        generated_at=GENERATED_AT,
        scale=ScaleInfo(
            source=scale_source,
            scale_factor=Measurement(value=1.0, ci_95=(0.99, 1.01), unit=Unit.RATIO),
        ),
        drift_correction=DriftCorrection(
            enabled=True, method=DriftMethod.POSE_GRAPH, loop_closures=1,
            residual_closure_error=Measurement.symmetric(0.028, 0.010),
            ablation_footprint_area=Measurement.relative(
                round(footprint * 1.04, 3), area_rel, Unit.SQUARE_METERS
            ),
        ),
        property_totals=PropertyTotals(
            room_count=len(built),
            total_floor_area=Measurement.relative(total, area_rel, Unit.SQUARE_METERS),
            footprint_area=Measurement.relative(footprint, area_rel, Unit.SQUARE_METERS),
        ),
        rooms=built,
        adjacencies=[Adjacency(room_a_id="living_room", room_b_id="bedroom",
                               via_opening_id="op_lr_door", confidence=0.9)]
        if len(built) > 1 else [],
        damage=damage, concealed_flags=flags, scope=scope,
        quality=QualityReport(
            overall_confidence=confidence,
            interval_method="Fixture: intervals asserted by hand to exercise the calibration row.",
            warnings=list(warnings),
        ),
    )


# --------------------------------------------------------------------------
# The three fixture captures
# --------------------------------------------------------------------------


def cap_lidar_a() -> Plan:
    """Clean capture. Every gate passes."""
    return build_plan(
        capture_id="cap_lidar_a", tier=Tier.LIDAR,
        rooms=[
            RoomSpec("living_room", "Living Room", (0.0, 0.0), [4.008, 2.994, 4.011, 3.003],
                     2.452, 12.05,
                     [("op_lr_door", 1, OpeningType.DOOR, 0.816),
                      ("op_lr_window", 0, OpeningType.WINDOW, 1.209)]),
            RoomSpec("bedroom", "Bedroom", (4.008, 0.0), [3.494, 3.006, 3.512, 2.997],
                     2.452, 10.54,
                     [("op_bed_door", 3, OpeningType.DOOR, 0.767)]),
        ],
        footprint=22.61, wall_half=0.020, ceiling_half=0.012, opening_half=0.015,
        area_rel=0.015, confidence=0.88, scale_source=ScaleSource.LIDAR_DEPTH,
        warnings=[], with_damage=True,
    )


def cap_lidar_b() -> Plan:
    """Second capture of the same property. Living room wanders, bedroom repeats."""
    return build_plan(
        capture_id="cap_lidar_b", tier=Tier.LIDAR,
        rooms=[
            RoomSpec("living_room", "Living Room", (0.0, 0.0), [4.031, 2.981, 4.018, 3.021],
                     2.437, 12.02,
                     [("op_lr_door", 1, OpeningType.DOOR, 0.822),
                      ("op_lr_window", 0, OpeningType.WINDOW, 1.221)]),
            RoomSpec("bedroom", "Bedroom", (4.031, 0.0), [3.499, 3.002, 3.508, 3.001],
                     2.449, 10.51,
                     [("op_bed_door", 3, OpeningType.DOOR, 0.763)]),
        ],
        footprint=22.68, wall_half=0.020, ceiling_half=0.012, opening_half=0.015,
        area_rel=0.015, confidence=0.85, scale_source=ScaleSource.LIDAR_DEPTH,
        warnings=["operator re-walked the living room; two passes merged"],
    )


def cap_photo_a() -> Plan:
    """Photo tier: wall lengths hold, everything that depends on scale does not."""
    return build_plan(
        capture_id="cap_photo_a", tier=Tier.PHOTO,
        rooms=[
            RoomSpec("living_room", "Living Room", (0.0, 0.0), [4.21, 2.83, 3.78, 3.19],
                     2.490, 12.90,
                     [("op_lr_door", 1, OpeningType.DOOR, 0.870),
                      # op_lr_window is missed entirely; this closet was never there.
                      ("op_lr_phantom_closet", 2, OpeningType.CLOSET, 0.640)]),
            RoomSpec("bedroom", "Bedroom", (4.21, 0.0), [3.29, 3.22, 3.71, 2.72],
                     2.540, 12.70,
                     [("op_bed_door", 3, OpeningType.DOOR, 0.710)]),
        ],
        footprint=25.60, wall_half=0.050, ceiling_half=0.040, opening_half=0.030,
        area_rel=0.030, confidence=0.83, scale_source=ScaleSource.STRUCTURAL_PRIOR,
        warnings=[
            "scale recovered from a door-height prior; intervals below are the pipeline's "
            "own claim and the calibration row shows they are too tight"
        ],
    )


def write_plans() -> List[Path]:
    written = []
    for plan in (cap_lidar_a(), cap_lidar_b(), cap_photo_a()):
        directory = BENCH / "results" / plan.capture_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "plan.json"
        path.write_text(plan.to_json() + "\n")
        written.append(path)
    return written


# --------------------------------------------------------------------------
# Capture fixtures for `cozmo run`
# --------------------------------------------------------------------------

# 1x1 JPEG, so the photo fixture contains real images rather than renamed text.
TINY_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0"
    "aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAA"
    "AAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AKp//2Q=="
)


def write_photo_capture() -> Path:
    root = CAPTURES / "demo_photo"
    for room, count in (("living_room", 4), ("bedroom", 3)):
        folder = root / "rooms" / room
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            (folder / f"IMG_{index:04d}.jpg").write_bytes(TINY_JPEG)
    (root / "capture.json").write_text(json.dumps({
        "capture_id": "demo_photo",
        "tier": "photo",
        "space_id": "demo_property",
        "device": {"model": "iPhone 15", "os_version": "18.5", "has_lidar": False},
        "captured_at": "2026-09-01T11:20:00+00:00",
        "operator": "fixture",
        "declared_rooms": ["living_room", "bedroom"],
        "notes": "Photo-tier fixture: 1x1 placeholder images, enough to exercise the reader.",
    }, indent=2) + "\n")
    return root


def main() -> None:
    print("ground truth :", write_ground_truth())
    for path in write_plans():
        print("plan         :", path)
    print("photo capture:", write_photo_capture())


if __name__ == "__main__":
    main()
