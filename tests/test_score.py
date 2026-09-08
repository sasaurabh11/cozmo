"""Scorer arithmetic.

These tests build tiny plans in memory so each gate's arithmetic is checked on
its own, including the boundary cases -- a gate that is wrong at its threshold
is worse than no gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import pytest

from cozmo.benchmark.score import (
    FAIL, PASS, SKIP, GroundTruth, GTRow, LoadedPlan, gate_ceiling_height,
    gate_ceiling_spread, gate_footprint, gate_interval_coverage, gate_opening_widths,
    gate_repeatability, gate_wall_lengths, load_ground_truth, render_table, score_results,
)
from cozmo.schema import (
    DriftCorrection, DriftMethod, Measurement, Opening, OpeningType, Plan, Point2D,
    Pose2D, PropertyTotals, QualityReport, Room, ScaleInfo, ScaleSource, Tier, Unit, Wall,
)


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------


def make_plan(
    capture_id: str,
    tier: Tier,
    wall_lengths: Sequence[float],
    ceiling: float = 2.450,
    openings: Sequence[Tuple[str, float]] = (),
    footprint: float = 12.0,
    wall_half: float = 0.02,
    ceiling_half: float = 0.012,
    opening_half: float = 0.015,
    room_id: str = "living_room",
) -> LoadedPlan:
    ceiling_m = Measurement.symmetric(ceiling, ceiling_half)
    walls = [
        Wall(
            id=f"{room_id}_w{i}",
            start=Point2D(x=0.0, y=0.0), end=Point2D(x=length, y=0.0),
            length=Measurement.symmetric(length, wall_half), height=ceiling_m,
            opening_ids=[o[0] for o in openings] if i == 0 else [],
        )
        for i, length in enumerate(wall_lengths)
    ]
    room = Room(
        id=room_id, name=room_id, pose=Pose2D(x=0.0, y=0.0), walls=walls,
        openings=[
            Opening(
                id=opening_id, wall_id=f"{room_id}_w0", type=OpeningType.DOOR,
                width=Measurement.symmetric(width, opening_half),
                height=Measurement.symmetric(2.03, opening_half),
                offset_along_wall=Measurement.symmetric(0.9, 0.03),
            )
            for opening_id, width in openings
        ],
        ceiling_height=ceiling_m,
        floor_area=Measurement.relative(footprint, 0.02, Unit.SQUARE_METERS),
        perimeter=Measurement.relative(sum(wall_lengths), 0.02),
    )
    plan = Plan(
        capture_id=capture_id, tier=tier, pipeline_version="test",
        scale=ScaleInfo(
            source=ScaleSource.LIDAR_DEPTH,
            scale_factor=Measurement(value=1.0, ci_95=(0.99, 1.01), unit=Unit.RATIO),
        ),
        drift_correction=DriftCorrection(enabled=True, method=DriftMethod.POSE_GRAPH),
        property_totals=PropertyTotals(
            room_count=1,
            total_floor_area=Measurement.relative(footprint, 0.02, Unit.SQUARE_METERS),
            footprint_area=Measurement.relative(footprint, 0.02, Unit.SQUARE_METERS),
        ),
        rooms=[room],
        quality=QualityReport(overall_confidence=0.8, interval_method="test"),
    )
    return LoadedPlan(path=Path(f"/tmp/{capture_id}/plan.json"), plan=plan)


def make_gt(rows: Sequence[Tuple[str, str, str, str, float]]) -> GroundTruth:
    return GroundTruth(
        path=Path("memory.csv"),
        rows=[
            GTRow(room=r, element=e, element_id=eid, dimension=d, value_m=v, method="laser")
            for r, e, eid, d, v in rows
        ],
    )


WALL_GT = make_gt([
    ("living_room", "wall", "living_room_w0", "length", 4.000),
    ("living_room", "wall", "living_room_w1", "length", 3.000),
])


# --------------------------------------------------------------------------
# wall lengths
# --------------------------------------------------------------------------


class TestWallLengths:
    def test_all_within_tolerance_passes(self):
        lp = make_plan("c", Tier.LIDAR, [4.010, 2.995])
        result = gate_wall_lengths(lp, WALL_GT)
        assert result.status == PASS
        assert result.value == pytest.approx(1.0)

    def test_lidar_tolerance_is_max_of_2cm_and_1_percent(self):
        # 4.0 m wall: 1% = 4 cm, so a 3.9 cm error is inside the gate ...
        assert gate_wall_lengths(make_plan("c", Tier.LIDAR, [4.039, 3.000]), WALL_GT).status == PASS
        # ... but on the 3.0 m wall 1% = 3 cm, and 3.1 cm is not.
        result = gate_wall_lengths(make_plan("c", Tier.LIDAR, [4.000, 3.031]), WALL_GT)
        assert result.value == pytest.approx(0.5)
        assert result.status == FAIL  # 50% < 85%

    def test_photo_tier_allows_eight_percent(self):
        lp = make_plan("c", Tier.PHOTO, [4.310, 3.230])  # 7.75%, 7.67%
        assert gate_wall_lengths(lp, WALL_GT).status == PASS
        lp = make_plan("c", Tier.PHOTO, [4.330, 3.250])  # 8.25%, 8.33%
        assert gate_wall_lengths(lp, WALL_GT).status == FAIL

    def test_pass_fraction_is_the_gate_not_the_worst_error(self):
        gt = make_gt([("living_room", "wall", f"living_room_w{i}", "length", 4.000) for i in range(8)])
        lengths = [4.005] * 7 + [4.500]  # one gross outlier, 7/8 = 87.5% >= 85%
        result = gate_wall_lengths(make_plan("c", Tier.LIDAR, lengths), gt)
        assert result.status == PASS
        assert result.detail["worst_wall"] == "living_room_w7"

    def test_no_matching_ground_truth_is_skip_not_pass(self):
        gt = make_gt([("other", "wall", "other_room_w0", "length", 4.0)])
        assert gate_wall_lengths(make_plan("c", Tier.LIDAR, [4.0]), gt).status == SKIP


# --------------------------------------------------------------------------
# ceiling height
# --------------------------------------------------------------------------


class TestCeilingHeight:
    GT = make_gt([("living_room", "room", "living_room", "ceiling_height", 2.450)])

    def test_boundary_at_1_5_cm(self):
        assert gate_ceiling_height(make_plan("c", Tier.LIDAR, [4.0], ceiling=2.435), self.GT).status == PASS
        assert gate_ceiling_height(make_plan("c", Tier.LIDAR, [4.0], ceiling=2.434), self.GT).status == FAIL

    def test_every_room_must_pass(self):
        gt = make_gt([
            ("living_room", "room", "living_room", "ceiling_height", 2.450),
            ("bedroom", "room", "bedroom", "ceiling_height", 2.450),
        ])
        lp = make_plan("c", Tier.LIDAR, [4.0], ceiling=2.452)
        bad = make_plan("c2", Tier.LIDAR, [4.0], ceiling=2.500, room_id="bedroom")
        lp.plan.rooms = lp.plan.rooms + bad.plan.rooms
        result = gate_ceiling_height(lp, gt)
        assert result.status == FAIL
        assert result.value == pytest.approx(0.05)


class TestCeilingSpread:
    """The gate that separates repeatable-but-biased from unrepeatable."""

    GT = make_gt([("living_room", "room", "living_room", "ceiling_height", 2.450)])

    def test_spread_within_1cm_passes_even_when_both_are_biased(self):
        a = make_plan("a", Tier.LIDAR, [4.0], ceiling=2.480)
        b = make_plan("b", Tier.LIDAR, [4.0], ceiling=2.487)
        (spread,) = gate_ceiling_spread([a, b])
        assert spread.status == PASS                      # repeatable ...
        assert gate_ceiling_height(a, self.GT).status == FAIL  # ... but biased

    def test_spread_beyond_1cm_fails(self):
        a = make_plan("a", Tier.LIDAR, [4.0], ceiling=2.450)
        b = make_plan("b", Tier.LIDAR, [4.0], ceiling=2.462)
        (spread,) = gate_ceiling_spread([a, b])
        assert spread.status == FAIL
        assert spread.value == pytest.approx(0.012)

    def test_single_capture_is_skip(self):
        (result,) = gate_ceiling_spread([make_plan("a", Tier.LIDAR, [4.0])])
        assert result.status == SKIP


# --------------------------------------------------------------------------
# openings: detection is scored
# --------------------------------------------------------------------------


class TestOpeningWidths:
    GT = make_gt([
        ("living_room", "opening", "op_door", "width", 0.810),
        ("living_room", "opening", "op_window", "width", 1.200),
    ])

    def test_both_found_and_accurate(self):
        lp = make_plan("c", Tier.LIDAR, [4.0], openings=[("op_door", 0.815), ("op_window", 1.190)])
        result = gate_opening_widths(lp, self.GT)
        assert result.status == PASS
        assert result.detail["denominator"] == 2

    def test_missed_opening_counts_as_a_miss(self):
        lp = make_plan("c", Tier.LIDAR, [4.0], openings=[("op_door", 0.810)])
        result = gate_opening_widths(lp, self.GT)
        assert result.detail["missed"] == ["op_window"]
        assert result.detail["denominator"] == 2
        assert result.value == pytest.approx(0.5)
        assert result.status == FAIL

    def test_phantom_opening_counts_as_a_miss(self):
        lp = make_plan("c", Tier.LIDAR, [4.0], openings=[
            ("op_door", 0.810), ("op_window", 1.200), ("op_ghost", 0.900),
        ])
        result = gate_opening_widths(lp, self.GT)
        assert result.detail["phantom"] == ["op_ghost"]
        assert result.detail["denominator"] == 3
        assert result.value == pytest.approx(2 / 3, abs=1e-4)
        assert result.status == FAIL

    def test_a_perfect_but_incomplete_detector_still_fails(self):
        """Reporting only what you are sure of must not buy accuracy."""
        gt = make_gt([("living_room", "opening", f"op_{i}", "width", 0.810) for i in range(10)])
        lp = make_plan("c", Tier.LIDAR, [4.0], openings=[(f"op_{i}", 0.810) for i in range(8)])
        result = gate_opening_widths(lp, gt)
        assert result.value == pytest.approx(0.8)  # 8 exact / 10 in truth
        assert result.status == FAIL

    def test_width_boundary_at_2cm(self):
        gt = make_gt([("living_room", "opening", "op_door", "width", 0.810)])
        assert gate_opening_widths(
            make_plan("c", Tier.LIDAR, [4.0], openings=[("op_door", 0.830)]), gt).status == PASS
        assert gate_opening_widths(
            make_plan("c", Tier.LIDAR, [4.0], openings=[("op_door", 0.831)]), gt).status == FAIL


# --------------------------------------------------------------------------
# footprint, repeatability, calibration
# --------------------------------------------------------------------------


class TestFootprint:
    GT = make_gt([("", "property", "property", "footprint_area", 22.500)])

    def test_photo_tier_eight_percent_band(self):
        assert gate_footprint(make_plan("c", Tier.PHOTO, [4.0], footprint=24.20), self.GT).status == PASS
        result = gate_footprint(make_plan("c", Tier.PHOTO, [4.0], footprint=24.40), self.GT)
        assert result.status == FAIL
        assert result.value == pytest.approx(0.0844, abs=1e-4)

    def test_video_tier_three_percent_band(self):
        assert gate_footprint(make_plan("c", Tier.VIDEO, [4.0], footprint=23.10), self.GT).status == PASS
        assert gate_footprint(make_plan("c", Tier.VIDEO, [4.0], footprint=23.30), self.GT).status == FAIL

    def test_missing_ground_truth_is_skip(self):
        assert gate_footprint(make_plan("c", Tier.LIDAR, [4.0]), WALL_GT).status == SKIP


class TestRepeatability:
    def test_agreement_within_1cm_passes(self):
        a = make_plan("a", Tier.LIDAR, [4.000, 3.000])
        b = make_plan("b", Tier.LIDAR, [4.009, 2.995])
        (result,) = gate_repeatability([a, b])
        assert result.status == PASS
        assert result.scope == "lidar:living_room"

    def test_relative_rule_rescues_long_walls(self):
        # 1.4 cm apart on a 6 m wall is 0.23% -- inside the 0.5% arm.
        a = make_plan("a", Tier.LIDAR, [6.000])
        b = make_plan("b", Tier.LIDAR, [6.014])
        assert gate_repeatability([a, b])[0].status == PASS

    def test_disagreement_fails_and_names_the_wall(self):
        a = make_plan("a", Tier.LIDAR, [4.000, 3.000])
        b = make_plan("b", Tier.LIDAR, [4.000, 3.030])
        (result,) = gate_repeatability([a, b])
        assert result.status == FAIL
        assert result.detail["comparisons"][-1]["wall_id"] == "living_room_w1"
        assert result.value == pytest.approx(0.03)

    def test_different_tiers_are_not_repeats_of_each_other(self):
        a = make_plan("a", Tier.LIDAR, [4.000])
        b = make_plan("b", Tier.PHOTO, [4.300])
        (result,) = gate_repeatability([a, b])
        assert result.status == SKIP


class TestIntervalCoverage:
    GT = make_gt([
        ("living_room", "wall", "living_room_w0", "length", 4.000),
        ("living_room", "wall", "living_room_w1", "length", 3.000),
        ("living_room", "room", "living_room", "ceiling_height", 2.450),
    ])

    def test_honest_intervals_pass(self):
        lp = make_plan("c", Tier.LIDAR, [4.012, 2.990], ceiling=2.455)
        result = gate_interval_coverage(lp, self.GT)
        assert result.status == PASS
        assert result.value == pytest.approx(1.0)

    def test_overconfident_intervals_fail_even_when_accurate(self):
        """Small errors with even smaller intervals is the failure this catches."""
        lp = make_plan("c", Tier.LIDAR, [4.012, 2.990], ceiling=2.455,
                       wall_half=0.002, ceiling_half=0.001)
        result = gate_interval_coverage(lp, self.GT)
        assert result.status == FAIL
        assert result.value == pytest.approx(0.0)

    def test_coverage_counts_every_dimension_with_truth(self):
        lp = make_plan("c", Tier.LIDAR, [4.012, 3.100], ceiling=2.455)
        result = gate_interval_coverage(lp, self.GT)
        assert len(result.detail["checks"]) == 3
        assert result.value == pytest.approx(2 / 3, abs=1e-4)


# --------------------------------------------------------------------------
# ground-truth loading and the full report
# --------------------------------------------------------------------------


class TestGroundTruthLoading:
    def test_reads_the_fixture(self, ground_truth_csv):
        gt = load_ground_truth(ground_truth_csv)
        assert len(gt.rows) == 16
        assert gt.lookup("living_room_w0", "length").value_m == pytest.approx(4.0)
        assert gt.rooms_with_openings == {"living_room", "bedroom"}

    def test_missing_column_is_an_error(self, tmp_path):
        path = tmp_path / "gt.csv"
        path.write_text("room,element,element_id,dimension\nliving_room,wall,w0,length\n")
        with pytest.raises(ValueError, match="missing required column"):
            load_ground_truth(path)

    def test_unknown_element_is_an_error(self, tmp_path):
        path = tmp_path / "gt.csv"
        path.write_text(
            "room,element,element_id,dimension,value_m,method,notes\n"
            "living_room,doorknob,dk1,width,0.05,tape,\n"
        )
        with pytest.raises(ValueError, match="unknown element"):
            load_ground_truth(path)

    def test_non_numeric_value_is_an_error(self, tmp_path):
        path = tmp_path / "gt.csv"
        path.write_text(
            "room,element,element_id,dimension,value_m,method,notes\n"
            "living_room,wall,w0,length,about four metres,laser,\n"
        )
        with pytest.raises(ValueError, match="is not a number"):
            load_ground_truth(path)


class TestFullReport:
    def test_fixture_report_has_both_pass_and_fail(self, results_dir, ground_truth_csv):
        report = score_results(results_dir, ground_truth_csv)
        counts = report.counts
        assert counts["pass"] > 0 and counts["fail"] > 0
        assert report.failed

    def test_every_expected_gate_is_present(self, results_dir, ground_truth_csv):
        report = score_results(results_dir, ground_truth_csv)
        assert {g.gate for g in report.gates} == {
            "wall_lengths", "ceiling_height", "opening_widths", "footprint",
            "interval_coverage", "repeatability", "ceiling_spread",
        }

    def test_known_fixture_verdicts(self, results_dir, ground_truth_csv):
        """The fixture is built to land on specific sides of specific gates."""
        report = score_results(results_dir, ground_truth_csv)
        verdicts = {(g.gate, g.scope): g.status for g in report.gates}
        assert verdicts[("wall_lengths", "cap_lidar_a")] == PASS
        assert verdicts[("opening_widths", "cap_lidar_a")] == PASS
        assert verdicts[("opening_widths", "cap_photo_a")] == FAIL      # missed + phantom
        assert verdicts[("footprint", "cap_photo_a")] == FAIL
        assert verdicts[("interval_coverage", "cap_photo_a")] == FAIL   # confident garbage
        assert verdicts[("repeatability", "lidar:living_room")] == FAIL
        assert verdicts[("repeatability", "lidar:bedroom")] == PASS
        assert verdicts[("ceiling_spread", "lidar:living_room")] == FAIL

    def test_photo_detection_detail_names_the_miss_and_the_phantom(self, results_dir, ground_truth_csv):
        report = score_results(results_dir, ground_truth_csv)
        gate = next(g for g in report.gates
                    if g.gate == "opening_widths" and g.scope == "cap_photo_a")
        assert gate.detail["missed"] == ["op_lr_window"]
        assert gate.detail["phantom"] == ["op_lr_phantom_closet"]

    def test_table_renders_every_gate(self, results_dir, ground_truth_csv):
        table = render_table(score_results(results_dir, ground_truth_csv))
        assert "GATE THRESHOLD" in table
        assert "PASS" in table and "FAIL" in table
        assert table.count("\n") >= len(score_results(results_dir, ground_truth_csv).gates)
