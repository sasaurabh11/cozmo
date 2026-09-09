"""The non-geometric half of the contract.

Everything here runs without model weights: the rule engine, the scope
catalogue and the projection maths are all deterministic code, and they are the
parts a reviewer needs to be able to check by hand. Detector behaviour is
exercised separately, and is skipped when the weights are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cozmo.semantics.detect import (
    OPENING_KIND_BY_PROMPT,
    OpeningConsensus,
    _match_prompt,
    cross_check_openings,
)
from cozmo.semantics.project import (
    CELL_M,
    ProjectedRegion,
    SurfacePlane,
    merge_regions,
    project_mask_to_surface,
)
from cozmo.semantics.rules import (
    RuleEngine,
    RuleError,
    load_rules,
)
from cozmo.semantics.scope import load_catalogue, price_regions, quantity_for

import os

WEIGHTS_PRESENT = (Path("weights") / "grounding-dino-tiny" / "model.safetensors").is_file()

# Loading torch in this process is not safe: the geometry tests have already
# imported open3d, and the two bundle different OpenMP runtimes -- together they
# deadlock inside inference at 0% CPU. That is exactly why the pipeline runs
# detection in a child process (cozmo/semantics/worker.py), and it is why these
# tests are opt-in rather than automatic:
#
#     COZMO_TEST_DETECTOR=1 pytest tests/test_semantics.py -k Detector
#
# Everything else in this file -- rules, scope, projection, cross-check -- is
# pure numpy and runs everywhere.
RUN_DETECTOR_TESTS = os.environ.get("COZMO_TEST_DETECTOR") == "1"


# --------------------------------------------------------------------------
# rules
# --------------------------------------------------------------------------


def context(**overrides):
    base = dict(
        region_id="dmg_01", room_id="room_a", surface_id="room_a_w0_surface",
        damage_class="water", surface_kind="wall", area_m2=0.40, max_extent_m=0.90,
        bbox_width_m=0.80, bbox_height_m=0.50, area_ci_95=(0.35, 0.45),
        min_height_above_floor_m=0.60, max_height_above_floor_m=1.10,
        confidence=0.6, severity=0.5, wall_has_opening=False,
        distance_to_exterior_corner_m=None,
    )
    base.update(overrides)
    return base


class TestRuleFile:
    def test_ships_a_usable_rule_set(self):
        rules = load_rules()
        assert len(rules) >= 6
        assert all(rule.text for rule in rules)
        assert all(rule.recommended_action for rule in rules)
        assert len({rule.id for rule in rules}) == len(rules)

    def test_unknown_field_is_rejected_at_load(self, tmp_path):
        """A rule that references a field nobody supplies never fires, silently."""
        path = tmp_path / "rules.yaml"
        path.write_text(
            "version: 1\nrules:\n"
            "  - id: BAD-01\n    text: nonsense\n"
            "    predicate: {field: wetness_vibes, op: gt, value: 1}\n"
        )
        with pytest.raises(RuleError, match="unknown field"):
            load_rules(path)

    def test_unknown_operator_is_rejected(self, tmp_path):
        path = tmp_path / "rules.yaml"
        path.write_text(
            "version: 1\nrules:\n"
            "  - id: BAD-02\n    text: nonsense\n"
            "    predicate: {field: area_m2, op: approximately, value: 1}\n"
        )
        with pytest.raises(RuleError, match="unknown operator"):
            load_rules(path)

    def test_duplicate_ids_are_rejected(self, tmp_path):
        path = tmp_path / "rules.yaml"
        path.write_text(
            "version: 1\nrules:\n"
            "  - id: R1\n    text: a\n    predicate: {field: area_m2, op: gt, value: 0}\n"
            "  - id: R1\n    text: b\n    predicate: {field: area_m2, op: gt, value: 0}\n"
        )
        with pytest.raises(RuleError, match="duplicate rule id"):
            load_rules(path)

    def test_a_rule_must_explain_itself(self, tmp_path):
        path = tmp_path / "rules.yaml"
        path.write_text(
            "version: 1\nrules:\n"
            "  - id: R1\n    predicate: {field: area_m2, op: gt, value: 0}\n"
        )
        with pytest.raises(RuleError, match="explain itself"):
            load_rules(path)


class TestRuleEngine:
    def test_water_at_the_floor_fires_the_subfloor_rule(self):
        firings = RuleEngine().evaluate_region(
            context(damage_class="water", min_height_above_floor_m=0.08)
        )
        assert "CD-WATER-SUBFLOOR-01" in {f.rule.id for f in firings}

    def test_water_high_on_the_wall_does_not(self):
        firings = RuleEngine().evaluate_region(
            context(damage_class="water", min_height_above_floor_m=1.20)
        )
        assert "CD-WATER-SUBFLOOR-01" not in {f.rule.id for f in firings}

    def test_a_firing_carries_the_values_that_caused_it(self):
        """The contract asks for the rule that fired *and* its inputs."""
        firing = next(
            f for f in RuleEngine().evaluate_region(context(min_height_above_floor_m=0.05))
            if f.rule.id == "CD-WATER-SUBFLOOR-01"
        )
        values = firing.triggering_values()
        assert values["min_height_above_floor_m"] == 0.05
        assert values["damage_class"] == "water"
        assert firing.rule.text and firing.rule.recommended_action

    def test_ceiling_mould_near_a_corner_fires_thermal_bridging(self):
        firings = RuleEngine().evaluate_region(context(
            damage_class="mold", surface_kind="ceiling", distance_to_exterior_corner_m=0.30,
        ))
        assert "CD-MOULD-THERMAL-BRIDGE-01" in {f.rule.id for f in firings}

    def test_same_mould_away_from_the_corner_does_not(self):
        firings = RuleEngine().evaluate_region(context(
            damage_class="mold", surface_kind="ceiling", distance_to_exterior_corner_m=1.40,
        ))
        assert "CD-MOULD-THERMAL-BRIDGE-01" not in {f.rule.id for f in firings}

    def test_a_missing_field_does_not_crash_or_fire(self):
        firings = RuleEngine().evaluate_region(context(
            damage_class="mold", surface_kind="ceiling", distance_to_exterior_corner_m=None,
        ))
        assert "CD-MOULD-THERMAL-BRIDGE-01" not in {f.rule.id for f in firings}

    def test_failed_conditions_are_recorded_too(self):
        """A flag has to be arguable, which means showing what did not match."""
        engine = RuleEngine()
        rule = next(r for r in engine.rules if r.id == "CD-WATER-SUBFLOOR-01")
        fired, evaluations = rule.evaluate(context(min_height_above_floor_m=1.5))
        assert not fired
        assert any(not e.passed for e in evaluations)
        assert {e.field for e in evaluations} >= {"damage_class", "min_height_above_floor_m"}


# --------------------------------------------------------------------------
# scope
# --------------------------------------------------------------------------


class TestScope:
    def test_catalogue_loads(self):
        catalogue = load_catalogue()
        assert catalogue.lookup("water", "wall")
        assert "mold" in catalogue.damage_classes

    def test_cut_back_margin_grows_the_quantity(self):
        catalogue = load_catalogue()
        row = next(r for r in catalogue.lookup("water", "wall") if r.basis == "area")
        quantity, _ci, basis = quantity_for(row, context(bbox_width_m=1.0, bbox_height_m=1.0))
        # 1.0 x 1.0 grown by 0.30 each side is 1.6 x 1.6 = 2.56, plus 10% waste.
        assert quantity == pytest.approx(2.56 * 1.10, rel=1e-3)
        assert "cut back" in basis and "waste" in basis

    def test_basis_states_the_arithmetic(self):
        catalogue = load_catalogue()
        row = next(r for r in catalogue.lookup("crack", "wall"))
        _quantity, _ci, basis = quantity_for(row, context(max_extent_m=1.20))
        assert "1.20" in basis and "overrun" in basis

    def test_surface_basis_prices_the_whole_surface(self):
        catalogue = load_catalogue()
        row = next(r for r in catalogue.lookup("water", "wall") if r.basis == "surface")
        quantity, _ci, basis = quantity_for(row, context(), surface_area_m2=8.0)
        assert quantity == pytest.approx(8.0 * 1.05)
        assert "whole surface" in basis

    def test_minimum_charge_is_applied_and_explained(self):
        catalogue = load_catalogue()
        row = next(r for r in catalogue.lookup("water", "wall") if r.basis == "area")
        quantity, _ci, basis = quantity_for(
            row, context(bbox_width_m=0.01, bbox_height_m=0.01, area_m2=0.0001)
        )
        assert quantity >= row.min_quantity
        assert "minimum charge" in basis

    def test_lines_are_keyed_to_the_surface(self):
        lines = price_regions([context()], load_catalogue(), {"room_a_w0_surface": 8.0})
        assert lines
        assert all(line.surface_id == "room_a_w0_surface" for line in lines)
        assert all(line.damage_region_ids == ["dmg_01"] for line in lines)

    def test_an_unpriced_combination_produces_nothing_rather_than_a_guess(self):
        lines = price_regions([context(damage_class="crack", surface_kind="floor")], load_catalogue())
        assert lines == []

    def test_interval_brackets_the_quantity(self):
        for line in price_regions([context()], load_catalogue(), {"room_a_w0_surface": 8.0}):
            assert line.ci_95[0] <= line.quantity <= line.ci_95[1]


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------


def a_wall(width: float = 4.0, height: float = 2.5) -> SurfacePlane:
    """A wall in the z = 2 plane, facing the camera at the origin."""
    return SurfacePlane(
        surface_id="w0_surface", room_id="room_a", kind="wall",
        origin=np.array([-width / 2, 0.0, 2.0]),
        normal=np.array([0.0, 0.0, -1.0]),
        u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 1.0, 0.0]),
        u_extent=width, v_extent=height, area_m2=width * height, wall_id="w0",
    )


class TestProjection:
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    # Phone held at chest height, level. With the camera on the floor plane the
    # optical axis sits at the wall's own origin and half of any mask projects
    # below the floor.
    T = np.array([0.0, 1.2, 0.0])

    def test_a_known_square_projects_to_its_true_area(self):
        """A 100x100 px patch at 2 m through f=500 is 0.40 x 0.40 m = 0.16 m2."""
        mask = np.zeros((480, 640), dtype=bool)
        mask[190:290, 270:370] = True
        region = project_mask_to_surface(
            mask, self.K, np.eye(3), self.T, [a_wall()],
            label="water stain", damage_class="water", score=0.8, frame_index=7,
        )
        assert region is not None
        assert region.area_m2 == pytest.approx(0.16, abs=0.02)
        assert region.surface_id == "w0_surface"
        assert region.frames == [7]

    def test_extent_and_bbox_are_metric(self):
        mask = np.zeros((480, 640), dtype=bool)
        mask[190:290, 270:370] = True
        region = project_mask_to_surface(mask, self.K, np.eye(3), self.T, [a_wall()])
        assert region.bbox_width_m == pytest.approx(0.40, abs=0.03)
        assert region.max_extent_m == pytest.approx(0.57, abs=0.05)

    def test_height_above_floor_comes_from_the_surface_frame(self):
        """Two patches at different image rows must land at different heights.

        The ordering is asserted rather than a number: with ``R`` the identity
        the camera's +y is world +y, so rows *above* the image centre land
        *lower* on the wall. Getting that backwards is exactly the bug this
        catches, and it would put every stain on the wrong part of the wall --
        and with it, every rule that asks how close the damage is to the floor.
        """
        upper = np.zeros((480, 640), dtype=bool)
        upper[100:140, 300:340] = True
        lower = np.zeros((480, 640), dtype=bool)
        lower[340:380, 300:340] = True

        high_row = project_mask_to_surface(upper, self.K, np.eye(3), self.T, [a_wall()])
        low_row = project_mask_to_surface(lower, self.K, np.eye(3), self.T, [a_wall()])
        assert high_row.min_height_above_floor_m < low_row.min_height_above_floor_m
        assert high_row.min_height_above_floor_m == pytest.approx(1.2 - 0.56, abs=0.05)

    def test_height_is_never_negative(self):
        """Cell rounding must not report a region below the floor."""
        region = ProjectedRegion(
            surface_id="s", room_id="r", surface_kind="wall", label="", damage_class="water",
            cells={(0, -3), (1, -2), (2, 5)},
        )
        assert region.min_height_above_floor_m == 0.0

    def test_a_mask_that_misses_every_surface_returns_nothing(self):
        mask = np.zeros((480, 640), dtype=bool)
        mask[190:290, 270:370] = True
        # Camera turned around: the wall is behind it.
        R = np.diag([1.0, 1.0, -1.0])
        assert project_mask_to_surface(mask, self.K, R, self.T, [a_wall()]) is None

    def test_an_empty_mask_returns_nothing(self):
        assert project_mask_to_surface(
            np.zeros((480, 640), dtype=bool), self.K, np.eye(3), self.T, [a_wall()]
        ) is None

    def test_area_interval_brackets_the_area(self):
        mask = np.zeros((480, 640), dtype=bool)
        mask[190:290, 270:370] = True
        region = project_mask_to_surface(mask, self.K, np.eye(3), self.T, [a_wall()])
        low, high = region.area_ci_95()
        assert low <= region.area_m2 <= high


class TestMerging:
    def _region(self, cells, frame, score=0.5):
        return ProjectedRegion(
            surface_id="s1", room_id="r", surface_kind="wall", label="water stain",
            damage_class="water", cells=set(cells), score=score, frames=[frame],
        )

    def test_the_same_stain_from_two_frames_becomes_one_region(self):
        a = self._region({(0, 0), (1, 0), (1, 1)}, frame=1)
        b = self._region({(1, 0), (1, 1), (2, 1)}, frame=2, score=0.8)
        merged = merge_regions([a, b])
        assert len(merged) == 1
        assert merged[0].cells == {(0, 0), (1, 0), (1, 1), (2, 1)}
        assert merged[0].frames == [1, 2]
        assert merged[0].score == 0.8

    def test_area_is_the_union_not_the_sum(self):
        cells = {(i, 0) for i in range(10)}
        merged = merge_regions([self._region(cells, 1), self._region(cells, 2)])
        assert len(merged) == 1
        assert merged[0].area_m2 == pytest.approx(10 * CELL_M * CELL_M)

    def test_separate_stains_stay_separate(self):
        a = self._region({(0, 0), (1, 0)}, frame=1)
        b = self._region({(50, 50), (51, 50)}, frame=2)
        assert len(merge_regions([a, b])) == 2

    def test_different_classes_never_merge(self):
        a = self._region({(0, 0), (1, 0)}, frame=1)
        b = self._region({(0, 0), (1, 0)}, frame=2)
        b.damage_class = "mold"
        assert len(merge_regions([a, b])) == 2


# --------------------------------------------------------------------------
# opening cross-check
# --------------------------------------------------------------------------


class GeometricOpening:
    def __init__(self, wall_index, kind, width, offset, height=2.0, sill=0.0, confidence=0.9):
        self.wall_index, self.kind = wall_index, kind
        self.width_m, self.height_m = width, height
        self.offset_along_wall_m, self.sill_height_m = offset, sill
        self.confidence = confidence


def semantic_opening(wall_id, label, u0, u1, v0=0.0, v1=2.0, score=0.6):
    cells = {(int(u / CELL_M), int(v / CELL_M))
             for u in np.arange(u0, u1, CELL_M) for v in (v0, v1 - CELL_M)}
    return ProjectedRegion(
        surface_id=f"{wall_id}_surface", room_id="room_a", surface_kind="wall",
        label=label, damage_class="", cells=cells, score=score, wall_id=wall_id,
    )


class TestOpeningCrossCheck:
    index_by_id = {"room_a_w0": 0, "room_a_w1": 1}

    def test_both_sources_on_one_opening_are_recorded_as_agreeing(self):
        geometric = [GeometricOpening(0, "door", 0.85, 1.10)]
        semantic = [semantic_opening("room_a_w0", "door", 1.05, 1.95)]
        (result,) = cross_check_openings(geometric, semantic, self.index_by_id)
        assert sorted(result.sources) == ["geometry", "semantic"]
        assert result.agreed
        # Geometry measures from depth, so its dimensions win.
        assert result.width_m == pytest.approx(0.85)

    def test_geometry_alone_still_reports(self):
        (result,) = cross_check_openings([GeometricOpening(0, "door", 0.85, 1.10)], [], self.index_by_id)
        assert result.sources == ["geometry"]
        assert not result.agreed

    def test_detector_alone_still_reports(self):
        """The case that matters: a wall the depth sensor never saw properly."""
        semantic = [semantic_opening("room_a_w1", "door", 0.50, 1.35)]
        (result,) = cross_check_openings([], semantic, self.index_by_id)
        assert result.sources == ["semantic"]
        assert result.wall_index == 1
        assert "detector only" in result.note

    def test_an_implausible_detector_only_width_is_dropped(self):
        """A projected mask implying a 0.30 m door is a partial view, not a door."""
        semantic = [semantic_opening("room_a_w1", "door", 0.50, 0.80)]
        assert cross_check_openings([], semantic, self.index_by_id) == []

    def test_disagreement_on_type_is_recorded_not_hidden(self):
        geometric = [GeometricOpening(0, "door", 0.85, 1.10)]
        semantic = [semantic_opening("room_a_w0", "window", 1.05, 1.95)]
        (result,) = cross_check_openings(geometric, semantic, self.index_by_id)
        assert result.agreed
        assert "disagree" in result.note

    def test_openings_on_different_walls_do_not_pair(self):
        geometric = [GeometricOpening(0, "door", 0.85, 1.10)]
        semantic = [semantic_opening("room_a_w1", "door", 1.05, 1.95)]
        results = cross_check_openings(geometric, semantic, self.index_by_id)
        assert len(results) == 2
        assert {tuple(r.sources) for r in results} == {("geometry",), ("semantic",)}


class TestPromptMapping:
    def test_prompts_map_to_contract_classes(self):
        from cozmo.semantics.detect import DAMAGE_CLASS_BY_PROMPT, DAMAGE_PROMPTS

        assert set(DAMAGE_CLASS_BY_PROMPT) == set(DAMAGE_PROMPTS)
        assert DAMAGE_CLASS_BY_PROMPT["mould"] == "mold"

    def test_doorway_and_door_are_the_same_kind(self):
        assert OPENING_KIND_BY_PROMPT["doorway"] == OPENING_KIND_BY_PROMPT["door"] == "door"

    def test_partial_spans_map_back_to_their_prompt(self):
        """Grounding DINO returns spans like "water" for the prompt "water stain"."""
        prompts = ["water stain", "mould", "cracked drywall"]
        assert _match_prompt("water", prompts) == "water stain"
        assert _match_prompt("cracked drywall", prompts) == "cracked drywall"
        assert _match_prompt("sofa", prompts) is None


@pytest.mark.skipif(
    not (WEIGHTS_PRESENT and RUN_DETECTOR_TESTS),
    reason="needs weights and COZMO_TEST_DETECTOR=1 (torch cannot share this process with open3d)",
)
class TestDetector:
    def test_detector_loads_from_local_weights(self):
        from cozmo.semantics.detect import OpenVocabularyDetector

        detector = OpenVocabularyDetector()
        assert detector.model is not None
        assert detector.device in ("mps", "cuda", "cpu")

    def test_a_blank_frame_yields_nothing_implausible(self):
        from cozmo.semantics.detect import OpenVocabularyDetector, detect_damage

        detector = OpenVocabularyDetector()
        blank = np.full((480, 640, 3), 200, dtype=np.uint8)
        for detection in detect_damage(detector, blank):
            assert detection.mask.sum() / blank[:, :, 0].size <= 0.35
