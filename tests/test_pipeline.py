"""Pipeline orchestration: determinism, provenance, tier dispatch.

The reconstruction is stubbed, so what is worth testing today is everything
around it -- the parts that have to be true before any number can be believed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cozmo.io.capture import load_capture
from cozmo.pipeline.run import (
    MANIFEST_FILENAME, PLAN_FILENAME, build_stub_plan, hash_directory, run_capture,
)

# The photo tier now does real reconstruction (VGGT). Tests in this file that
# only care about CLI/manifest/dispatch plumbing -- not reconstruction
# accuracy, which is tested separately in test_recon.py -- use the stub
# backbone so they stay fast and do not need model weights.
PHOTO_STUB_KW = {"backbone_name": "stub"}
from cozmo.schema import DriftMethod, Plan, Tier
from cozmo.seed import DEFAULT_SEED, set_global_seeds


class TestCaptureDispatch:
    def test_tier_comes_from_capture_json(self, photo_capture, lidar_capture):
        assert load_capture(photo_capture).tier is Tier.PHOTO
        assert load_capture(lidar_capture).tier is Tier.LIDAR

    def test_photo_rooms_are_discovered_from_folders(self, photo_capture):
        bundle = load_capture(photo_capture)
        assert bundle.room_names() == ["bedroom", "living_room"]
        assert bundle.summary()["payload"]["image_count"] == 7

    def test_photo_tier_reconstructs_with_the_stub_backbone(self, photo_capture, tmp_path):
        """CLI/manifest plumbing only -- see test_recon.py for real accuracy."""
        result = run_capture(photo_capture, tmp_path / "out", **PHOTO_STUB_KW)
        assert result.plan.tier is Tier.PHOTO
        assert len(result.plan.rooms) == 1
        assert result.plan.rooms[0].walls

    def test_lidar_capture_exposes_frames_and_intrinsics(self, lidar_capture):
        bundle = load_capture(lidar_capture)
        summary = bundle.summary()["payload"]
        assert summary["frame_count"] == 16
        assert summary["depth_frames"] == 16
        assert summary["depth_resolution"] == [128, 96]
        # Intrinsics must come out scaled to the depth stream, not the RGB one.
        assert summary["depth_intrinsics_fx"] == pytest.approx(92.0, abs=0.5)

    def test_ingest_reports_how_much_was_seen_above_camera_height(self, lidar_capture):
        """The ingest ceiling check is a cheap proxy, and it is only a proxy.

        It counts points above camera height, which tall walls also supply, so a
        capture with no ceiling can still pass it. The real decision is made in
        planes.py against the plane itself -- see the ceiling-method tests.
        """
        summary = load_capture(lidar_capture).summary()["payload"]
        assert 0.0 <= summary["fraction_above_camera_height"] <= 1.0
        assert "ceiling_likely_captured" in summary

    def test_missing_capture_json_is_a_clear_error(self, tmp_path):
        (tmp_path / "rooms").mkdir()
        with pytest.raises(FileNotFoundError, match="capture.json"):
            load_capture(tmp_path)

    def test_unknown_tier_is_rejected(self, tmp_path):
        (tmp_path / "capture.json").write_text(json.dumps({"capture_id": "x", "tier": "thermal"}))
        with pytest.raises(Exception, match="tier"):
            load_capture(tmp_path)


class TestRunOutputs:
    def test_writes_plan_and_manifest(self, photo_capture, tmp_path):
        result = run_capture(photo_capture, tmp_path / "out", **PHOTO_STUB_KW)
        assert (tmp_path / "out" / PLAN_FILENAME).is_file()
        assert (tmp_path / "out" / MANIFEST_FILENAME).is_file()
        assert Plan.from_json(result.plan_path.read_text()).capture_id == "demo_photo"

    def test_manifest_records_the_provenance_the_brief_asks_for(self, photo_capture, tmp_path):
        result = run_capture(
            photo_capture, tmp_path / "out", command=["cozmo", "run", "--input", "x"], **PHOTO_STUB_KW
        )
        manifest = json.loads(result.manifest_path.read_text())
        assert set(manifest["git"]) == {"commit", "branch", "describe", "dirty"}
        assert manifest["command"] == "cozmo run --input x"
        assert manifest["seed"]["seed"] == DEFAULT_SEED
        assert len(manifest["input"]["sha256"]) == 64
        assert manifest["outputs"][PLAN_FILENAME] == manifest["outputs"][PLAN_FILENAME]
        assert manifest["pipeline_version"] and manifest["schema_version"]

    def test_photo_tier_calibration_is_honest_about_being_uncalibrated(self, photo_capture, tmp_path):
        """The photo tier is real reconstruction now, not a stub -- but it must
        still say plainly that its intervals are not yet checked against
        ground truth (see cozmo/pipeline/photo.py)."""
        result = run_capture(photo_capture, tmp_path / "out", **PHOTO_STUB_KW)
        assert "Uncalibrated" in result.plan.quality.calibration_note
        assert result.plan.quality.semantics_available is False

    def test_stub_tier_plans_admit_they_are_stubs(self):
        """The video tier is still the hardcoded stub; build_stub_plan is its
        seam and is unit-tested directly here (video has no fixture capture
        of its own yet)."""
        from cozmo.io.capture import CaptureBundle, CaptureManifest
        from cozmo.schema import Tier as _Tier

        bundle = CaptureBundle(
            root=Path("."),
            manifest=CaptureManifest(capture_id="video_stub", tier=_Tier.VIDEO),
            payload=None, warnings=[],
        )
        plan = build_stub_plan(bundle)
        assert any("STUB PIPELINE" in w for w in plan.quality.warnings)

    def test_lidar_drift_flag_reaches_the_plan_and_the_ablation(self, lidar_capture, tmp_path):
        on = run_capture(lidar_capture, tmp_path / "on", drift_correction=True).plan
        off = run_capture(lidar_capture, tmp_path / "off", drift_correction=False).plan

        assert on.drift_correction.enabled is True
        assert on.drift_correction.method is DriftMethod.MANHATTAN_SNAP
        assert off.drift_correction.method is DriftMethod.NONE_POSES_AS_IS
        # The ablation arm reports the other arm's footprint, so the two runs
        # can be diffed on the number the drift gate is about.
        assert off.property_totals.footprint_area.value == pytest.approx(
            on.drift_correction.ablation_footprint_area.value
        )
        assert off.property_totals.footprint_area.value > on.property_totals.footprint_area.value


class TestDeterminism:
    def test_two_runs_produce_identical_plans(self, photo_capture, tmp_path, monkeypatch):
        monkeypatch.setenv("SOURCE_DATE_EPOCH", "1756728000")
        first = run_capture(photo_capture, tmp_path / "a", **PHOTO_STUB_KW).plan_path.read_text()
        second = run_capture(photo_capture, tmp_path / "b", **PHOTO_STUB_KW).plan_path.read_text()
        assert first == second

    def test_same_room_gets_the_same_geometry_across_captures(self, photo_capture, tmp_path):
        """Repeatability starts here: identical input, identical output."""
        a = run_capture(photo_capture, tmp_path / "a", **PHOTO_STUB_KW).plan
        b = run_capture(photo_capture, tmp_path / "b", **PHOTO_STUB_KW).plan
        assert [w.length.value for w in a.rooms[0].walls] == [w.length.value for w in b.rooms[0].walls]

    def test_same_room_gets_the_same_geometry_across_lidar_runs(self, lidar_capture, tmp_path):
        """The repeatability gate starts here: identical input, identical walls."""
        a = run_capture(lidar_capture, tmp_path / "a").plan
        b = run_capture(lidar_capture, tmp_path / "b").plan
        assert [round(w.length.value, 6) for w in a.rooms[0].walls] == \
               [round(w.length.value, 6) for w in b.rooms[0].walls]

    def test_seed_record_lists_what_was_seeded(self):
        record = set_global_seeds(1234)
        assert record["seed"] == 1234
        assert "random" in record["seeded"] and "numpy" in record["seeded"]
        assert "torch" in record["seeded"] or "torch" in record["unavailable"]


class TestInputHash:
    def test_hash_is_stable_and_covers_content(self, tmp_path):
        root = tmp_path / "cap"
        (root / "rooms").mkdir(parents=True)
        (root / "rooms" / "a.jpg").write_bytes(b"one")
        before = hash_directory(root)
        assert hash_directory(root)["sha256"] == before["sha256"]

        (root / "rooms" / "a.jpg").write_bytes(b"two")
        assert hash_directory(root)["sha256"] != before["sha256"]

    def test_hash_covers_file_names_not_just_bytes(self, tmp_path):
        root = tmp_path / "cap"
        root.mkdir()
        (root / "a.jpg").write_bytes(b"same")
        first = hash_directory(root)["sha256"]
        (root / "a.jpg").rename(root / "b.jpg")
        assert hash_directory(root)["sha256"] != first

    def test_counts_files_and_bytes(self, photo_capture):
        summary = hash_directory(photo_capture)
        assert summary["file_count"] == 8  # 7 images + capture.json
        assert summary["total_bytes"] > 0
