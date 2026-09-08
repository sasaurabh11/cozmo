"""End-to-end CLI behaviour -- the acceptance criteria, as tests."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from cozmo.cli import app

runner = CliRunner()


class TestRunCommand:
    def test_run_on_a_photo_capture(self, photo_capture, tmp_path):
        result = runner.invoke(app, ["run", "--input", str(photo_capture), "--out", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "tier         photo   (from capture.json)" in result.output
        assert (tmp_path / "plan.json").is_file()
        assert (tmp_path / "run_manifest.json").is_file()

    def test_run_on_a_lidar_capture(self, lidar_capture, tmp_path):
        result = runner.invoke(app, ["run", "-i", str(lidar_capture), "-o", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "tier         lidar" in result.output

    def test_drift_correction_off_is_the_ablation_arm(self, lidar_capture, tmp_path):
        result = runner.invoke(app, [
            "run", "-i", str(lidar_capture), "-o", str(tmp_path), "--drift-correction", "off",
        ])
        assert result.exit_code == 0, result.output
        assert "poses_as_is (off)" in result.output
        plan = json.loads((tmp_path / "plan.json").read_text())
        assert plan["drift_correction"]["enabled"] is False

    def test_there_is_no_tier_flag(self, photo_capture, tmp_path):
        result = runner.invoke(app, [
            "run", "-i", str(photo_capture), "-o", str(tmp_path), "--tier", "lidar",
        ])
        assert result.exit_code != 0

    def test_missing_capture_json_exits_cleanly(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        result = runner.invoke(app, ["run", "-i", str(empty), "-o", str(tmp_path / "out")])
        assert result.exit_code == 2
        assert "capture.json" in result.output

    def test_seed_is_recorded_in_the_manifest(self, photo_capture, tmp_path):
        result = runner.invoke(app, [
            "run", "-i", str(photo_capture), "-o", str(tmp_path), "--seed", "77",
        ])
        assert result.exit_code == 0, result.output
        assert json.loads((tmp_path / "run_manifest.json").read_text())["seed"]["seed"] == 77


class TestBenchmarkCommand:
    def _run(self, results_dir, ground_truth_csv, out_dir, *extra):
        return runner.invoke(app, [
            "benchmark", "--results", str(results_dir),
            "--ground-truth", str(ground_truth_csv), "--out", str(out_dir), *extra,
        ])

    def test_prints_a_gate_table_with_pass_and_fail(self, results_dir, ground_truth_csv, tmp_path):
        result = self._run(results_dir, ground_truth_csv, tmp_path)
        assert result.exit_code == 0, result.output
        assert "GATE" in result.output and "STATUS" in result.output
        assert "PASS" in result.output and "FAIL" in result.output
        for gate in ("wall_lengths", "ceiling_height", "opening_widths",
                     "footprint", "repeatability", "interval_coverage", "ceiling_spread"):
            assert gate in result.output

    def test_writes_machine_readable_results(self, results_dir, ground_truth_csv, tmp_path):
        assert self._run(results_dir, ground_truth_csv, tmp_path).exit_code == 0
        payload = json.loads((tmp_path / "results.json").read_text())
        assert payload["summary"]["pass"] > 0 and payload["summary"]["fail"] > 0
        assert len(payload["plans"]) == 3
        assert len(payload["ground_truth"]["sha256"]) == 64
        # Failures carry the detail needed to debug them, not just a verdict.
        opening_gate = next(g for g in payload["gates"]
                            if g["gate"] == "opening_widths" and g["scope"] == "cap_photo_a")
        assert opening_gate["detail"]["missed"] and opening_gate["detail"]["phantom"]

    def test_strict_exits_non_zero_when_a_gate_fails(self, results_dir, ground_truth_csv, tmp_path):
        assert self._run(results_dir, ground_truth_csv, tmp_path, "--strict").exit_code == 1

    def test_accepts_a_single_plan_file(self, results_dir, ground_truth_csv, tmp_path):
        plan = results_dir / "cap_lidar_a" / "plan.json"
        result = self._run(plan, ground_truth_csv, tmp_path)
        assert result.exit_code == 0, result.output
        assert len(json.loads((tmp_path / "results.json").read_text())["plans"]) == 1

    def test_run_output_can_be_scored_directly(self, photo_capture, ground_truth_csv, tmp_path):
        """`cozmo run` then `cozmo benchmark` over its output directory."""
        out = tmp_path / "run"
        assert runner.invoke(app, ["run", "-i", str(photo_capture), "-o", str(out)]).exit_code == 0
        result = self._run(out, ground_truth_csv, tmp_path / "bench")
        assert result.exit_code == 0, result.output
        # The stub property is not the fixture property, so its gates are SKIP:
        # no ground truth, no verdict.
        assert "SKIP" in result.output


class TestVersionCommand:
    def test_reports_all_three_versions(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "cozmo" in result.output and "pipeline" in result.output and "schema" in result.output
