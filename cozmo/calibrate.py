"""Fit interval calibration factors from (predicted, ground truth) pairs.

    cozmo calibrate --captures DIR --ground-truth CSV --out calibration/calibration.json

For every capture directory under ``--captures`` (found recursively by its own
``capture.json``, one capture per directory, any tier), runs the real
pipeline at that capture's own tier with every calibration factor forced to
1.0 (see ``run_capture``'s ``ignore_calibration``), then pairs every
measurement in the resulting plan with its ground-truth row exactly the way
``cozmo.benchmark.score.covered_measurements`` does -- reused from there
rather than re-implemented, so the fit and the benchmark's own coverage gate
are always checking the same thing.

For each (tier, quantity kind) group this fits the smallest half-width scale
factor whose scaled interval covers at least ``TARGET_COVERAGE`` of that
group's truth values. This intentionally fits and evaluates on the same set:
a real assignment-scale benchmark has too few captures to hold out a
validation split without leaving single-digit sample counts per group, and a
fitted factor with N=3 is not the same claim as one with N=300 -- every
group's sample count and achieved coverage are printed next to its factor so
nobody mistakes one for the other.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import PIPELINE_VERSION
from .benchmark.score import GroundTruth, covered_measurements, load_ground_truth
from .calibration import CALIBRATION_FILE_VERSION, MAX_SCALE, MIN_SAMPLES_TO_TRUST, MIN_SCALE
from .io.capture import load_capture
from .pipeline.run import run_capture
from .schema import Plan

log = logging.getLogger("cozmo.calibrate")

TARGET_COVERAGE = 0.95


@dataclass(frozen=True)
class Sample:
    tier: str
    kind: str
    element_id: str
    capture_id: str
    predicted: float
    truth: float
    half_width: float

    @property
    def required_scale(self) -> float:
        """The smallest factor that would have made this one sample's
        interval cover the truth: 0 if it already did, +inf if the asserted
        half-width is zero and the prediction was still wrong (no finite
        scale fixes that)."""
        error = abs(self.predicted - self.truth)
        if self.half_width <= 0:
            return 0.0 if error == 0 else float("inf")
        return error / self.half_width


def _plan_samples(plan: Plan, gt: GroundTruth, plan_path: Path) -> List[Sample]:
    """Reuses cozmo.benchmark.score's own (measurement, ground truth) pairing
    -- one definition of "which measurements get checked", shared by the
    coverage gate and the fit."""
    from .benchmark.score import LoadedPlan

    lp = LoadedPlan(path=plan_path, plan=plan)
    tier = plan.tier.value
    return [
        Sample(tier, c["kind"], c["element_id"], plan.capture_id, c["predicted"], c["truth"], c["half_width"])
        for c in covered_measurements(lp, gt)
    ]


def collect_samples(
    captures_dir: Path,
    ground_truth_csv: Path,
    backbone_name: str = "vggt",
    weights_dir: Optional[Path] = None,
    seed: int = 0,
) -> List[Sample]:
    """Run the real pipeline over every capture.json-rooted capture under
    ``captures_dir`` and collect every measurement with matching ground truth.

    Runs with ``ignore_calibration=True``: fitting against an already
    -calibrated prediction would just re-derive the previous factor instead
    of checking it against truth.
    """
    gt = load_ground_truth(ground_truth_csv)
    capture_dirs = sorted({p.parent for p in Path(captures_dir).rglob("capture.json")})
    if not capture_dirs:
        raise FileNotFoundError(f"no capture.json found under {captures_dir}")

    samples: List[Sample] = []
    for capture_dir in capture_dirs:
        tier = load_capture(capture_dir).tier   # fail fast on a malformed capture
        log.info("calibration run: %s (%s)", capture_dir.name, tier.value)
        with tempfile.TemporaryDirectory(prefix="cozmo-calibrate-") as scratch:
            out_dir = Path(scratch)
            result = run_capture(
                capture_dir, out_dir, seed=seed, backbone_name=backbone_name,
                weights_dir=weights_dir, semantics=False, ignore_calibration=True,
            )
            samples.extend(_plan_samples(result.plan, gt, result.plan_path))
    return samples


def fit_factors(samples: Sequence[Sample]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Group samples by (tier, kind); fit the smallest scale factor whose
    scaled interval covers >= TARGET_COVERAGE of that group's truth values.

    The target-coverage quantile of a group's own ``required_scale`` values
    is exactly that smallest factor: scaling every half-width in the group by
    the k-th percentile of ``required_scale`` covers k% of the group by
    construction, and no smaller scale can, since ``required_scale`` is
    monotonic in the width one sample actually needed.
    """
    groups: Dict[Tuple[str, str], List[Sample]] = {}
    for s in samples:
        groups.setdefault((s.tier, s.kind), []).append(s)

    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for (tier, kind), members in sorted(groups.items()):
        required = np.array([m.required_scale for m in members], dtype=float)
        finite = required[np.isfinite(required)]
        if len(finite) < len(required):
            log.warning(
                "%s/%s: %d sample(s) have a zero asserted half-width and a nonzero error -- no "
                "finite scale covers them; excluded from the fit",
                tier, kind, len(required) - len(finite),
            )
        n = len(members)
        if len(finite) == 0:
            scale, coverage = 1.0, 0.0
        else:
            raw_scale = float(np.percentile(finite, TARGET_COVERAGE * 100.0))
            # Calibration only ever widens (floor of 1.0): narrowing an
            # already-generous placeholder on evidence this thin is the
            # "confident garbage" failure mode the project exists to avoid,
            # not a result to ship.
            scale = float(np.clip(max(raw_scale, 1.0), MIN_SCALE, MAX_SCALE))
            coverage = float(np.mean(required <= scale + 1e-9))
        out.setdefault(tier, {})[kind] = {
            "scale": round(scale, 4),
            "n_samples": n,
            "achieved_coverage": round(coverage, 4),
            "trusted": n >= MIN_SAMPLES_TO_TRUST,
        }
    return out


def write_calibration(
    factors: Dict[str, Dict[str, Dict[str, Any]]], out_path: Path, source: Dict[str, Any],
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "calibration_version": CALIBRATION_FILE_VERSION,
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "target_coverage": TARGET_COVERAGE,
        "pipeline_version": PIPELINE_VERSION,
        "source": source,
        "factors": factors,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out_path


def coverage_report(factors: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
    """Per-tier, per-quantity achieved coverage -- the number
    cozmo.benchmark.score.gate_interval_coverage_by_kind reports once the
    file this writes is loaded and a benchmark is re-run."""
    if not factors:
        return "no calibration samples collected"
    headers = ("TIER", "QUANTITY", "N", "SCALE", "COVERAGE", "TRUSTED")
    rows = [
        (tier, kind, str(f["n_samples"]), f"{f['scale']:.3f}",
         f"{f['achieved_coverage'] * 100:.1f}%", "yes" if f["trusted"] else "no")
        for tier in sorted(factors) for kind, f in sorted(factors[tier].items())
    ]
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]

    def line(cells: Sequence[str]) -> str:
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)).rstrip()

    return "\n".join([line(headers), "  ".join("-" * w for w in widths), *[line(r) for r in rows]])


def run_calibration(
    captures_dir: Path,
    ground_truth_csv: Path,
    out_path: Path,
    backbone_name: str = "vggt",
    weights_dir: Optional[Path] = None,
    seed: int = 0,
) -> Tuple[Path, Dict[str, Dict[str, Dict[str, Any]]]]:
    samples = collect_samples(
        captures_dir, ground_truth_csv, backbone_name=backbone_name, weights_dir=weights_dir, seed=seed,
    )
    if not samples:
        raise ValueError(
            f"no (prediction, ground truth) pair found across the captures under {captures_dir}; "
            f"nothing to calibrate"
        )
    factors = fit_factors(samples)
    written = write_calibration(
        factors, out_path,
        source={
            "captures_dir": str(Path(captures_dir).resolve()),
            "ground_truth_csv": str(Path(ground_truth_csv).resolve()),
            "sample_count": len(samples),
            "backbone": backbone_name,
        },
    )
    log.info("wrote %s\n%s", written, coverage_report(factors))
    return written, factors


__all__ = [
    "Sample", "collect_samples", "coverage_report", "fit_factors", "run_calibration",
    "write_calibration",
]
