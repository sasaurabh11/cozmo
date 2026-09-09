"""Interval calibration: a checked correction on top of an asserted half-width.

Every Measurement the pipeline builds already carries a half-width reasoned
from something structural -- an occupancy-grid cell size, cross-view plane
agreement, scale-cue disagreement. That reasoning says *why* one wall's
interval should be wider than another's; it does not say whether the result
is actually wide enough. This module is the one place that gets checked: for
each (tier, quantity kind) pair, ``cozmo calibrate`` fits a multiplicative
scale factor against (prediction, ground truth) pairs collected across the
benchmark's captures, and this module applies that factor to every
Measurement of that kind before it reaches a plan.

Load order: the ``COZMO_CALIBRATION_FILE`` env var, else
``DEFAULT_CALIBRATION_PATH``, else nothing on disk -- in which case every
factor is 1.0, which is exactly the pre-calibration behaviour. Calibration is
additive, never a hard dependency: a pipeline with no calibration file yet
still runs, and says so in ``quality.calibration_note``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from .schema import Measurement, Room, SurfaceKind

log = logging.getLogger("cozmo.calibration")

CALIBRATION_FILE_VERSION = "1.0.0"
DEFAULT_CALIBRATION_PATH = Path(
    os.environ.get("COZMO_CALIBRATION_FILE", "calibration/calibration.json")
)

# The quantity kinds cozmo.benchmark.score.covered_measurements pairs against
# ground truth -- calibration is fit and applied against exactly these, no
# more, no fewer, so a fitted factor always means "checked against truth".
QUANTITY_KINDS = ("wall_length", "opening_width", "ceiling_height", "floor_area", "footprint_area")

MIN_SAMPLES_TO_TRUST = 3
# Calibration only ever widens an interval (MIN_SCALE == 1.0): narrowing a
# placeholder half-width on the handful of captures a benchmark set this size
# can supply is exactly the "confident garbage on thin evidence" failure mode
# the project exists to catch, not commit. MAX_SCALE bounds how far one badly
# -fit group (small N, one outlier) can blow up an interval.
MIN_SCALE = 1.0
MAX_SCALE = 6.0


@dataclass(frozen=True)
class QuantityFactor:
    scale: float
    n_samples: int
    achieved_coverage: float
    trusted: bool


_UNCALIBRATED = QuantityFactor(scale=1.0, n_samples=0, achieved_coverage=float("nan"), trusted=False)


@dataclass
class CalibrationSet:
    path: Optional[Path]
    calibration_version: Optional[str]
    fitted_at: Optional[str]
    factors: Dict[str, Dict[str, QuantityFactor]] = field(default_factory=dict)

    def detail(self, tier: str, kind: str) -> QuantityFactor:
        return self.factors.get(tier, {}).get(kind, _UNCALIBRATED)

    def factor(self, tier: str, kind: str) -> float:
        return self.detail(tier, kind).scale

    @property
    def is_loaded(self) -> bool:
        return self.calibration_version is not None


def _empty(path: Optional[Path] = None) -> CalibrationSet:
    return CalibrationSet(path=path, calibration_version=None, fitted_at=None, factors={})


# The identity calibration: every factor 1.0. Used by cozmo.calibrate to
# measure the pipeline's *asserted* half-widths -- fitting against an
# already-calibrated prediction would just re-derive the previous factor
# instead of checking it.
IDENTITY_CALIBRATION = _empty()


def load_calibration(path: Optional[Path] = None) -> CalibrationSet:
    target = Path(path) if path is not None else DEFAULT_CALIBRATION_PATH
    if not target.is_file():
        log.info(
            "no calibration file at %s; every interval factor defaults to 1.0 (uncalibrated)", target
        )
        return _empty(target)
    try:
        raw = json.loads(target.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("calibration file %s unreadable (%s); falling back to uncalibrated", target, exc)
        return _empty(target)

    factors: Dict[str, Dict[str, QuantityFactor]] = {}
    for tier, kinds in raw.get("factors", {}).items():
        factors[tier] = {}
        for kind, entry in kinds.items():
            factors[tier][kind] = QuantityFactor(
                scale=float(entry["scale"]), n_samples=int(entry["n_samples"]),
                achieved_coverage=float(entry["achieved_coverage"]), trusted=bool(entry["trusted"]),
            )
    return CalibrationSet(
        path=target,
        calibration_version=raw.get("calibration_version"),
        fitted_at=raw.get("fitted_at"),
        factors=factors,
    )


# Loaded once at import time -- "the pipeline loads it at runtime" means once
# per process, not once per measurement. A newly fitted calibration file takes
# effect on the next run of the CLI, not the next line of already-running code.
CALIBRATION = load_calibration()


def reload_default() -> CalibrationSet:
    """Re-read DEFAULT_CALIBRATION_PATH (or COZMO_CALIBRATION_FILE) into the
    module-level default. `cozmo calibrate` and tests call this after writing
    a new file so the current process picks it up without a restart."""
    global CALIBRATION
    CALIBRATION = load_calibration()
    return CALIBRATION


def calibration_note_for(tier: str, calibration: Optional[CalibrationSet] = None) -> str:
    """What QualityReport.calibration_note says about this tier's intervals."""
    cal = calibration if calibration is not None else CALIBRATION
    if not cal.is_loaded:
        return (
            "Uncalibrated: no calibration file found (run `cozmo calibrate` against a "
            "benchmark set with ground truth). Every interval factor defaults to 1.0 -- "
            "these are the pipeline's own asserted half-widths, unchecked."
        )
    kinds = cal.factors.get(tier, {})
    trusted = sum(1 for f in kinds.values() if f.trusted)
    return (
        f"Calibrated (calibration v{cal.calibration_version}, fitted {cal.fitted_at}): "
        f"{trusted}/{len(QUANTITY_KINDS)} quantity kind(s) for the {tier} tier have a fitted "
        f"factor backed by >= {MIN_SAMPLES_TO_TRUST} ground-truth samples; the rest default to "
        f"1.0 (asserted, not yet checked)."
    )


# --------------------------------------------------------------------------
# Applying a calibration set to a Measurement / Room
# --------------------------------------------------------------------------


def recalibrate_measurement(
    measurement: Measurement, tier: str, kind: str, calibration: Optional[CalibrationSet] = None,
) -> Measurement:
    """Rescale one Measurement's half-width by its (tier, kind) calibration
    factor, keeping the central value unchanged. This is the one place a
    placeholder interval becomes a calibrated one: the value the pipeline
    reconstructed is never touched, only how wide it admits to being wrong.
    """
    cal = calibration if calibration is not None else CALIBRATION
    factor = cal.factor(tier, kind)
    if factor == 1.0:
        return measurement
    return Measurement.symmetric(measurement.value, measurement.half_width * factor, measurement.unit)


def recalibrate_room(room: Room, tier: str, calibration: Optional[CalibrationSet] = None) -> Room:
    """Apply per-quantity-kind calibration to every Measurement in one Room.

    Wall/opening/ceiling/floor-area kinds match exactly what
    cozmo.benchmark.score.covered_measurements checks against ground truth
    (QUANTITY_KINDS); a wall surface's own area has no ground-truth kind of
    its own and is left as the pipeline asserted it.
    """
    cal = calibration if calibration is not None else CALIBRATION

    walls = [
        w.model_copy(update={
            "length": recalibrate_measurement(w.length, tier, "wall_length", cal),
            "height": recalibrate_measurement(w.height, tier, "ceiling_height", cal),
        })
        for w in room.walls
    ]
    openings = [
        o.model_copy(update={
            "width": recalibrate_measurement(o.width, tier, "opening_width", cal),
            "height": recalibrate_measurement(o.height, tier, "opening_width", cal),
            "offset_along_wall": recalibrate_measurement(o.offset_along_wall, tier, "opening_width", cal),
            "sill_height": (
                recalibrate_measurement(o.sill_height, tier, "opening_width", cal)
                if o.sill_height is not None else None
            ),
        })
        for o in room.openings
    ]
    surfaces = [
        (
            s.model_copy(update={"area": recalibrate_measurement(s.area, tier, "floor_area", cal)})
            if s.kind in (SurfaceKind.FLOOR, SurfaceKind.CEILING) else s
        )
        for s in room.surfaces
    ]
    return room.model_copy(update={
        "walls": walls,
        "openings": openings,
        "surfaces": surfaces,
        "ceiling_height": recalibrate_measurement(room.ceiling_height, tier, "ceiling_height", cal),
        "floor_area": recalibrate_measurement(room.floor_area, tier, "floor_area", cal),
        "perimeter": recalibrate_measurement(room.perimeter, tier, "wall_length", cal),
    })


__all__ = [
    "CALIBRATION", "CALIBRATION_FILE_VERSION", "CalibrationSet", "DEFAULT_CALIBRATION_PATH",
    "IDENTITY_CALIBRATION", "MAX_SCALE", "MIN_SAMPLES_TO_TRUST", "MIN_SCALE", "QUANTITY_KINDS",
    "QuantityFactor", "calibration_note_for", "load_calibration", "recalibrate_measurement",
    "recalibrate_room", "reload_default",
]
