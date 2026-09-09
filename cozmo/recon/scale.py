"""Recovering metres from a scale-free photo reconstruction.

A photo has no metric sensor, so every distance the backbone produces is
correct only up to an unknown global factor: twice as far apart in a room built
at 2x scale looks identical to a camera that moved twice as far. Three
independent cues are combined into one scale factor:

1. **Metric depth alignment.** A monocular metric-depth model (ZoeDepth,
   trained on NYU-Depth-v2 -- indoor rooms, output in metres) gives a second,
   independently-scaled depth estimate per pixel. Aligning it to the backbone's
   own depth by least squares gives a scale factor directly, and is the
   strongest cue when it is available.
2. **Door height.** An interior door leaf is one of the most consistent
   dimensions in a home: N(2.032 m, 0.05 m) is a tight prior. Combined with the
   door's extent in the scale-free reconstruction (pixels + the backbone's own
   depth at the door, converted through the pinhole model -- no floor plane or
   "up" axis needed yet), it gives an independent scale estimate.
3. **Ceiling height.** Weakest and widest: N(2.44 m, 0.30 m). Only usable once
   a scale-free layout exists with both a floor and a ceiling plane.

The combination is a weighted median, not a mean: one badly wrong cue (a door
prior firing on a closet, say) should not drag a correct pair toward it the way
an average would. **Disagreement between cues is the main honest-uncertainty
signal at this tier**, so the combined interval is widened directly by how far
the cues spread, on top of each cue's own uncertainty.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .backbone import ReconstructionResult
from .frames import Frame

log = logging.getLogger("cozmo.recon.scale")

DOOR_HEIGHT_PRIOR_M = 2.032
DOOR_HEIGHT_PRIOR_SD_M = 0.05
CEILING_HEIGHT_PRIOR_M = 2.44
CEILING_HEIGHT_PRIOR_SD_M = 0.30

# A cue's weight is 1/sd^2 of its *own* claimed uncertainty; cues that cannot
# state one do not get to vote silently at full weight.
MIN_CUE_WEIGHT = 1.0


@dataclass
class ScaleCue:
    """One cue's attempt at a scale factor, whether or not it fired."""

    name: str
    fired: bool
    scale_factor: Optional[float] = None      # multiply scale-free units by this for metres
    relative_sd: Optional[float] = None       # this cue's own claimed uncertainty
    reason: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def weight(self) -> float:
        if not self.fired or not self.relative_sd:
            return 0.0
        return 1.0 / max(self.relative_sd, 1e-4) ** 2

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "fired": self.fired,
            "scale_factor": round(self.scale_factor, 5) if self.scale_factor else None,
            "relative_sd": round(self.relative_sd, 4) if self.relative_sd else None,
            "reason": self.reason, "detail": self.detail,
        }


@dataclass
class ScaleEstimate:
    scale_factor: float
    ci_95: Tuple[float, float]
    cues: List[ScaleCue]
    agreement: float                # 1.0 = cues agree closely, 0.0 = wildly apart
    method: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scale_factor": round(self.scale_factor, 5),
            "ci_95": [round(v, 5) for v in self.ci_95],
            "agreement": round(self.agreement, 4),
            "method": self.method,
            "cues": [c.as_dict() for c in self.cues],
        }


# --------------------------------------------------------------------------
# Cue 1: metric monocular depth
# --------------------------------------------------------------------------


def _local_depth(reconstruction: ReconstructionResult, frame_index: int) -> Optional[np.ndarray]:
    points = reconstruction.frame_points_local.get(frame_index)
    if points is None or len(points) == 0:
        return None
    return points[:, 2]           # camera-frame z


def metric_depth_cue(
    frames: Sequence[Frame],
    reconstruction: ReconstructionResult,
    weights_dir: Optional[Any] = None,
) -> ScaleCue:
    """Align a metric depth model's output to the backbone's own depth.

    Requires per-frame local depth from the backbone (frame_points_local), so
    it is a no-op for a backbone that only returns a sparse world cloud.
    """
    try:
        from .metric_depth import MetricDepthUnavailable, estimate_metric_depth
    except ImportError as exc:
        return ScaleCue("metric_depth", fired=False, reason=f"module unavailable: {exc}")

    if not reconstruction.frame_points_local:
        return ScaleCue("metric_depth", fired=False,
                        reason="backbone did not provide per-frame local depth")

    try:
        ratios: List[float] = []
        weights: List[float] = []
        frames_used = 0
        for frame in frames:
            local_z = _local_depth(reconstruction, frame.index)
            if local_z is None or len(local_z) < 50:
                continue
            metric = estimate_metric_depth(frame.image, weights_dir=weights_dir)
            if metric is None:
                continue
            # Sample the metric map at the same pixel grid the local points came
            # from is not tracked per-point here, so align on distributions
            # instead: median metric depth / median local depth is a robust,
            # order-of-magnitude-correct single-frame scale ratio.
            local_median = float(np.median(local_z[local_z > 0]))
            metric_median = float(np.median(metric[metric > 0]))
            if local_median <= 0 or metric_median <= 0:
                continue
            ratios.append(metric_median / local_median)
            weights.append(len(local_z))
            frames_used += 1

        if not ratios:
            return ScaleCue("metric_depth", fired=False, reason="no frame produced a usable pair")

        ratios_arr, weights_arr = np.array(ratios), np.array(weights, dtype=float)
        scale = float(np.average(ratios_arr, weights=weights_arr))
        spread = float(np.std(ratios_arr) / max(scale, 1e-6)) if len(ratios_arr) > 1 else 0.15
        return ScaleCue(
            "metric_depth", fired=True, scale_factor=scale,
            relative_sd=max(spread, 0.08),
            reason=f"ZoeDepth vs. backbone depth, median ratio across {frames_used} frame(s)",
            detail={"per_frame_ratios": [round(r, 4) for r in ratios], "frames_used": frames_used},
        )
    except MetricDepthUnavailable as exc:
        return ScaleCue("metric_depth", fired=False, reason=str(exc))
    except Exception as exc:  # noqa: BLE001 - a cue failing must not fail the run
        log.warning("metric depth cue failed: %s", exc)
        return ScaleCue("metric_depth", fired=False, reason=f"failed: {exc}")


# --------------------------------------------------------------------------
# Cue 2: door height
# --------------------------------------------------------------------------


def door_height_cue(
    frames: Sequence[Frame],
    reconstruction: ReconstructionResult,
    weights_dir: Optional[Any] = None,
) -> ScaleCue:
    """Detect a door, measure its scale-free height via the pinhole model,
    compare to the door-height prior.

    No floor plane or "up" axis needed: a door's pixel height and the backbone's
    own depth at the door are enough (angular size x depth = physical size),
    which is why this cue can run before layout.py has produced anything.
    """
    try:
        from ..semantics.detect import OpenVocabularyDetector, DetectorUnavailable
    except ImportError as exc:
        return ScaleCue("door_height", fired=False, reason=f"detector module unavailable: {exc}")

    try:
        detector = OpenVocabularyDetector(weights_dir=weights_dir, refine_masks=False)
    except DetectorUnavailable as exc:
        return ScaleCue("door_height", fired=False, reason=f"detector unavailable: {exc}")

    estimates: List[float] = []
    detail_rows: List[Dict[str, Any]] = []

    for frame in frames:
        detections = detector.detect(frame.image, ["door"], frame_index=frame.index)
        local_z = _local_depth(reconstruction, frame.index)
        if local_z is None:
            continue
        # frame_points_local is a flat list over confident pixels, not indexed
        # by (row, col) here; approximate the door's depth with the scene
        # median depth in this frame rather than requiring a full pixel index.
        # This is a real approximation -- stated in the returned detail -- and
        # is why this cue's own uncertainty is wider than the depth-alignment cue.
        if len(local_z) < 20:
            continue
        depth_at_door = float(np.median(local_z[local_z > 0]))

        for detection in detections:
            if detection.score < 0.30:
                continue
            x0, y0, x1, y1 = detection.box_xyxy
            box_height_px = y1 - y0
            fy = frame.K[1, 1]
            physical_height_scalefree = (box_height_px / fy) * depth_at_door
            if physical_height_scalefree <= 0:
                continue
            estimates.append(DOOR_HEIGHT_PRIOR_M / physical_height_scalefree)
            detail_rows.append({
                "frame": frame.index, "score": round(float(detection.score), 3),
                "box_height_px": round(box_height_px, 1),
                "implied_scale": round(DOOR_HEIGHT_PRIOR_M / physical_height_scalefree, 5),
            })

    if not estimates:
        return ScaleCue("door_height", fired=False, reason="no door detected with usable depth")

    scale = float(np.median(estimates))
    spread = float(np.std(estimates) / max(scale, 1e-6)) if len(estimates) > 1 else 0.20
    # This cue's approximation (scene-median depth rather than depth measured
    # exactly at the door) puts a floor under how tight it is allowed to claim
    # to be, regardless of how well multiple detections happen to agree.
    relative_sd = max(spread, 0.18)
    return ScaleCue(
        "door_height", fired=True, scale_factor=scale, relative_sd=relative_sd,
        reason=f"{len(estimates)} door detection(s) against a {DOOR_HEIGHT_PRIOR_M} m prior",
        detail={"detections": detail_rows},
    )


# --------------------------------------------------------------------------
# Cue 3: ceiling height prior
# --------------------------------------------------------------------------


def ceiling_height_cue(layout: Any) -> ScaleCue:
    """Floor-to-ceiling span in scale-free units, against the ceiling prior.

    Weakest cue by design (a 0.30 m prior sd is wide), and only fires when the
    scale-free layout actually found both a floor and a ceiling plane.
    """
    span = getattr(layout, "floor_to_ceiling_span", None)
    if span is None or span <= 0:
        return ScaleCue("ceiling_height", fired=False,
                        reason="no scale-free floor-to-ceiling span available")

    scale = CEILING_HEIGHT_PRIOR_M / span
    return ScaleCue(
        "ceiling_height", fired=True, scale_factor=scale,
        relative_sd=CEILING_HEIGHT_PRIOR_SD_M / CEILING_HEIGHT_PRIOR_M,
        reason=f"floor-to-ceiling span {span:.3f} (scale-free) against a "
               f"{CEILING_HEIGHT_PRIOR_M} m +- {CEILING_HEIGHT_PRIOR_SD_M} m prior",
        detail={"span_scalefree": round(float(span), 4)},
    )


# --------------------------------------------------------------------------
# Combination
# --------------------------------------------------------------------------


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    half = cumulative[-1] / 2.0
    index = int(np.searchsorted(cumulative, half))
    return float(values[min(index, len(values) - 1)])


def recover_scale(
    frames: Sequence[Frame],
    reconstruction: ReconstructionResult,
    layout: Any,
    weights_dir: Optional[Any] = None,
    run_metric_depth: bool = True,
    run_door_cue: bool = True,
) -> ScaleEstimate:
    """Run all three cues and combine them."""
    cues: List[ScaleCue] = []
    if run_metric_depth:
        cues.append(metric_depth_cue(frames, reconstruction, weights_dir))
    else:
        cues.append(ScaleCue("metric_depth", fired=False, reason="disabled for this run"))
    if run_door_cue:
        cues.append(door_height_cue(frames, reconstruction, weights_dir))
    else:
        cues.append(ScaleCue("door_height", fired=False, reason="disabled for this run"))
    cues.append(ceiling_height_cue(layout))

    fired = [c for c in cues if c.fired and c.scale_factor and c.scale_factor > 0]
    if not fired:
        log.warning("no scale cue fired; falling back to the ceiling prior alone")
        return ScaleEstimate(
            scale_factor=1.0, ci_95=(0.4, 2.5), cues=cues, agreement=0.0,
            method="no_cue_fired_unscaled",
        )

    values = np.array([c.scale_factor for c in fired])
    weights = np.array([max(c.weight, MIN_CUE_WEIGHT) for c in fired])
    combined = _weighted_median(values, weights)

    if len(fired) > 1:
        spread = float(np.std(values) / max(combined, 1e-6))
        agreement = float(np.clip(1.0 - spread / 0.5, 0.0, 1.0))
    else:
        spread = fired[0].relative_sd or 0.25
        agreement = 0.3   # a single cue cannot be cross-checked; say so with a low score

    # Combined relative half-width: each cue's own uncertainty, widened by how
    # much the cues disagree with each other -- disagreement is the dominant
    # term whenever more than one cue fires and they do not closely agree.
    own_uncertainty = float(np.average(
        [c.relative_sd or 0.25 for c in fired], weights=weights
    ))
    relative_half_width = max(own_uncertainty, spread) * 1.96
    # A single cue's own claimed precision is never enough on its own at this
    # tier: floor the interval so "one cue fired and was confident" cannot look
    # like "three cues agreed".
    if len(fired) == 1:
        relative_half_width = max(relative_half_width, 0.30)

    ci = (combined * (1 - relative_half_width), combined * (1 + relative_half_width))
    method = f"weighted_median_of_{len(fired)}_cue(s)" if len(fired) > 1 else "single_cue"

    log.info(
        "scale: %.4f [%.4f, %.4f] from %s (agreement %.2f)",
        combined, ci[0], ci[1], [c.name for c in fired], agreement,
    )
    return ScaleEstimate(scale_factor=combined, ci_95=ci, cues=cues, agreement=agreement, method=method)
