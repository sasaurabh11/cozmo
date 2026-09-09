"""Open-vocabulary detection, and the cross-check against geometry.

Grounding DINO finds things from text prompts; SAM 2 turns its boxes into masks
worth projecting. Neither model is fine-tuned on damage, which is the point of
using an open-vocabulary detector -- and also the reason every detection carries
its score into the output rather than being silently accepted.

Weights are fetched by ``scripts/fetch_weights.sh`` with a pinned revision and a
sha256, and loaded strictly from disk (``local_files_only``): a pipeline that
quietly downloads a different checkpoint mid-benchmark is not reproducible.
When the weights are absent the detector says so and the pipeline continues
without it, because a missing model must degrade the plan, not crash the run.

**Openings are cross-checked, not replaced.** Geometry finds a doorway as a hole
in a wall plane; the detector finds one as a door. They fail differently -- a
mirror or a dark recess fools the geometry, an unobserved wall defeats it
entirely, and a poster of a door fools the detector -- so a detection from
either source is reported, and the plan records which sources agreed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("cozmo.semantics.detect")

DEFAULT_WEIGHTS_DIR = Path(os.environ.get("COZMO_WEIGHTS_DIR", "weights"))
GROUNDING_DINO_DIR = "grounding-dino-tiny"
SAM2_DIR = "sam2.1-hiera-tiny"

OPENING_PROMPTS = ("door", "window", "doorway")
DAMAGE_PROMPTS = ("water stain", "mould", "cracked drywall", "burn mark", "missing drywall")

# Prompt to the damage class in the output contract.
DAMAGE_CLASS_BY_PROMPT: Dict[str, str] = {
    "water stain": "water",
    "mould": "mold",
    "cracked drywall": "crack",
    "burn mark": "fire_smoke",
    "missing drywall": "missing_material",
}

OPENING_KIND_BY_PROMPT: Dict[str, str] = {
    "door": "door",
    "doorway": "door",
    "window": "window",
}

DEFAULT_BOX_THRESHOLD = 0.30
DEFAULT_TEXT_THRESHOLD = 0.25
# Damage is held to a higher bar than openings. An open-vocabulary model asked
# for "cracked drywall" will happily return the wall, and at 0.25 it does.
DAMAGE_BOX_THRESHOLD = 0.35

# A detection covering more of the frame than this is the model describing the
# surface, not a defect on it. Observed directly: "cracked drywall" at 0.26
# returning a mask over 683k of 691k pixels.
MAX_MASK_IMAGE_FRACTION = 0.35
# ... and below this it is a speck, too small to project to a metric extent.
MIN_MASK_PIXELS = 200

# Detection runs on a downscaled frame: Grounding DINO resizes internally
# anyway, and the intrinsics are scaled to match so projection stays correct.
DETECT_LONG_EDGE = 960


class DetectorUnavailable(RuntimeError):
    """Weights are missing or unloadable. The run continues without semantics."""


@dataclass
class Detection:
    label: str
    prompt: str
    score: float
    box_xyxy: Tuple[float, float, float, float]
    frame_index: int
    mask: Optional[np.ndarray] = field(default=None, repr=False)
    mask_source: str = "box"        # "sam2" once refined

    @property
    def damage_class(self) -> Optional[str]:
        return DAMAGE_CLASS_BY_PROMPT.get(self.prompt)

    @property
    def opening_kind(self) -> Optional[str]:
        return OPENING_KIND_BY_PROMPT.get(self.prompt)


def _match_prompt(label: str, prompts: Sequence[str]) -> Optional[str]:
    """Grounding DINO returns text spans; map one back to the prompt it came from."""
    text = label.strip().lower()
    if text in prompts:
        return text
    # The model splits and recombines spans ("water" from "water stain"), so
    # fall back to the longest prompt sharing a word with the returned span.
    candidates = [p for p in prompts if p in text or text in p]
    if candidates:
        return max(candidates, key=len)
    words = set(text.split())
    overlapping = [p for p in prompts if words & set(p.split())]
    return max(overlapping, key=len) if overlapping else None


class OpenVocabularyDetector:
    """Grounding DINO for boxes, SAM 2 for masks. Loaded from local weights only."""

    def __init__(
        self,
        weights_dir: Optional[Path] = None,
        device: Optional[str] = None,
        box_threshold: float = DEFAULT_BOX_THRESHOLD,
        text_threshold: float = DEFAULT_TEXT_THRESHOLD,
        refine_masks: bool = True,
    ) -> None:
        self.weights_dir = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.refine_masks = refine_masks

        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise DetectorUnavailable(f"torch/transformers not installed: {exc}") from exc

        self._torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available() else "cpu")

        dino_dir = self.weights_dir / GROUNDING_DINO_DIR
        if not (dino_dir / "model.safetensors").is_file():
            raise DetectorUnavailable(
                f"Grounding DINO weights not found in {dino_dir}. "
                f"Run scripts/fetch_weights.sh first."
            )
        self.processor = AutoProcessor.from_pretrained(dino_dir, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            dino_dir, local_files_only=True
        ).to(self.device).eval()

        self.sam_processor = None
        self.sam_model = None
        sam_dir = self.weights_dir / SAM2_DIR
        if refine_masks and (sam_dir / "model.safetensors").is_file():
            try:
                from transformers import Sam2Model, Sam2Processor

                self.sam_processor = Sam2Processor.from_pretrained(sam_dir, local_files_only=True)
                self.sam_model = Sam2Model.from_pretrained(
                    sam_dir, local_files_only=True
                ).to(self.device).eval()
            except Exception as exc:  # noqa: BLE001 - refinement is optional
                log.warning("SAM 2 unavailable (%s); masks fall back to detection boxes", exc)
        elif refine_masks:
            log.warning("SAM 2 weights not found in %s; masks fall back to detection boxes", sam_dir)

        # Counted so the manifest can report how much the plausibility filter
        # threw away -- a filter nobody can see the effect of is a filter nobody
        # can argue with.
        self.rejected = 0
        log.info("detector ready on %s (mask refinement: %s)",
                 self.device, "SAM 2" if self.sam_model else "boxes only")

    # -- inference ---------------------------------------------------------
    def detect(
        self,
        image: np.ndarray,
        prompts: Sequence[str],
        frame_index: int = -1,
        box_threshold: Optional[float] = None,
    ) -> List[Detection]:
        """Detect ``prompts`` in one RGB image (H, W, 3), uint8."""
        torch = self._torch
        height, width = image.shape[:2]
        # Grounding DINO wants one lowercase phrase per prompt, period-separated.
        text = ". ".join(p.lower() for p in prompts) + "."

        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.box_threshold if box_threshold is None else box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(height, width)],
        )[0]

        detections: List[Detection] = []
        for score, label, box in zip(results["scores"], results["text_labels"], results["boxes"]):
            prompt = _match_prompt(str(label), list(prompts))
            if prompt is None:
                continue
            x0, y0, x1, y1 = (float(v) for v in box.tolist())
            detections.append(Detection(
                label=str(label), prompt=prompt, score=float(score),
                box_xyxy=(x0, y0, x1, y1), frame_index=frame_index,
            ))

        if detections:
            self._refine(image, detections)
        return self._plausible(detections, image.shape[:2])

    def _plausible(self, detections: List[Detection], shape: Tuple[int, int]) -> List[Detection]:
        """Drop masks too large or too small to be a defect on a surface."""
        pixels = shape[0] * shape[1]
        kept: List[Detection] = []
        for detection in detections:
            if detection.mask is None:
                continue
            covered = int(detection.mask.sum())
            fraction = covered / pixels if pixels else 0.0
            if covered < MIN_MASK_PIXELS:
                log.debug("dropped %s: %d px is too small to measure", detection.prompt, covered)
                continue
            if fraction > MAX_MASK_IMAGE_FRACTION:
                log.debug(
                    "dropped %s: mask covers %.0f%% of the frame, which is the surface, "
                    "not a defect on it", detection.prompt, 100 * fraction,
                )
                continue
            kept.append(detection)
        self.rejected += len(detections) - len(kept)
        return kept

    def _refine(self, image: np.ndarray, detections: List[Detection]) -> None:
        """Replace box masks with SAM 2 masks where possible."""
        for detection in detections:
            detection.mask = _box_mask(image.shape[:2], detection.box_xyxy)
            detection.mask_source = "box"

        if self.sam_model is None:
            return

        torch = self._torch
        boxes = [[list(d.box_xyxy) for d in detections]]
        try:
            inputs = self.sam_processor(
                images=image, input_boxes=boxes, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.sam_model(**inputs, multimask_output=False)
            masks = self.sam_processor.post_process_masks(
                outputs.pred_masks, inputs["original_sizes"]
            )[0]
        except Exception as exc:  # noqa: BLE001 - a failed refinement is not a failed run
            log.warning("SAM 2 refinement failed (%s); keeping box masks", exc)
            return

        array = masks.cpu().numpy() if hasattr(masks, "cpu") else np.asarray(masks)
        for index, detection in enumerate(detections):
            if index >= len(array):
                break
            mask = array[index]
            while mask.ndim > 2:
                mask = mask[0]
            if mask.shape == image.shape[:2] and mask.any():
                detection.mask = mask.astype(bool)
                detection.mask_source = "sam2"


def _box_mask(shape: Tuple[int, int], box: Tuple[float, float, float, float]) -> np.ndarray:
    """Rectangular mask, used when SAM 2 is unavailable or declines."""
    height, width = shape
    x0, y0, x1, y1 = box
    mask = np.zeros((height, width), dtype=bool)
    mask[
        max(0, int(y0)):min(height, int(np.ceil(y1))),
        max(0, int(x0)):min(width, int(np.ceil(x1))),
    ] = True
    return mask


# --------------------------------------------------------------------------
# Cross-check against the geometric opening detections
# --------------------------------------------------------------------------


@dataclass
class OpeningConsensus:
    """One opening, and which detectors found it."""

    wall_index: int
    kind: str
    width_m: float
    height_m: float
    offset_along_wall_m: float
    sill_height_m: float
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    note: str = ""

    @property
    def agreed(self) -> bool:
        return len(self.sources) > 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "wall_index": self.wall_index, "kind": self.kind,
            "width_m": round(self.width_m, 3), "height_m": round(self.height_m, 3),
            "offset_along_wall_m": round(self.offset_along_wall_m, 3),
            "sources": list(self.sources), "confidence": round(self.confidence, 3),
            "note": self.note,
        }


def _overlap_fraction(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    low = max(a[0], b[0])
    high = min(a[1], b[1])
    if high <= low:
        return 0.0
    shorter = min(a[1] - a[0], b[1] - b[0])
    return (high - low) / shorter if shorter > 0 else 0.0


# Detector-only openings are held to the same width bands the geometric
# detector applies. A projected mask that implies a 0.42 m door is a partial
# view of one, and reporting it as an opening costs a phantom on the gate.
SEMANTIC_DOOR_WIDTH_RANGE = (0.55, 1.40)
SEMANTIC_WINDOW_WIDTH_RANGE = (0.35, 2.60)


def cross_check_openings(
    geometric: Sequence[Any],
    semantic: Sequence[Any],
    wall_index_by_id: Optional[Mapping[str, int]] = None,
    overlap_threshold: float = 0.40,
) -> List[OpeningConsensus]:
    """Union the two opening sources, recording which of them fired.

    ``geometric`` are ``cozmo.geometry.openings.OpeningDetection``; ``semantic``
    are projected regions carrying a ``wall_id`` and a bounding box on the wall.
    Where both fire on the same span, geometry supplies the dimensions -- it
    measures them from depth, while the detector only bounds them in pixels --
    and the consensus records both sources.
    """
    wall_index_by_id = wall_index_by_id or {}
    consensus: List[OpeningConsensus] = []
    claimed: set = set()

    for detection in geometric:
        span = (detection.offset_along_wall_m,
                detection.offset_along_wall_m + detection.width_m)
        sources = ["geometry"]
        note = ""
        confidence = float(detection.confidence)

        for index, region in enumerate(semantic):
            if index in claimed:
                continue
            wall_index = wall_index_by_id.get(getattr(region, "wall_id", None) or "", -1)
            if wall_index != detection.wall_index:
                continue
            u0, _v0, u1, _v1 = region.bbox
            if _overlap_fraction(span, (u0, u1)) < overlap_threshold:
                continue
            claimed.add(index)
            sources.append("semantic")
            confidence = min(0.97, confidence + 0.15 * float(region.score))
            semantic_kind = OPENING_KIND_BY_PROMPT.get(region.label, region.label)
            if semantic_kind and semantic_kind != detection.kind:
                note = (
                    f"sources disagree on type: geometry says {detection.kind}, "
                    f"detector says {semantic_kind}; geometry's dimensions kept"
                )
            break

        consensus.append(OpeningConsensus(
            wall_index=detection.wall_index, kind=detection.kind,
            width_m=detection.width_m, height_m=detection.height_m,
            offset_along_wall_m=detection.offset_along_wall_m,
            sill_height_m=detection.sill_height_m,
            sources=sources, confidence=confidence, note=note,
        ))

    # Detections geometry missed entirely. These are the ones that matter: an
    # opening in a wall the depth sensor never saw properly is invisible to the
    # geometric detector, and a missed opening scores as a miss.
    for index, region in enumerate(semantic):
        if index in claimed:
            continue
        wall_index = wall_index_by_id.get(getattr(region, "wall_id", None) or "", -1)
        if wall_index < 0:
            continue
        u0, v0, u1, v1 = region.bbox
        kind = OPENING_KIND_BY_PROMPT.get(region.label, "door")
        width = u1 - u0
        band = SEMANTIC_DOOR_WIDTH_RANGE if kind == "door" else SEMANTIC_WINDOW_WIDTH_RANGE
        if not (band[0] <= width <= band[1]):
            log.debug(
                "detector-only %s of %.2f m is outside the plausible band %s; dropped",
                kind, width, band,
            )
            continue
        consensus.append(OpeningConsensus(
            wall_index=wall_index, kind=kind,
            width_m=u1 - u0, height_m=v1 - v0,
            offset_along_wall_m=u0, sill_height_m=v0,
            sources=["semantic"], confidence=float(region.score) * 0.8,
            note="detector only; geometry did not find a hole here, so the extent "
                 "comes from the projected mask and is looser than a measured one",
        ))

    log.info(
        "openings after cross-check: %d (%d agreed, %d geometry-only, %d detector-only)",
        len(consensus),
        sum(1 for c in consensus if c.agreed),
        sum(1 for c in consensus if c.sources == ["geometry"]),
        sum(1 for c in consensus if c.sources == ["semantic"]),
    )
    return consensus


def detect_damage(
    detector: OpenVocabularyDetector,
    image: np.ndarray,
    frame_index: int = -1,
) -> List[Detection]:
    """Damage prompts at the damage threshold."""
    return detector.detect(image, DAMAGE_PROMPTS, frame_index, box_threshold=DAMAGE_BOX_THRESHOLD)


def detect_openings(
    detector: OpenVocabularyDetector,
    image: np.ndarray,
    frame_index: int = -1,
) -> List[Detection]:
    """Opening prompts at the standard threshold."""
    return detector.detect(image, OPENING_PROMPTS, frame_index)
