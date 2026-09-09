"""Video-tier frame sampling.

Deliberate simplification, disclosed rather than hidden: there is no separate
video reconstruction pipeline. A handheld walkthrough is decoded at a fixed
stride, frames too blurred to be useful (motion blur, a fast pan) are dropped
by a Laplacian-variance floor, and the sharpest survivors are capped at what
the reconstruction backbone handles well -- then those frames are handed
straight to the existing photo-tier path (cozmo/pipeline/photo.py's
``_reconstruct_room``), unmodified. More views, same code.

The sampling parameters (stride, blur threshold, frame cap) and what actually
happened at each step (frames decoded, frames surviving the stride, frames
surviving the blur filter, frames finally used) are returned as
``VideoSamplingResult`` and recorded in the plan's ``quality.video_sampling``
-- a video capture's plan has to say how its views were chosen, the same way
a photo capture's plan says which EXIF focal length it used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

log = logging.getLogger("cozmo.io.video")

# Every 15th decoded frame. A typical handheld walkthrough is a slow arc
# around the room at ~30 fps; 15 gives roughly one sampled frame every half
# second, which is enough parallax between views for the reconstruction
# backbone without sampling so densely that most of the budget is spent on
# near-duplicate frames.
DEFAULT_STRIDE_FRAMES = 15

# Laplacian variance below this on a sharp 8-bit frame reads as motion blur or
# an out-of-focus pan, not texture-free wall. Chosen empirically against
# clearly-blurred vs. clearly-sharp handheld phone stills; a capture that
# trips this threshold on nearly everything says so in the sampling summary
# rather than silently degrading.
DEFAULT_BLUR_THRESHOLD = 80.0

# The reconstruction backbone's own contract is 2-8 views per room (see
# cozmo/pipeline/photo.py); 8 is the top of that band.
DEFAULT_MAX_FRAMES = 8

MIN_FRAMES_REQUIRED = 2


@dataclass(frozen=True)
class VideoSamplingParams:
    stride_frames: int = DEFAULT_STRIDE_FRAMES
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD
    max_frames: int = DEFAULT_MAX_FRAMES


@dataclass
class VideoSamplingResult:
    """Everything worth recording about how a video became a photo folder."""

    params: VideoSamplingParams
    frame_paths: List[Path]
    source_video: Path
    total_video_frames: int
    frames_at_stride: int        # survived the fixed stride, before the blur filter
    frames_kept_blur: int        # survived the blur filter, before the cap
    frames_dropped_blur: int
    sharpness_scores: List[float]   # Laplacian variance of every frame finally used
    capped: bool                    # True if more frames survived than max_frames allows

    def summary(self) -> Dict[str, Any]:
        return {
            "stride_frames": self.params.stride_frames,
            "blur_threshold": self.params.blur_threshold,
            "max_frames": self.params.max_frames,
            "source_video": str(self.source_video),
            "total_video_frames": self.total_video_frames,
            "frames_at_stride": self.frames_at_stride,
            "frames_kept_after_blur_filter": self.frames_kept_blur,
            "frames_dropped_for_blur": self.frames_dropped_blur,
            "frames_used": len(self.frame_paths),
            "capped_to_max_frames": self.capped,
            "sharpness_scores": [round(s, 1) for s in self.sharpness_scores],
        }


def _laplacian_variance(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def sample_frames(
    video_path: Path,
    out_dir: Path,
    stride_frames: int = DEFAULT_STRIDE_FRAMES,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> VideoSamplingResult:
    """Decode ``video_path`` at a fixed stride, drop frames below
    ``blur_threshold`` Laplacian variance, then cap the sharpest survivors at
    ``max_frames``. Frames are written as JPEGs into ``out_dir`` in
    chronological order, so the result is an ordinary photo folder and
    everything downstream of this function -- EXIF/FOV intrinsics, the
    reconstruction backbone, layout, scale recovery -- is the unmodified
    photo-tier path.
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    candidates: List[Tuple[int, np.ndarray, float]] = []  # (frame_index, image_bgr, sharpness)
    index = 0
    stride_hits = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if index % stride_frames == 0:
                stride_hits += 1
                sharpness = _laplacian_variance(frame_bgr)
                if sharpness >= blur_threshold:
                    candidates.append((index, frame_bgr, sharpness))
            index += 1
    finally:
        cap.release()

    frames_kept_blur = len(candidates)
    frames_dropped_blur = stride_hits - frames_kept_blur

    capped = False
    if len(candidates) > max_frames:
        # Keep the sharpest max_frames, then restore chronological order. A
        # reconstruction backbone wants views spread across the walkthrough,
        # not just its N sharpest instants bunched together in time -- but a
        # cap has to select by *some* signal, and sharpness is the one
        # already computed for the blur filter. When quality is roughly
        # uniform across the walk this is just an even subsample.
        candidates = sorted(candidates, key=lambda c: c[2], reverse=True)[:max_frames]
        candidates.sort(key=lambda c: c[0])
        capped = True

    if len(candidates) < MIN_FRAMES_REQUIRED:
        raise ValueError(
            f"only {len(candidates)} frame(s) survived stride={stride_frames} and "
            f"blur_threshold={blur_threshold} out of {total} decoded video frame(s); need at "
            f"least {MIN_FRAMES_REQUIRED}. Try a smaller --video-stride or a lower "
            f"--video-blur-threshold."
        )

    frame_paths: List[Path] = []
    sharpness_scores: List[float] = []
    for frame_index, image_bgr, sharpness in candidates:
        path = out_dir / f"frame_{frame_index:06d}.jpg"
        cv2.imwrite(str(path), image_bgr)
        frame_paths.append(path)
        sharpness_scores.append(sharpness)

    log.info(
        "%s: %d video frame(s) -> %d at stride %d -> %d after blur filter (>= %.1f) -> %d used%s",
        video_path.name, total, stride_hits, stride_frames, frames_kept_blur, blur_threshold,
        len(frame_paths), " (capped)" if capped else "",
    )

    return VideoSamplingResult(
        params=VideoSamplingParams(stride_frames, blur_threshold, max_frames),
        frame_paths=frame_paths,
        source_video=video_path,
        total_video_frames=total,
        frames_at_stride=stride_hits,
        frames_kept_blur=frames_kept_blur,
        frames_dropped_blur=frames_dropped_blur,
        sharpness_scores=sharpness_scores,
        capped=capped,
    )
