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
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

log = logging.getLogger("cozmo.io.video")

# Every 15th decoded frame. A typical handheld walkthrough is a slow arc
# around the room at ~30 fps; 15 gives roughly one sampled frame every half
# second, which is enough parallax between views for the reconstruction
# backbone without sampling so densely that most of the budget is spent on
# near-duplicate frames.
DEFAULT_STRIDE_FRAMES = 15

# Laplacian variance below this reads as motion blur or an out-of-focus pan,
# not texture-free wall. Measured against a real 1920x1440 H.264 iPhone
# walkthrough clip (captures/apartment_lidar's own rgb.mp4): sampled frames
# there ranged 1.9-34 (median 6.7, p90 14.1), an order of magnitude below what
# the same test found for uncompressed JPEG stills. Heavy video compression
# smooths exactly the high-frequency detail this metric looks for, so a
# still-photo threshold silently rejects every video frame -- 15.0 sits above
# that clip's median (keeps the sharper half) and below its sharp tail.
# A capture whose frames sit oddly relative to this raises a clear error
# naming the counts involved, rather than silently reconstructing from noise.
DEFAULT_BLUR_THRESHOLD = 15.0

# The reconstruction backbone's own contract is 2-8 views per room (see
# cozmo/pipeline/photo.py); 8 is the top of that band. This is a cap PER ROOM,
# not per video: a walkthrough that crosses three rooms needs three rooms'
# worth of frames sampled before it can be split into them.
DEFAULT_MAX_FRAMES = 8

MIN_FRAMES_REQUIRED = 2

# How many rooms one walkthrough is assumed to be able to cross, and therefore
# how large a pool is sampled before segmentation (DEFAULT_MAX_FRAMES x this).
# Only the frames of the rooms actually found are reconstructed, so a
# single-room video costs nothing extra beyond embedding the larger pool.
VIDEO_MAX_ROOMS = 6

# Consecutive sampled frames whose DINOv2 descriptors agree at least this well
# are the same room; a drop below it is a room boundary. A walkthrough crossing
# a doorway changes almost everything in frame at once, which is exactly what a
# semantic descriptor is good at spotting -- far more reliable than a colour
# histogram, which a lamp being switched on can move as much as a doorway does.
ROOM_CUT_SIMILARITY = 0.55

# A room needs at least MIN_FRAMES_REQUIRED frames to reconstruct at all, so a
# shorter run of frames than this is a glimpse through a doorway (or a turn in
# a corridor), not a room worth claiming. Merged into the neighbour it most
# resembles instead of becoming a room of its own.
MIN_FRAMES_PER_ROOM = 2


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
        # Sharpest-per-time-bucket, NOT globally sharpest. Sharpness is not
        # uniform across a walkthrough: an operator is steadiest before they
        # start moving and blurriest while actually walking, so taking the
        # globally sharpest N collapses the whole selection onto whichever
        # stretch the camera was most stationary. On a real 37 s, 1105-frame
        # walk through three rooms that picked 8 frames from the first 11
        # seconds and discarded the other two rooms entirely -- the
        # reconstruction then had only one room to find, which is exactly the
        # "a multi-room video only ever produces one room" symptom.
        #
        # Splitting the clip into max_frames equal spans and taking each
        # span's sharpest survivor keeps the blur filter's benefit while
        # guaranteeing the whole walk is represented.
        first, last = candidates[0][0], candidates[-1][0]
        span = max(1, last - first + 1)
        buckets: Dict[int, Tuple[int, np.ndarray, float]] = {}
        for candidate in candidates:
            bucket = min(max_frames - 1, int((candidate[0] - first) * max_frames / span))
            if bucket not in buckets or candidate[2] > buckets[bucket][2]:
                buckets[bucket] = candidate
        candidates = [buckets[b] for b in sorted(buckets)]
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


def segment_frames_into_rooms(
    embeddings: Sequence[Sequence[float]],
    cut_similarity: float = ROOM_CUT_SIMILARITY,
    min_frames_per_room: int = MIN_FRAMES_PER_ROOM,
) -> List[List[int]]:
    """Split a walkthrough's frames into per-room runs, by appearance.

    ``embeddings`` are one descriptor per sampled frame, in capture order.
    Returns lists of frame positions -- one list per room found, in order.

    A walkthrough is a sequence, not a set: the frames of one room are
    contiguous in time, and crossing a doorway swaps out nearly everything in
    view at once. So the cut is made where *consecutive* descriptors stop
    agreeing, rather than by clustering frames globally (which happily puts
    two visits to the same room in one group and then cannot say which of the
    two the geometry belongs to).

    Runs shorter than ``min_frames_per_room`` cannot be reconstructed and are
    not rooms; each is merged into whichever neighbouring run it resembles
    more, so a glimpse through a doorway joins the room it was glimpsed from
    instead of becoming a room with one photo of somebody else's bathroom.
    """
    vectors = [np.asarray(e, dtype=float) for e in embeddings]
    if not vectors:
        return []
    if len(vectors) == 1:
        return [[0]]

    unit = [v / max(float(np.linalg.norm(v)), 1e-9) for v in vectors]
    similarity = [float(np.dot(unit[i], unit[i + 1])) for i in range(len(unit) - 1)]

    runs: List[List[int]] = [[0]]
    for position, agreement in enumerate(similarity, start=1):
        if agreement < cut_similarity:
            runs.append([position])
        else:
            runs[-1].append(position)

    # Merge runs too short to reconstruct into their more similar neighbour.
    merged = True
    while merged and len(runs) > 1:
        merged = False
        for index, run in enumerate(runs):
            if len(run) >= min_frames_per_room:
                continue
            before = runs[index - 1] if index > 0 else None
            after = runs[index + 1] if index + 1 < len(runs) else None
            if before is not None and after is not None:
                to_before = float(np.dot(unit[run[0]], unit[before[-1]]))
                to_after = float(np.dot(unit[run[-1]], unit[after[0]]))
                target = index - 1 if to_before >= to_after else index + 1
            else:
                target = index - 1 if before is not None else index + 1
            runs[target] = sorted(runs[target] + run)
            runs.pop(index)
            merged = True
            break

    log.info(
        "walkthrough split into %d room(s) by appearance: %s (consecutive similarity %s)",
        len(runs), [len(r) for r in runs], [round(s, 2) for s in similarity],
    )
    return runs
