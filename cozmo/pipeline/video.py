"""Video tier: sample frames, then hand them to the Phase 4 photo path.

Deliberate simplification, disclosed rather than hidden: there is no separate
video reconstruction pipeline. cozmo.io.video decodes the walkthrough at a
fixed stride, drops frames a Laplacian-variance floor calls too blurred to be
useful, and caps the sharpest survivors at what the reconstruction backbone
handles well (2-8 views, the photo tier's own contract) -- then those frames
are handed straight to :func:`cozmo.pipeline.photo._reconstruct_room`,
unmodified. More views, same code.

Single room only, for the same reason the photo tier started single-room:
one video file is one walkthrough of one room (cozmo.io.capture locates
exactly one video per video-tier capture), and stitching several video
walkthroughs into one property is not attempted here.
"""

from __future__ import annotations

import logging
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..calibration import CalibrationSet
from ..io.capture import CaptureBundle, VideoCapture
from ..io.video import (
    DEFAULT_BLUR_THRESHOLD,
    DEFAULT_MAX_FRAMES,
    DEFAULT_STRIDE_FRAMES,
    sample_frames,
)
from ..schema import Plan, Tier
from .photo import DEFAULT_BACKBONE, _assemble_single_room_plan, _reconstruct_room

log = logging.getLogger("cozmo.pipeline.video")


def build_video_plan(
    bundle: CaptureBundle,
    generated_at: Optional[datetime] = None,
    backbone_name: str = DEFAULT_BACKBONE,
    weights_dir: Optional[Path] = None,
    run_metric_depth_cue: bool = False,
    run_door_cue: bool = False,
    stride_frames: int = DEFAULT_STRIDE_FRAMES,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
    max_frames: int = DEFAULT_MAX_FRAMES,
    frames_out_dir: Optional[Path] = None,
    calibration: Optional[CalibrationSet] = None,
) -> Tuple[Plan, Dict[str, Any]]:
    """Reconstruct one room from a handheld walkthrough video.

    ``frames_out_dir`` is where sampled frames are written; a caller that
    wants to keep them (e.g. for a run's manifest to point at) can pass a
    stable path, otherwise a temporary directory is used and discarded.
    """
    video = bundle.payload
    if not isinstance(video, VideoCapture):
        raise TypeError(f"expected a video capture, got {type(video).__name__}")

    room_id = (bundle.manifest.declared_rooms or ["room_1"])[0]

    mark = time.time()
    if frames_out_dir is not None:
        frames_out_dir = Path(frames_out_dir)
        frames_out_dir.mkdir(parents=True, exist_ok=True)
        sampling = sample_frames(
            video.video, frames_out_dir, stride_frames=stride_frames,
            blur_threshold=blur_threshold, max_frames=max_frames,
        )
        result = _reconstruct_room(
            frames_out_dir, room_id, backbone_name=backbone_name, weights_dir=weights_dir,
            run_metric_depth_cue=run_metric_depth_cue, run_door_cue=run_door_cue,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="cozmo-video-frames-") as scratch:
            scratch_dir = Path(scratch)
            sampling = sample_frames(
                video.video, scratch_dir, stride_frames=stride_frames,
                blur_threshold=blur_threshold, max_frames=max_frames,
            )
            result = _reconstruct_room(
                scratch_dir, room_id, backbone_name=backbone_name, weights_dir=weights_dir,
                run_metric_depth_cue=run_metric_depth_cue, run_door_cue=run_door_cue,
            )
    sample_s = round(time.time() - mark, 3)
    result.timings["video_sample_s"] = sample_s

    extra_warnings = [
        "video tier reuses the photo-tier reconstruction path on sampled frames; no "
        "video-specific reconstruction (temporal smoothing, multi-frame bundle adjustment) runs"
    ]
    extra_degradations = []
    if sampling.frames_dropped_blur > 0:
        extra_degradations.append(
            f"{sampling.frames_dropped_blur} of {sampling.frames_at_stride} sampled frame(s) "
            f"dropped for motion blur (Laplacian variance below {sampling.params.blur_threshold})"
        )
    if len(sampling.frame_paths) <= 3:
        extra_degradations.append(
            f"only {len(sampling.frame_paths)} sharp frame(s) survived sampling; reconstruction "
            f"quality tracks the photo tier's own thin-view floor"
        )

    plan, details = _assemble_single_room_plan(
        bundle, result, Tier.VIDEO, generated_at, backbone_name,
        extra_warnings=extra_warnings,
        extra_degradations=extra_degradations,
        video_sampling=sampling.summary(),
        drift_note_prefix="Video tier (frames sampled from the walkthrough, photo-tier reconstruction). ",
        calibration=calibration,
    )
    return plan, details
