"""Video tier: sample frames, split the walk into rooms, reuse the photo path.

Deliberate simplification, disclosed rather than hidden: there is no separate
video reconstruction pipeline. cozmo.io.video decodes the walkthrough at a
fixed stride, drops frames a Laplacian-variance floor calls too blurred to be
useful, and keeps the sharpest survivor of each equal span of the clip -- then
those frames are handed straight to the photo tier's own code. More views,
same code.

What a video *does* need beyond a photo folder is a decision the photo tier
never has to make: **which room is this frame of?** A photo capture is handed
one folder per room by whoever took it. A 40-second walk through a flat is one
file, and the operator crosses doorways in the middle of it. So the frames are
segmented into per-room runs by appearance (DINOv2 descriptors of consecutive
frames -- see :func:`cozmo.io.video.segment_frames_into_rooms`) and each run is
written into its own folder, at which point the walkthrough *is* a photo
capture and :func:`cozmo.pipeline.photo.build_multi_room_photo_plan` does the
rest -- reconstruction, stitching, adjacency, overlap resolution -- unchanged.

A walk that never leaves one room segments into one run and takes the
single-room path, which is what it was doing before this segmentation existed.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..calibration import CalibrationSet
from ..io.capture import CaptureBundle, VideoCapture
from ..io.video import (
    DEFAULT_BLUR_THRESHOLD,
    DEFAULT_MAX_FRAMES,
    DEFAULT_STRIDE_FRAMES,
    VIDEO_MAX_ROOMS,
    VideoSamplingResult,
    sample_frames,
    segment_frames_into_rooms,
)
from ..schema import Plan, Tier
from .photo import (
    DEFAULT_BACKBONE,
    _assemble_single_room_plan,
    _reconstruct_room,
    build_multi_room_photo_plan,
)

log = logging.getLogger("cozmo.pipeline.video")

EMBED_TIMEOUT_S = 600


def _embed_frames(paths: Sequence[Path], weights_dir: Optional[Path]) -> Tuple[List[List[float]], List[str]]:
    """One DINOv2 descriptor per frame, from a child process.

    Same process boundary as everything else that touches torch here: the
    parent already has open3d loaded, and the two cannot share an address
    space on this platform. Reuses the matcher's own model rather than loading
    a second copy -- see cozmo/stitch/worker.py.

    Returns ``([], [reason])`` when the model is unavailable, which the caller
    treats as "cannot segment" rather than as a failed run.
    """
    with tempfile.TemporaryDirectory(prefix="cozmo-video-embed-") as scratch:
        request_path = Path(scratch) / "request.json"
        response_path = Path(scratch) / "response.json"
        request_path.write_text(json.dumps({
            "embed_frames": [str(p) for p in paths],
            "weights_dir": str(weights_dir) if weights_dir else None,
        }))
        command = [sys.executable, "-m", "cozmo.stitch.worker",
                   str(request_path), str(response_path)]
        log.info("embedding %d sampled frame(s) to split the walk into rooms", len(paths))
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True,
                timeout=EMBED_TIMEOUT_S, check=False,
                cwd=str(Path(__file__).resolve().parents[2]),
            )
        except subprocess.TimeoutExpired:
            return [], [f"frame embedding timed out after {EMBED_TIMEOUT_S}s"]

        if not response_path.is_file():
            return [], [
                f"frame embedding produced no response (exit {completed.returncode}): "
                f"{(completed.stderr or '').strip()[-400:]}"
            ]
        payload = json.loads(response_path.read_text())

    if not payload.get("ok"):
        return [], [f"frame embedding unavailable ({payload.get('error', 'unknown error')})"]
    return list(payload.get("embeddings", [])), []


def _room_names(count: int, declared: Sequence[str]) -> List[str]:
    """Names for the rooms a walk was split into.

    ``declared_rooms`` is used only when it names exactly as many rooms as the
    walk turned out to contain: a walkthrough is captured in the order it was
    walked, which is the order someone naturally lists the rooms in. Any other
    count and the names cannot be matched to segments without guessing, so
    positional names are used and the mismatch is reported.
    """
    if len(declared) == count:
        return list(declared)
    return [f"room_{i + 1}" for i in range(count)]


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
    """Reconstruct a property from a handheld walkthrough video.

    ``max_frames`` is a cap **per room**, not per video: a pool of up to
    ``max_frames * VIDEO_MAX_ROOMS`` frames is sampled across the whole clip,
    segmented into rooms, and each room's run is then capped at ``max_frames``.

    ``frames_out_dir`` is where sampled frames are written; a caller that wants
    to keep them (e.g. to point a manifest at) can pass a stable path,
    otherwise a temporary directory is used and discarded.
    """
    video = bundle.payload
    if not isinstance(video, VideoCapture):
        raise TypeError(f"expected a video capture, got {type(video).__name__}")

    mark = time.time()
    keep_frames = frames_out_dir is not None
    scratch_root = Path(frames_out_dir) if keep_frames else Path(
        tempfile.mkdtemp(prefix="cozmo-video-frames-")
    )
    pool_dir = scratch_root / "pool"
    pool_dir.mkdir(parents=True, exist_ok=True)

    try:
        sampling = sample_frames(
            video.video, pool_dir, stride_frames=stride_frames,
            blur_threshold=blur_threshold, max_frames=max_frames * VIDEO_MAX_ROOMS,
        )
        sample_s = round(time.time() - mark, 3)

        mark = time.time()
        embeddings, embed_warnings = _embed_frames(sampling.frame_paths, weights_dir)
        embed_s = round(time.time() - mark, 3)

        if embeddings:
            runs = segment_frames_into_rooms(embeddings)
        else:
            # Cannot tell one room from another without the descriptors, so
            # the walk is treated as one room -- stated, not assumed silently.
            runs = [list(range(len(sampling.frame_paths)))]

        return _plan_from_runs(
            bundle=bundle, sampling=sampling, runs=runs, scratch_root=scratch_root,
            generated_at=generated_at, backbone_name=backbone_name, weights_dir=weights_dir,
            run_metric_depth_cue=run_metric_depth_cue, run_door_cue=run_door_cue,
            max_frames=max_frames, calibration=calibration,
            embed_warnings=embed_warnings,
            timings={"video_sample_s": sample_s, "video_embed_s": embed_s},
        )
    finally:
        if not keep_frames:
            shutil.rmtree(scratch_root, ignore_errors=True)


def _plan_from_runs(
    bundle: CaptureBundle,
    sampling: VideoSamplingResult,
    runs: Sequence[Sequence[int]],
    scratch_root: Path,
    generated_at: Optional[datetime],
    backbone_name: str,
    weights_dir: Optional[Path],
    run_metric_depth_cue: bool,
    run_door_cue: bool,
    max_frames: int,
    calibration: Optional[CalibrationSet],
    embed_warnings: Sequence[str],
    timings: Dict[str, float],
) -> Tuple[Plan, Dict[str, Any]]:
    """Turn per-room frame runs into a Plan, via the photo tier's own code."""
    sharpness = sampling.sharpness_scores
    paths = sampling.frame_paths
    names = _room_names(len(runs), list(bundle.manifest.declared_rooms or []))

    extra_warnings: List[str] = [
        "video tier reuses the photo-tier reconstruction path on sampled frames; no "
        "video-specific reconstruction (temporal smoothing, multi-frame bundle adjustment) runs"
    ]
    extra_warnings.extend(embed_warnings)
    extra_degradations: List[str] = []
    if sampling.frames_dropped_blur > 0:
        extra_degradations.append(
            f"{sampling.frames_dropped_blur} of {sampling.frames_at_stride} sampled frame(s) "
            f"dropped for motion blur (Laplacian variance below {sampling.params.blur_threshold})"
        )
    if embed_warnings:
        extra_degradations.append(
            "the walkthrough could not be split into rooms (see warnings), so every frame is "
            "treated as one room; a walk that crossed a doorway will have been reconstructed "
            "as a single impossible room"
        )

    declared = list(bundle.manifest.declared_rooms or [])
    if declared and len(declared) != len(runs):
        extra_warnings.append(
            f"capture.json declares {len(declared)} room(s) {declared} but the walkthrough "
            f"splits into {len(runs)} by appearance; using positional names"
        )

    # Each run becomes a room folder, capped at max_frames by keeping the
    # sharpest frames of that run and restoring capture order.
    rooms_dir = scratch_root / "rooms"
    room_frame_map: Dict[str, List[int]] = {}
    for name, run in zip(names, runs):
        chosen = sorted(run, key=lambda p: sharpness[p], reverse=True)[:max_frames]
        chosen.sort()
        folder = rooms_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        for position in chosen:
            shutil.copy2(paths[position], folder / paths[position].name)
        room_frame_map[name] = [int(paths[p].stem.split("_")[-1]) for p in chosen]

    video_sampling = dict(sampling.summary())
    video_sampling.update({
        # `max_frames` is the per-room cap the caller asked for and
        # `frames_used` the frames actually reconstructed; the pool sampled
        # before segmentation is larger and is reported under its own keys,
        # rather than quietly redefining either of those.
        "max_frames": max_frames,
        "frames_used": sum(len(v) for v in room_frame_map.values()),
        "pool_max_frames": sampling.params.max_frames,
        "pool_frames_sampled": len(paths),
        "rooms_found": len(runs),
        "room_frame_indices": room_frame_map,
        "segmented_by": "dinov2_consecutive_similarity" if not embed_warnings else "none",
    })

    single_room = len(runs) == 1
    if single_room:
        room_id = names[0]
        result = _reconstruct_room(
            rooms_dir / room_id, room_id, backbone_name=backbone_name, weights_dir=weights_dir,
            run_metric_depth_cue=run_metric_depth_cue, run_door_cue=run_door_cue,
        )
        result.timings.update(timings)
        plan, details = _assemble_single_room_plan(
            bundle, result, Tier.VIDEO, generated_at, backbone_name,
            extra_warnings=extra_warnings,
            extra_degradations=extra_degradations,
            video_sampling=video_sampling,
            drift_note_prefix=(
                "Video tier (frames sampled from the walkthrough, photo-tier reconstruction). "
            ),
            calibration=calibration,
        )
        details["video_sampling"] = video_sampling
        return plan, details

    # More than one room: the scratch directory now looks exactly like a
    # photo-tier capture, so the multi-room photo path takes it from here.
    log.info("walkthrough covers %d room(s): %s", len(runs), names)
    video_bundle = dataclasses.replace(bundle, root=scratch_root)
    plan, details = build_multi_room_photo_plan(
        video_bundle, generated_at=generated_at, backbone_name=backbone_name,
        weights_dir=weights_dir, run_metric_depth_cue=run_metric_depth_cue,
        run_door_cue=run_door_cue, calibration=calibration, tier=Tier.VIDEO,
    )

    quality = plan.quality.model_copy(update={
        "warnings": list(plan.quality.warnings) + extra_warnings,
        "degradations": list(plan.quality.degradations) + extra_degradations,
        "video_sampling": video_sampling,
    })
    plan = plan.model_copy(update={"quality": quality})
    details["video_sampling"] = video_sampling
    details["timings_s"] = {**details.get("timings_s", {}), **timings}
    return plan, details
