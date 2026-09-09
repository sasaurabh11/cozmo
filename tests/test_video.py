"""Video tier: frame sampling in isolation, then the whole tier end to end.

The video tier is deliberately not a separate reconstruction pipeline (see
cozmo/io/video.py and cozmo/pipeline/video.py): frames are sampled from a
walkthrough clip and handed to the existing photo-tier path. These tests
cover both halves -- the sampler's own stride/blur/cap behaviour, and the
whole tier running through `run_capture` with the stub backbone so it stays
fast and needs no model weights (see test_recon.py for real reconstruction
accuracy, exercised separately and opt-in).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import pytest

from cozmo.io.video import sample_frames
from cozmo.pipeline.run import run_capture
from cozmo.schema import Tier

FRAME_SIZE = (320, 240)


def _write_video(
    path: Path,
    n_frames: int,
    blurred_indices: Optional[Sequence[int]] = None,
    fps: float = 15.0,
    seed: int = 0,
    blur_ramp: bool = False,
) -> None:
    """A short synthetic clip with real texture (random rectangles), so
    Laplacian variance is meaningfully nonzero and distinguishes sharp frames
    from the ones in `blurred_indices`, which get a heavy Gaussian blur.

    ``blur_ramp`` instead softens progressively from start to end -- the
    sharpness profile of a real handheld walk, steadiest before the operator
    starts moving. Every frame stays well above a low blur threshold, so what
    it exercises is the *cap*, not the blur filter.
    """
    blurred = set(blurred_indices or ())
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, FRAME_SIZE)
    try:
        for i in range(n_frames):
            image = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
            for _ in range(25):
                x0, y0 = int(rng.integers(0, FRAME_SIZE[0])), int(rng.integers(0, FRAME_SIZE[1]))
                x1, y1 = int(rng.integers(0, FRAME_SIZE[0])), int(rng.integers(0, FRAME_SIZE[1]))
                color = tuple(int(c) for c in rng.integers(40, 255, size=3))
                cv2.rectangle(image, (x0, y0), (x1, y1), color, -1)
            if i in blurred:
                image = cv2.GaussianBlur(image, (31, 31), 20)
            elif blur_ramp and i > 0:
                # Sharpest at the start, softening steadily to the end.
                sigma = 0.05 + 2.0 * (i / max(1, n_frames - 1))
                image = cv2.GaussianBlur(image, (7, 7), sigma)
            writer.write(image)
    finally:
        writer.release()


class TestSampleFrames:
    def test_stride_and_cap(self, tmp_path):
        video = tmp_path / "walk.mp4"
        _write_video(video, n_frames=60)
        result = sample_frames(video, tmp_path / "frames", stride_frames=5, max_frames=8)

        assert result.total_video_frames == 60
        assert result.frames_at_stride == 12          # frames 0, 5, ..., 55
        assert result.capped is True
        assert len(result.frame_paths) == 8
        # Chronological order is restored after capping by sharpness.
        indices = [int(p.stem.split("_")[1]) for p in result.frame_paths]
        assert indices == sorted(indices)
        for path in result.frame_paths:
            assert path.is_file()

    def test_no_cap_needed_keeps_every_sampled_frame(self, tmp_path):
        video = tmp_path / "walk.mp4"
        _write_video(video, n_frames=20)
        result = sample_frames(video, tmp_path / "frames", stride_frames=5, max_frames=8)
        assert result.frames_at_stride == 4
        assert result.capped is False
        assert len(result.frame_paths) == 4

    def test_blur_filter_drops_blurred_frames(self, tmp_path):
        video = tmp_path / "walk.mp4"
        # Every stride-5 frame from 0..45 is blurred; the rest of the clip is sharp.
        _write_video(video, n_frames=90, blurred_indices=range(0, 46, 5))
        result = sample_frames(video, tmp_path / "frames", stride_frames=5, max_frames=20)

        assert result.frames_dropped_blur > 0
        assert result.frames_kept_blur == result.frames_at_stride - result.frames_dropped_blur
        assert len(result.frame_paths) == result.frames_kept_blur
        # None of the frames that survived should be among the deliberately blurred indices.
        kept_indices = {int(p.stem.split("_")[1]) for p in result.frame_paths}
        assert kept_indices.isdisjoint(set(range(0, 46, 5)))

    def test_too_few_sharp_frames_raises(self, tmp_path):
        video = tmp_path / "walk.mp4"
        _write_video(video, n_frames=10, blurred_indices=range(10))
        with pytest.raises(ValueError, match="survived"):
            sample_frames(video, tmp_path / "frames", stride_frames=1, blur_threshold=80.0)

    def test_the_cap_spans_the_whole_clip_not_just_its_sharpest_stretch(self, tmp_path):
        """A walkthrough is sharpest where the operator stood still, which is
        almost never spread evenly across the walk. Capping by global sharpness
        therefore collapsed the whole selection onto one stretch: on a real
        37 s walk through three rooms it took all 8 frames from the first 11
        seconds and discarded the other two rooms, so the reconstruction had
        only one room to find. The cap has to sample the clip, not its
        steadiest moment.
        """
        video = tmp_path / "walk.mp4"
        # Sharpness falls steadily from start to end, as on a real walk.
        _write_video(video, n_frames=120, blur_ramp=True)
        result = sample_frames(
            video, tmp_path / "frames", stride_frames=3, blur_threshold=1.0, max_frames=6
        )
        kept = sorted(int(p.stem.split("_")[-1]) for p in result.frame_paths)
        assert result.capped
        # Frames from the back half of the clip must survive the cap: taking
        # the globally sharpest six here keeps only the opening.
        assert max(kept) > 60, f"cap kept only the sharp opening stretch: {kept}"
        assert min(kept) < 60, f"cap kept only the tail: {kept}"
        # And they should be spread rather than bunched: with six frames over
        # 120, no gap should swallow more than half the clip.
        gaps = [b - a for a, b in zip(kept, kept[1:])]
        assert max(gaps) < 60, f"cap left a hole across the walk: {kept}"

    def test_summary_reports_the_parameters_used(self, tmp_path):
        video = tmp_path / "walk.mp4"
        _write_video(video, n_frames=30)
        result = sample_frames(video, tmp_path / "frames", stride_frames=3, blur_threshold=50.0, max_frames=6)
        summary = result.summary()
        assert summary["stride_frames"] == 3
        assert summary["blur_threshold"] == 50.0
        assert summary["max_frames"] == 6
        assert summary["frames_used"] == len(result.frame_paths)


def _write_video_capture(root: Path, n_frames: int = 40) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _write_video(root / "walkthrough.mp4", n_frames=n_frames)
    (root / "capture.json").write_text(json.dumps({
        "capture_id": "video_demo",
        "tier": "video",
        "declared_rooms": ["living_room"],
        "video_path": "walkthrough.mp4",
        "device": {"model": "synthetic-video", "has_lidar": False},
    }, indent=2) + "\n")
    return root


class TestVideoTierEndToEnd:
    """Real dispatch through run_capture, real frame sampling, the stub
    reconstruction backbone (--backbone stub) so this stays fast and needs no
    model weights -- see test_recon.py for real reconstruction accuracy."""

    def test_video_tier_runs_end_to_end(self, tmp_path):
        capture_dir = _write_video_capture(tmp_path / "cap")
        result = run_capture(
            capture_dir, tmp_path / "out", backbone_name="stub", semantics=False,
            video_stride=4, video_blur_threshold=10.0, video_max_frames=6,
        )

        plan = result.plan
        assert plan.tier is Tier.VIDEO
        assert len(plan.rooms) == 1
        room = plan.rooms[0]
        assert room.walls
        for wall in room.walls:
            assert wall.length.ci_95[0] <= wall.length.value <= wall.length.ci_95[1]

        sampling = plan.quality.video_sampling
        assert sampling is not None
        assert sampling["stride_frames"] == 4
        assert sampling["max_frames"] == 6
        assert 2 <= sampling["frames_used"] <= 6

        manifest = json.loads(result.manifest_path.read_text())
        settings = manifest["reconstruction_settings"]
        assert settings["video_stride"] == 4
        assert settings["video_max_frames"] == 6
        assert result.plan_path.is_file()
        assert result.rendered  # plan.png/plan.svg were drawn

    def test_video_tier_is_honest_about_being_uncalibrated(self, tmp_path):
        capture_dir = _write_video_capture(tmp_path / "cap")
        result = run_capture(capture_dir, tmp_path / "out", backbone_name="stub", semantics=False)
        assert "ncalibrated" in result.plan.quality.calibration_note
        assert result.plan.quality.semantics_available is False

    def test_two_runs_produce_identical_plans(self, tmp_path):
        capture_dir = _write_video_capture(tmp_path / "cap")
        first = run_capture(capture_dir, tmp_path / "a", backbone_name="stub", semantics=False).plan
        second = run_capture(capture_dir, tmp_path / "b", backbone_name="stub", semantics=False).plan
        assert [w.length.value for w in first.rooms[0].walls] == \
               [w.length.value for w in second.rooms[0].walls]
