"""Capture dispatch.

Tier is a property of the capture, not of the invocation: it is read from
``capture.json`` inside the input directory and never passed as a CLI flag. A
capture that lies about its tier fails here rather than three stages later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, field_validator

from ..schema import StrictModel, Tier
from .photo import PhotoCapture, load_photo_capture
from .photo import summarize as summarize_photo
from .stray import StrayCapture, load_stray_capture
from .stray import summarize as summarize_stray

CAPTURE_MANIFEST_NAME = "capture.json"
VIDEO_SUFFIXES = {".mov", ".mp4", ".m4v"}


class DeviceInfo(StrictModel):
    """Which hardware produced this capture -- the row of the device matrix the
    run has to be judged against."""

    model: str = "unknown"
    os_version: Optional[str] = None
    has_lidar: Optional[bool] = None


class CaptureManifest(StrictModel):
    """``capture.json``: the one file every capture route must write."""

    capture_id: str
    tier: Tier
    device: DeviceInfo = Field(default_factory=DeviceInfo)
    captured_at: Optional[datetime] = None
    operator: Optional[str] = None
    # Rooms the operator says are in this capture. Advisory: the pipeline still
    # decides what it can actually reconstruct, but a mismatch is worth a warning.
    declared_rooms: List[str] = Field(default_factory=list)
    # Repeat captures of the same physical space share this key. The
    # repeatability gate pairs captures on it.
    space_id: Optional[str] = None
    video_path: Optional[str] = None
    notes: Optional[str] = None

    @field_validator("capture_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("capture_id must not be empty")
        return value.strip()


@dataclass
class VideoCapture:
    """Video tier: one handheld walkthrough clip. Located, not decoded."""

    root: Path
    video: Path
    size_bytes: int


@dataclass
class CaptureBundle:
    """Everything a pipeline stage needs to know about one capture."""

    root: Path
    manifest: CaptureManifest
    payload: Union[PhotoCapture, VideoCapture, StrayCapture]
    warnings: List[str]

    @property
    def tier(self) -> Tier:
        return self.manifest.tier

    @property
    def capture_id(self) -> str:
        return self.manifest.capture_id

    @property
    def space_id(self) -> str:
        """Repeat captures of one space share this; defaults to the capture id
        so a one-off capture is simply a space of size one."""
        return self.manifest.space_id or self.manifest.capture_id

    def room_names(self) -> List[str]:
        if isinstance(self.payload, PhotoCapture):
            return self.payload.room_names
        return list(self.manifest.declared_rooms)

    def summary(self) -> Dict[str, Any]:
        if isinstance(self.payload, PhotoCapture):
            payload = summarize_photo(self.payload)
        elif isinstance(self.payload, StrayCapture):
            payload = summarize_stray(self.payload)
        else:
            payload = {"video": self.payload.video.name, "size_bytes": self.payload.size_bytes}
        return {
            "capture_id": self.capture_id,
            "space_id": self.space_id,
            "tier": self.tier.value,
            "device": self.manifest.device.model_dump(mode="json"),
            "rooms": self.room_names(),
            "payload": payload,
            "warnings": self.warnings,
        }


def read_manifest(input_dir: Path) -> CaptureManifest:
    path = Path(input_dir) / CAPTURE_MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Every capture directory must carry a {CAPTURE_MANIFEST_NAME} "
            f"declaring at least capture_id and tier (one of: "
            f"{', '.join(t.value for t in Tier)})."
        )
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    return CaptureManifest.model_validate(raw)


def _load_video(root: Path, manifest: CaptureManifest) -> VideoCapture:
    if manifest.video_path:
        candidate = root / manifest.video_path
        if not candidate.is_file():
            raise FileNotFoundError(f"capture.json points at {candidate}, which does not exist")
        video = candidate
    else:
        found = sorted(
            (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES),
            key=lambda p: p.name,
        )
        if not found:
            raise FileNotFoundError(
                f"video-tier capture {root} contains no {sorted(VIDEO_SUFFIXES)} file "
                f"and capture.json sets no video_path"
            )
        video = found[0]
    return VideoCapture(root=root, video=video, size_bytes=video.stat().st_size)


def load_capture(input_dir: Path) -> CaptureBundle:
    """Read capture.json and hand back the loader output for its tier."""
    root = Path(input_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"input is not a directory: {root}")

    manifest = read_manifest(root)
    warnings: List[str] = []

    if manifest.tier is Tier.PHOTO:
        payload: Union[PhotoCapture, VideoCapture, StrayCapture] = load_photo_capture(root)
        warnings.extend(payload.warnings())
    elif manifest.tier is Tier.VIDEO:
        payload = _load_video(root, manifest)
    elif manifest.tier is Tier.LIDAR:
        payload = load_stray_capture(root)
        warnings.extend(payload.warnings)
        if not payload.has_depth:
            warnings.append("capture declares the LiDAR tier but carries no depth frames")
    else:  # pragma: no cover - Tier is exhaustive
        raise ValueError(f"unhandled tier {manifest.tier}")

    declared = set(manifest.declared_rooms)
    if declared and isinstance(payload, PhotoCapture):
        found = set(payload.room_names)
        for missing in sorted(declared - found):
            warnings.append(f"capture.json declares room '{missing}' but no folder was found")
        for extra in sorted(found - declared):
            warnings.append(f"found room folder '{extra}' not declared in capture.json")

    return CaptureBundle(root=root, manifest=manifest, payload=payload, warnings=warnings)
