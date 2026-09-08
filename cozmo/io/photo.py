"""Photo-tier reader: per-room photo folders.

The photo tier is the floor of the contract -- any picture in, results out -- so
this reader is deliberately forgiving about layout and loud about what it found.
It does no decoding; it resolves folders to an ordered, room-keyed index of
image paths that the reconstruction stage consumes.

Accepted layouts (checked in order)::

    <input>/rooms/<room_name>/*.jpg      # preferred, written by the protocol
    <input>/<room_name>/*.jpg            # bare per-room folders
    <input>/*.jpg                        # single unnamed room, named "room_1"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".dng"}

# Contract floor: 2 to 8 stills per room. Outside that band we still run, but
# the reason is recorded and the reconstruction is expected to widen intervals.
MIN_PHOTOS_PER_ROOM = 2
MAX_PHOTOS_PER_ROOM = 8


@dataclass
class RoomPhotos:
    room_name: str
    images: List[Path]
    warnings: List[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.images)


@dataclass
class PhotoCapture:
    root: Path
    rooms: List[RoomPhotos]
    layout: str

    @property
    def room_names(self) -> List[str]:
        return [r.room_name for r in self.rooms]

    @property
    def image_count(self) -> int:
        return sum(r.count for r in self.rooms)

    def warnings(self) -> List[str]:
        out: List[str] = []
        for room in self.rooms:
            out.extend(f"{room.room_name}: {w}" for w in room.warnings)
        return out


def _images_in(folder: Path) -> List[Path]:
    """Images directly inside a folder, sorted by name for determinism."""
    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: p.name,
    )


def _room_from_folder(folder: Path) -> Optional[RoomPhotos]:
    images = _images_in(folder)
    if not images:
        return None
    room = RoomPhotos(room_name=folder.name, images=images)
    if room.count < MIN_PHOTOS_PER_ROOM:
        room.warnings.append(f"{room.count} photo(s), protocol asks for >= {MIN_PHOTOS_PER_ROOM}")
    if room.count > MAX_PHOTOS_PER_ROOM:
        room.warnings.append(
            f"{room.count} photos, protocol caps at {MAX_PHOTOS_PER_ROOM}; extras are used but unweighted"
        )
    return room


def load_photo_capture(root: Path) -> PhotoCapture:
    """Resolve a photo-tier capture directory into per-room image lists."""
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"photo capture root is not a directory: {root}")

    rooms_dir = root / "rooms"
    if rooms_dir.is_dir():
        search_root, layout = rooms_dir, "rooms/<room>/*"
    else:
        search_root, layout = root, "<room>/*"

    rooms: List[RoomPhotos] = []
    for folder in sorted((p for p in search_root.iterdir() if p.is_dir()), key=lambda p: p.name):
        room = _room_from_folder(folder)
        if room is not None:
            rooms.append(room)

    if not rooms:
        loose = _images_in(search_root)
        if loose:
            rooms = [
                RoomPhotos(
                    room_name="room_1",
                    images=loose,
                    warnings=["photos were loose in the capture root; assumed a single room"],
                )
            ]
            layout = "<root>/*"

    if not rooms:
        raise FileNotFoundError(
            f"no images found under {root} (looked for {sorted(IMAGE_SUFFIXES)})"
        )

    return PhotoCapture(root=root, rooms=rooms, layout=layout)


def summarize(capture: PhotoCapture) -> Dict[str, object]:
    """Compact description for the run manifest."""
    return {
        "layout": capture.layout,
        "room_count": len(capture.rooms),
        "image_count": capture.image_count,
        "photos_per_room": {r.room_name: r.count for r in capture.rooms},
        "warnings": capture.warnings(),
    }
