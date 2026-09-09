"""Load a photo folder into Frame objects.

Photo tier input: 2 to 8 unposed stills per room, no depth, no poses, no
intrinsics. The only thing worth extracting per image is its focal length --
everything downstream (scale recovery, layout) works far better with a real
intrinsic than a guessed one, and the difference between the two has to be
visible in the output, not silently averaged away.

EXIF stores focal length in millimetres against a specific sensor (35mm-equiv
when available, native otherwise). Converting that to a pixel focal length
needs the sensor width, which most phone EXIF omits. Where EXIF gives us the
35mm-equivalent focal length, the conversion is exact regardless of the actual
sensor: a 35mm-equiv focal length behaves as if shot on a 36mm-wide sensor, so
fx_px = 35mm_equiv_focal_mm / 36mm * image_width_px.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger("cozmo.recon.frames")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif"}

# iPhone wide-camera horizontal FOV is close to 63-70 deg across models; 65 deg
# is the middle of that band and is used whenever EXIF gives us nothing to work
# with. This is a guess, and every frame that falls back to it says so.
DEFAULT_HORIZONTAL_FOV_DEG = 65.0

# The reference width a 35mm-equivalent focal length is defined against.
FULL_FRAME_SENSOR_WIDTH_MM = 36.0


@dataclass
class Frame:
    """One photo: pixels plus what we could recover about how it was shot."""

    index: int
    path: Path
    image: np.ndarray                 # (H, W, 3) uint8, RGB
    K: np.ndarray                     # (3, 3) intrinsics in pixel units
    intrinsics_source: str            # "exif_35mm_equiv" | "exif_focal_mm" | "fov_default"
    focal_length_mm: Optional[float] = None

    @property
    def size(self) -> Tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h

    @property
    def horizontal_fov_deg(self) -> float:
        w = self.image.shape[1]
        return float(np.degrees(2 * np.arctan2(w / 2, self.K[0, 0])))


def _read_exif(path: Path) -> dict:
    """Best-effort EXIF read. Returns {} for anything unreadable -- a missing
    or corrupt EXIF block is not a reason to fail the whole capture."""
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
    except ImportError:
        return {}

    try:
        with Image.open(path) as img:
            raw = img.getexif()
            if not raw:
                return {}
            tags = {TAGS.get(k, k): v for k, v in raw.items()}
            # IFD 0x8769 (Exif) carries FocalLength / FocalLengthIn35mmFilm on
            # most cameras; Pillow exposes it via get_ifd.
            try:
                exif_ifd = raw.get_ifd(0x8769)
                tags.update({TAGS.get(k, k): v for k, v in exif_ifd.items()})
            except Exception:
                pass
            return tags
    except Exception as exc:  # noqa: BLE001 - a bad file must not fail the folder
        log.debug("EXIF read failed for %s: %s", path.name, exc)
        return {}


def _focal_px_from_exif(tags: dict, width_px: int) -> Tuple[Optional[float], Optional[float], str]:
    """Returns (focal_px, focal_length_mm, source)."""
    equiv_35mm = tags.get("FocalLengthIn35mmFilm")
    if equiv_35mm:
        focal_px = float(equiv_35mm) / FULL_FRAME_SENSOR_WIDTH_MM * width_px
        return focal_px, None, "exif_35mm_equiv"

    focal_mm = tags.get("FocalLength")
    if focal_mm:
        # A native focal length without a sensor width cannot be converted to
        # pixels correctly; treated as informational only, not used for K.
        value = float(focal_mm[0]) / float(focal_mm[1]) if isinstance(focal_mm, tuple) else float(focal_mm)
        return None, value, "exif_focal_mm_unconvertible"

    return None, None, "fov_default"


def _load_image(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_frame(path: Path, index: int) -> Frame:
    image = _load_image(path)
    height, width = image.shape[:2]
    tags = _read_exif(path)
    focal_px, focal_mm, source = _focal_px_from_exif(tags, width)

    if focal_px is None:
        focal_px = (width / 2) / np.tan(np.radians(DEFAULT_HORIZONTAL_FOV_DEG / 2))
        if source == "exif_focal_mm_unconvertible":
            log.info(
                "%s: EXIF focal length %.1f mm has no usable sensor width; "
                "falling back to a %.0f deg FOV default",
                path.name, focal_mm or 0.0, DEFAULT_HORIZONTAL_FOV_DEG,
            )
            source = "fov_default_no_sensor_width"
        else:
            log.info(
                "%s: no usable EXIF focal length; assuming %.0f deg horizontal FOV",
                path.name, DEFAULT_HORIZONTAL_FOV_DEG,
            )

    K = np.array([
        [focal_px, 0.0, width / 2.0],
        [0.0, focal_px, height / 2.0],
        [0.0, 0.0, 1.0],
    ])

    return Frame(index=index, path=path, image=image, K=K,
                 intrinsics_source=source, focal_length_mm=focal_mm)


def load_photo_folder(root: Path) -> List[Frame]:
    """Load every image directly inside one room's photo folder, in name order."""
    root = Path(root)
    paths = sorted(
        (p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: p.name,
    )
    if not paths:
        raise FileNotFoundError(f"no images found in {root}")

    frames = [load_frame(path, index) for index, path in enumerate(paths)]
    sources = {f.intrinsics_source for f in frames}
    log.info("loaded %d frame(s) from %s (intrinsics: %s)", len(frames), root.name, sorted(sources))
    return frames


def summarize(frames: List[Frame]) -> dict:
    return {
        "frame_count": len(frames),
        "intrinsics_sources": [f.intrinsics_source for f in frames],
        "focal_px": [round(float(f.K[0, 0]), 1) for f in frames],
        "sizes": [f.size for f in frames],
    }
