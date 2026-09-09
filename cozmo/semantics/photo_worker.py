"""Grounding-DINO opening detection for the photo/video tiers.

This worker deliberately stays separate from the reconstruction process. The
photo/video pipeline has already loaded Open3D, while the detector loads torch;
keeping them in different interpreters avoids the OpenMP crashes seen on macOS.
Only image-space boxes cross the boundary. The parent process projects those
boxes through the reconstructed camera poses and wall planes.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import cv2

from .detect import DetectorUnavailable, OpenVocabularyDetector, detect_openings


def main(argv: List[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} request.json response.json", file=sys.stderr)
        return 2

    request = json.loads(Path(argv[1]).read_text())
    response_path = Path(argv[2])
    try:
        detector = OpenVocabularyDetector(
            weights_dir=request.get("weights_dir"),
            refine_masks=False,
        )
        detections: List[Dict[str, Any]] = []
        for frame in request["frames"]:
            image = cv2.imread(str(frame["path"]), cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for detection in detect_openings(detector, image, int(frame["index"])):
                detections.append({
                    "frame_index": detection.frame_index,
                    "prompt": detection.prompt,
                    "score": detection.score,
                    "box_xyxy": list(detection.box_xyxy),
                })
        payload = {
            "ok": True,
            "detector": "grounding-dino-tiny",
            "device": detector.device,
            "detections": detections,
        }
    except DetectorUnavailable as exc:
        payload = {
            "ok": False,
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "detections": [],
        }
    except Exception as exc:  # noqa: BLE001 - parent receives the cause
        payload = {
            "ok": False,
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "detections": [],
        }

    response_path.write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
