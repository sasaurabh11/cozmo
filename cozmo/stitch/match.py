"""Correspondences between rooms.

Two independent signals, because they fail differently:

* **Image matching.** DINOv2 gives one embedding per room (averaged over its
  frames) for coarse retrieval -- cheap, and enough to rule out room pairs that
  share no view of each other at all. Any pair that survives is checked
  properly with SuperPoint (keypoints) + LightGlue (matches): real
  correspondences, not a similarity score standing in for them.
* **Doorway geometry.** The strongest signal in a house is a doorway
  photographed from both sides: two rooms each reporting an opening of
  matching width and type on a wall is evidence a room-scale reconstruction
  can produce even when the two rooms' photos have no visual overlap at all
  (a door usually looks like a rectangle of hallway light from one side and a
  room interior from the other -- nothing for a keypoint matcher to grab).

Both signals are reported per room pair; :mod:`cozmo.stitch.graph` decides how
to combine them into one relative transform.

Runs in its own process (see :mod:`cozmo.stitch.worker`) for the same reason
:mod:`cozmo.semantics.detect` does: torch and open3d cannot share a process on
this platform, and the caller (``cozmo.pipeline.run``) already has open3d
loaded for the LiDAR path.
"""

from __future__ import annotations

import itertools
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("cozmo.stitch.match")

DEFAULT_WEIGHTS_DIR = Path(os.environ.get("COZMO_WEIGHTS_DIR", "weights"))

# Below this DINOv2 cosine similarity, two rooms are not worth spending
# SuperPoint+LightGlue on -- the coarse-retrieval shortlist the brief asks for.
DINOV2_SHORTLIST_THRESHOLD = 0.35

# A LightGlue match below this score is noise, not a correspondence.
MATCH_SCORE_THRESHOLD = 0.3
# Fewer real matches than this and the pair is not trustworthy enough to
# anchor a room-to-room transform on image evidence alone.
MIN_INLIER_MATCHES = 6
# Frame pairs tried per room pair, capped for cost -- most of the signal is in
# the best-covering view, not an exhaustive cross product.
MAX_FRAME_PAIRS_TRIED = 9

# Two openings are "the same doorway" when their widths agree this closely.
DOORWAY_WIDTH_TOLERANCE = 0.15


class MatcherUnavailable(RuntimeError):
    """Weights are missing or the models failed to load."""


@dataclass
class RoomFrames:
    """What match.py needs from one room: its frames and its openings."""

    room_id: str
    frame_paths: List[str]
    frame_indices: List[int]           # matches the reconstruction's own frame_index
    openings: List[Dict[str, Any]] = field(default_factory=list)  # {id, kind, width_m, wall_id}


@dataclass
class KeypointMatch:
    frame_index_a: int
    frame_index_b: int
    u_a: float
    v_a: float
    u_b: float
    v_b: float
    score: float


@dataclass
class DoorwayMatch:
    opening_id_a: str
    opening_id_b: str
    width_a_m: float
    width_b_m: float


@dataclass
class RoomPairMatch:
    room_a: str
    room_b: str
    dinov2_similarity: float
    keypoint_matches: List[KeypointMatch] = field(default_factory=list)
    doorway_matches: List[DoorwayMatch] = field(default_factory=list)
    best_frame_pair: Optional[Tuple[int, int]] = None

    @property
    def has_evidence(self) -> bool:
        return len(self.keypoint_matches) >= MIN_INLIER_MATCHES or bool(self.doorway_matches)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "room_a": self.room_a, "room_b": self.room_b,
            "dinov2_similarity": round(self.dinov2_similarity, 4),
            "keypoint_match_count": len(self.keypoint_matches),
            "keypoint_matches": [
                {"frame_index_a": m.frame_index_a, "frame_index_b": m.frame_index_b,
                 "u_a": m.u_a, "v_a": m.v_a, "u_b": m.u_b, "v_b": m.v_b, "score": round(m.score, 4)}
                for m in self.keypoint_matches
            ],
            "doorway_matches": [
                {"opening_id_a": d.opening_id_a, "opening_id_b": d.opening_id_b,
                 "width_a_m": d.width_a_m, "width_b_m": d.width_b_m}
                for d in self.doorway_matches
            ],
            "best_frame_pair": list(self.best_frame_pair) if self.best_frame_pair else None,
        }


def _match_doorways(room_a: RoomFrames, room_b: RoomFrames) -> List[DoorwayMatch]:
    matches = []
    for oa in room_a.openings:
        if oa.get("kind") != "door":
            continue
        for ob in room_b.openings:
            if ob.get("kind") != "door":
                continue
            wa, wb = float(oa["width_m"]), float(ob["width_m"])
            if abs(wa - wb) / max(wa, wb) <= DOORWAY_WIDTH_TOLERANCE:
                matches.append(DoorwayMatch(oa["id"], ob["id"], wa, wb))
    return matches


class RoomMatcher:
    """DINOv2 + SuperPoint + LightGlue, loaded from local weights only."""

    def __init__(self, weights_dir: Optional[Path] = None, device: Optional[str] = None) -> None:
        self.weights_dir = Path(weights_dir or DEFAULT_WEIGHTS_DIR)

        try:
            import torch
        except ImportError as exc:
            raise MatcherUnavailable(f"torch not installed: {exc}") from exc
        self._torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available() else "cpu")

        try:
            from transformers import (
                AutoImageProcessor, Dinov2Model,
                LightGlueForKeypointMatching, LightGlueImageProcessor,
                SuperPointForKeypointDetection, SuperPointImageProcessor,
            )
        except ImportError as exc:
            raise MatcherUnavailable(f"transformers not installed: {exc}") from exc

        def local(name: str) -> Path:
            path = self.weights_dir / name
            if not (path / "model.safetensors").is_file():
                raise MatcherUnavailable(
                    f"{name} weights not found in {path}. Run scripts/fetch_weights.sh first."
                )
            return path

        self.dinov2_proc = AutoImageProcessor.from_pretrained(local("dinov2-small"), local_files_only=True)
        self.dinov2 = Dinov2Model.from_pretrained(
            local("dinov2-small"), local_files_only=True
        ).to(self.device).eval()

        self.sp_proc = SuperPointImageProcessor.from_pretrained(local("superpoint"), local_files_only=True)
        self.sp_model = SuperPointForKeypointDetection.from_pretrained(
            local("superpoint"), local_files_only=True
        ).to(self.device).eval()

        self.lg_proc = LightGlueImageProcessor.from_pretrained(
            local("lightglue_superpoint"), local_files_only=True
        )
        self.lg_model = LightGlueForKeypointMatching.from_pretrained(
            local("lightglue_superpoint"), local_files_only=True
        ).to(self.device).eval()

        log.info("room matcher ready on %s", self.device)

    def _load(self, path: str) -> np.ndarray:
        import cv2
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not decode image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def embed_room(self, room: RoomFrames, max_frames: int = 3) -> np.ndarray:
        """One DINOv2 descriptor per room, averaged over up to a few frames."""
        torch = self._torch
        vectors = []
        for path in room.frame_paths[:max_frames]:
            image = self._load(path)
            # input_data_format is explicit, not defensive. transformers infers
            # the channel axis from the shape, and for a square image whose side
            # is <= 4 px -- (1, 1, 3), (3, 3, 3) -- (H, W, C) and (C, H, W) are
            # indistinguishable, so it guesses channels_first, warns, and then
            # normalises a 1-channel 1x3 image. cv2.imread always returns
            # (H, W, C), so saying so removes the guess.
            inputs = self.dinov2_proc(
                images=image, return_tensors="pt", input_data_format="channels_last"
            ).to(self.device)
            with torch.no_grad():
                out = self.dinov2(**inputs)
            vectors.append(out.pooler_output[0].cpu().numpy())
        mean = np.mean(vectors, axis=0)
        return mean / max(np.linalg.norm(mean), 1e-9)

    def match_frame_pair(
        self, path_a: str, path_b: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Real keypoint correspondences between two images.

        Returns ``(keypoints_a, keypoints_b, scores)``, each pixel-space in its
        own image, already filtered to matches -- ``post_process_keypoint_matching``
        does the un-padding and score-thresholding.
        """
        torch = self._torch
        image_a, image_b = self._load(path_a), self._load(path_b)

        inputs = self.lg_proc(
            images=[[image_a, image_b]], return_tensors="pt",
            input_data_format="channels_last",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.lg_model(**inputs)

        sizes = [[(image_a.shape[0], image_a.shape[1]), (image_b.shape[0], image_b.shape[1])]]
        result = self.lg_proc.post_process_keypoint_matching(
            outputs, sizes, threshold=MATCH_SCORE_THRESHOLD
        )[0]
        return (
            result["keypoints0"].cpu().numpy(),
            result["keypoints1"].cpu().numpy(),
            result["matching_scores"].cpu().numpy(),
        )

    def match_rooms(self, rooms: Sequence[RoomFrames]) -> List[RoomPairMatch]:
        """The full pipeline: DINOv2 shortlist, then real matching on survivors."""
        embeddings = {room.room_id: self.embed_room(room) for room in rooms}

        results: List[RoomPairMatch] = []
        for room_a, room_b in itertools.combinations(rooms, 2):
            similarity = float(np.dot(embeddings[room_a.room_id], embeddings[room_b.room_id]))
            doorways = _match_doorways(room_a, room_b)

            if similarity < DINOV2_SHORTLIST_THRESHOLD and not doorways:
                log.info(
                    "%s <-> %s: dinov2 similarity %.2f, no doorway match; not shortlisted",
                    room_a.room_id, room_b.room_id, similarity,
                )
                results.append(RoomPairMatch(room_a.room_id, room_b.room_id, similarity))
                continue

            pairs_tried = list(itertools.islice(
                itertools.product(
                    zip(room_a.frame_indices, room_a.frame_paths),
                    zip(room_b.frame_indices, room_b.frame_paths),
                ),
                MAX_FRAME_PAIRS_TRIED,
            ))

            best_matches: List[KeypointMatch] = []
            best_pair: Optional[Tuple[int, int]] = None
            for (idx_a, path_a), (idx_b, path_b) in pairs_tried:
                kp_a, kp_b, scores = self.match_frame_pair(path_a, path_b)
                if len(scores) > len(best_matches):
                    best_matches = [
                        KeypointMatch(idx_a, idx_b, float(kp_a[i, 0]), float(kp_a[i, 1]),
                                     float(kp_b[i, 0]), float(kp_b[i, 1]), float(scores[i]))
                        for i in range(len(scores))
                    ]
                    best_pair = (idx_a, idx_b)

            log.info(
                "%s <-> %s: dinov2 %.2f, %d keypoint match(es), %d doorway match(es)",
                room_a.room_id, room_b.room_id, similarity, len(best_matches), len(doorways),
            )
            results.append(RoomPairMatch(
                room_a.room_id, room_b.room_id, similarity,
                keypoint_matches=best_matches, doorway_matches=doorways, best_frame_pair=best_pair,
            ))
        return results
