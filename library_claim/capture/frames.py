"""Frame store and keyframe selection.

Every high-resolution frame the phone uploads is written to
``<sweep>/frames/<id>.jpg`` so that every ``frame_ref`` in the packet opens.
A frame becomes a *keyframe* (sent to the vision reader) when it is sharp and
either shows enough new area relative to the previous keyframe or enough time
has passed.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import cv2
import numpy as np

from ..schemas import FrameRecord, Imu
from . import quality

NEW_AREA_MIN = 0.35      # keyframe when <65 % of the frame overlaps the last keyframe
MAX_KEYFRAME_GAP_S = 2.5


class FrameStore:
    def __init__(self, sweep_dir: Path):
        self.dir = Path(sweep_dir)
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        self.frames: dict[str, FrameRecord] = {}
        self.order: list[str] = []
        self._lock = threading.Lock()
        self._prev_feat = None
        self._kf_feat = None
        self._kf_t = -1e9
        self._kf_size = (640, 480)

    def path(self, frame_id: str) -> Path:
        return self.dir / self.frames[frame_id].path

    def load(self, frame_id: str) -> np.ndarray:
        img = cv2.imread(str(self.path(frame_id)))
        if img is None:
            raise FileNotFoundError(frame_id)
        return img

    def add(self, jpeg: bytes, t: float, imu: dict | None = None) -> FrameRecord:
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Frame is not a decodable JPEG")
        with self._lock:
            idx = len(self.order)
            frame_id = f"f{idx:05d}"
            rel = f"frames/{frame_id}.jpg"
            (self.dir / rel).write_bytes(jpeg)
            grey = quality.working_grey(img)
            feat = quality.features(grey)
            motion = None
            if self._prev_feat is not None:
                _, motion, _ = quality.match_homography(self._prev_feat, feat)
            self._prev_feat = feat
            q = quality.assess(img, motion)
            rec = FrameRecord(id=frame_id, t=round(t, 3), path=rel, width=img.shape[1], height=img.shape[0],
                              imu=Imu(**(imu or {})), quality=q)
            rec.keyframe = self._is_keyframe(rec, feat, grey.shape[1], grey.shape[0])
            self.frames[frame_id] = rec
            self.order.append(frame_id)
            with open(self.dir / "frames.jsonl", "a", encoding="utf-8") as fh:
                fh.write(rec.model_dump_json() + "\n")
            return rec

    def _is_keyframe(self, rec: FrameRecord, feat, w: int, h: int) -> bool:
        q = rec.quality
        if q is None or "blur" in q.issues or "dark" in q.issues or "fast" in q.issues:
            return False
        new_area = 1.0
        if self._kf_feat is not None:
            H, _, _ = quality.match_homography(feat, self._kf_feat)
            if H is not None:
                new_area = 1.0 - quality.overlap_fraction(H, w, h)
        if new_area >= NEW_AREA_MIN or rec.t - self._kf_t >= MAX_KEYFRAME_GAP_S:
            self._kf_feat, self._kf_t, self._kf_size = feat, rec.t, (w, h)
            return True
        return False

    def keyframes(self) -> list[FrameRecord]:
        return [self.frames[i] for i in self.order if self.frames[i].keyframe]

    def frame_ref(self, frame_id: str) -> str:
        rec = self.frames.get(frame_id)
        if rec is None:
            return ""
        return f"{rec.path}#t={rec.t:.2f}"

    def dump_index(self) -> None:
        (self.dir / "frames_index.json").write_text(
            json.dumps([self.frames[i].model_dump() for i in self.order], indent=1), encoding="utf-8")
