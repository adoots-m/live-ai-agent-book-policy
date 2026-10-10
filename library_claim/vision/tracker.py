"""Stage: de-duplicate observations across overlapping keyframes.

A single sweep shows each spine in several keyframes. We chain planar
homographies between consecutive keyframes (ORB + RANSAC). A bookcase face
is close to planar, so a homography maps a spine in keyframe k into the
coordinates of the segment's first keyframe ("mosaic" coordinates). Spine
observations whose mosaic boxes overlap are the same physical book.

When tracking is lost (fast pan, turned to another wall) a new segment starts.
Across segments, only legible spines with matching text are merged; the rest
are kept and the segment boundary is listed for review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import cv2
import numpy as np

from ..schemas import FrameVision, SpineObs

WORK_W = 1280


def norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", (s or "").lower()).strip()


def text_sim(a: str, b: str) -> float:
    a, b = norm_text(a), norm_text(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def overlap_1d(a0, a1, b0, b1) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    return inter / max(1e-6, min(a1 - a0, b1 - b0))


def project_box(H: np.ndarray, box) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    pts = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).reshape(-1, 1, 2)
    q = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())


def project_quad(H: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1 = box
    pts = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


@dataclass
class Track:
    id: str
    kind: str                      # spine | item | unit
    segment: int
    obs: list = field(default_factory=list)
    mosaic_box: tuple = (0, 0, 0, 0)

    def best_spine(self) -> SpineObs:
        return max(self.obs, key=lambda o: (o.legible, len(o.title_read), len(o.text), "cut_at_edge" not in o.flags))


@dataclass
class KeyframeGeom:
    frame_id: str
    segment: int
    H_to_ref: np.ndarray           # full-res pixels of this keyframe -> segment reference pixels
    width: int
    height: int
    inliers: int = 0


class Tracker:
    def __init__(self):
        self._orb = cv2.ORB_create(nfeatures=2500)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.geom: dict[str, KeyframeGeom] = {}
        self.order: list[str] = []
        self._feat: dict[str, tuple] = {}
        self.segment = -1
        self.spines: list[Track] = []
        self.items: list[Track] = []
        self.units: list[Track] = []
        self.segment_breaks: list[str] = []   # frame ids where a new segment started
        self._n = 0

    # ---------------------------------------------------------------- geometry
    def _features(self, img: np.ndarray):
        h, w = img.shape[:2]
        s = WORK_W / w if w > WORK_W else 1.0
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        if s != 1.0:
            grey = cv2.resize(grey, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        kp, des = self._orb.detectAndCompute(grey, None)
        return kp, des, s

    def _homography(self, fa, fb):
        """Homography mapping full-res pixels of A into full-res pixels of B."""
        kpa, da, sa = fa
        kpb, db, sb = fb
        if da is None or db is None or len(kpa) < 30 or len(kpb) < 30:
            return None, 0
        knn = self._bf.knnMatch(da, db, k=2)
        good = [m for m, *rest in knn if rest and m.distance < 0.78 * rest[0].distance]
        if len(good) < 30:
            return None, 0
        pa = np.float32([kpa[m.queryIdx].pt for m in good]) / sa
        pb = np.float32([kpb[m.trainIdx].pt for m in good]) / sb
        H, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 6.0)
        if H is None:
            return None, 0
        inl = int(mask.sum())
        if inl < 30 or not _sane(H):
            return None, inl
        return H, inl

    def add_keyframe(self, frame_id: str, img: np.ndarray) -> KeyframeGeom:
        feat = self._features(img)
        h, w = img.shape[:2]
        best = None
        if self.order:
            for prev in self.order[-2:][::-1]:
                g_prev = self.geom[prev]
                if g_prev.segment != self.segment:
                    continue
                H, inl = self._homography(feat, self._feat[prev])
                if H is not None and (best is None or inl > best[1]):
                    best = (g_prev.H_to_ref @ H, inl)
        if best is None:
            self.segment += 1
            if self.order:
                self.segment_breaks.append(frame_id)
            g = KeyframeGeom(frame_id, self.segment, np.eye(3), w, h, 0)
        else:
            g = KeyframeGeom(frame_id, self.segment, best[0], w, h, best[1])
        self.geom[frame_id] = g
        self.order.append(frame_id)
        self._feat[frame_id] = feat
        # Keep memory bounded: we only ever match against the last two keyframes.
        for old in self.order[:-3]:
            self._feat.pop(old, None)
        return g

    # ---------------------------------------------------------------- association
    def _new_id(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}{self._n:04d}"

    def add_vision(self, fv: FrameVision) -> dict:
        g = self.geom[fv.frame_id]
        new_spines = 0
        for s in fv.spines:
            mb = project_box(g.H_to_ref, s.box)
            tr = self._match_spine(g.segment, mb, s)
            if tr is None:
                self.spines.append(Track(self._new_id("B"), "spine", g.segment, [s], mb))
                new_spines += 1
            else:
                tr.obs.append(s)
                tr.mosaic_box = _avg_box(tr.mosaic_box, mb, len(tr.obs))
        for it in fv.items:
            mb = project_box(g.H_to_ref, it.box)
            tr = next((t for t in self.items if t.segment == g.segment and t.obs[0].category == it.category
                       and box_iou(t.mosaic_box, mb) > 0.3), None)
            if tr is None:
                self.items.append(Track(self._new_id("I"), "item", g.segment, [it], mb))
            else:
                tr.obs.append(it)
        for u in fv.shelf_units:
            mb = project_box(g.H_to_ref, u.box)
            tr = next((t for t in self.units if t.segment == g.segment and box_iou(t.mosaic_box, mb) > 0.3), None)
            if tr is None:
                self.units.append(Track(self._new_id("U"), "unit", g.segment, [u], mb))
            else:
                tr.obs.append(u)
                tr.mosaic_box = _union(tr.mosaic_box, mb)
        return {"new_spines": new_spines}

    def _match_spine(self, segment: int, mb, s: SpineObs):
        best, best_score = None, 0.0
        for t in self.spines:
            if t.segment != segment:
                continue
            o = t.obs[-1]
            if o.orientation != s.orientation:
                continue
            if s.orientation == "vertical":
                ov = overlap_1d(t.mosaic_box[0], t.mosaic_box[2], mb[0], mb[2])
                vy = overlap_1d(t.mosaic_box[1], t.mosaic_box[3], mb[1], mb[3])
            else:
                ov = overlap_1d(t.mosaic_box[1], t.mosaic_box[3], mb[1], mb[3])
                vy = overlap_1d(t.mosaic_box[0], t.mosaic_box[2], mb[0], mb[2])
            if ov < 0.5 or vy < 0.5:
                continue
            bt = t.best_spine()
            if bt.legible and s.legible and text_sim(bt.title_read, s.title_read) < 0.5:
                continue   # geometry says same place, text says different book
            score = ov * vy
            if score > best_score:
                best, best_score = t, score
        return best

    def merge_across_segments(self) -> list[tuple[str, str]]:
        """Merge legible tracks with near-identical text that live in different segments."""
        merged = []
        keep: list[Track] = []
        for t in sorted(self.spines, key=lambda t: t.segment):
            b = t.best_spine()
            dup = None
            if b.legible:
                for k in keep:
                    if k.segment != t.segment:
                        kb = k.best_spine()
                        if kb.legible and text_sim(kb.text, b.text) > 0.88:
                            dup = k
                            break
            if dup:
                dup.obs.extend(t.obs)
                merged.append((t.id, dup.id))
            else:
                keep.append(t)
        self.spines = keep
        return merged

    def live_counts(self) -> dict:
        legible = sum(1 for t in self.spines if t.best_spine().legible)
        return {"spines": len(self.spines), "legible": legible, "unreadable": len(self.spines) - legible,
                "items": len(self.items), "shelf_units": len(self.units), "segments": self.segment + 1}


def _sane(H: np.ndarray) -> bool:
    """Reject degenerate homographies (mirror, extreme scale or perspective)."""
    if not np.isfinite(H).all():
        return False
    H = H / H[2, 2]
    det = np.linalg.det(H[:2, :2])
    return 0.25 < det < 4.0 and abs(H[2, 0]) < 2e-3 and abs(H[2, 1]) < 2e-3


def _avg_box(a, b, n):
    return tuple((a[i] * (n - 1) + b[i]) / n for i in range(4))


def _union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
