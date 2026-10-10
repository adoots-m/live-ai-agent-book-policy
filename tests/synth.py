"""Synthetic bookshelf scenes with exact ground truth, for testing the geometry
stages (tracking, de-duplication, marker scale, measurement) without a model."""

from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np

from library_claim.measure.scale import _DICT
from library_claim.schemas import FrameVision, ItemObs, ShelfUnitObs, SpineObs

PX_PER_CM = 8.0


@dataclass
class GTBook:
    id: str
    x: float          # cm, left of spine face
    y: float          # cm, top
    w: float          # cm on canvas (thickness if vertical)
    h: float          # cm on canvas (height if vertical)
    orientation: str
    title: str
    legible: bool

    @property
    def box_px(self):
        return (self.x * PX_PER_CM, self.y * PX_PER_CM, (self.x + self.w) * PX_PER_CM, (self.y + self.h) * PX_PER_CM)

    @property
    def height_cm(self):
        return self.h if self.orientation == "vertical" else self.w

    @property
    def thickness_cm(self):
        return self.w if self.orientation == "vertical" else self.h


WORDS = ["river", "night", "garden", "empire", "silent", "atlas", "winter", "glass", "history", "ocean", "stone",
         "letters", "kingdom", "harvest", "shadow", "machine", "orchard", "compass", "lantern", "meridian"]


def make_scene(seed: int = 1, shelves: int = 3, width_cm: float = 110, marker_cm: float = 15.0):
    rnd = random.Random(seed)
    rng = np.random.default_rng(seed)
    W, H = int((width_cm + 20) * PX_PER_CM), int((shelves * 36 + 30) * PX_PER_CM)
    # textured wall so ORB has features everywhere
    img = (rng.integers(90, 170, (H // 8, W // 8, 3))).astype(np.uint8)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    books: list[GTBook] = []
    unit_box = (8 * PX_PER_CM, 8 * PX_PER_CM, (width_cm + 12) * PX_PER_CM, (shelves * 36 + 12) * PX_PER_CM)
    cv2.rectangle(img, (int(unit_box[0]), int(unit_box[1])), (int(unit_box[2]), int(unit_box[3])), (40, 60, 90), 12)
    n = 0
    for s in range(shelves):
        base = 10 + (s + 1) * 36           # cm, shelf board top
        cv2.line(img, (int(10 * PX_PER_CM), int(base * PX_PER_CM)), (int((width_cm + 10) * PX_PER_CM), int(base * PX_PER_CM)), (30, 40, 60), 10)
        x = 12.0
        while x < width_cm - 6:
            n += 1
            if rnd.random() < 0.08 and x < width_cm - 30:      # a horizontal stack of 3
                L = rnd.uniform(20, 26)
                yb = base
                for k in range(3):
                    t = rnd.uniform(2.0, 3.5)
                    title = f"{rnd.choice(WORDS)} {rnd.choice(WORDS)}"
                    books.append(GTBook(f"g{n}-{k}", x, yb - t, L, t, "horizontal", title, rnd.random() < 0.8))
                    yb -= t
                x += L + 1.0
                continue
            h = rnd.uniform(17, 29)
            t = rnd.uniform(1.6, 5.0)
            title = f"{rnd.choice(WORDS)} {rnd.choice(WORDS)}"
            books.append(GTBook(f"g{n}", x, base - h, t, h, "vertical", title, rnd.random() < 0.8))
            x += t + rnd.uniform(0.05, 0.4)
    for b in books:
        x0, y0, x1, y1 = [int(v) for v in b.box_px]
        col = tuple(int(c) for c in rng.integers(20, 235, 3))
        cv2.rectangle(img, (x0, y0), (x1, y1), col, -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), (15, 15, 15), 1)
        # a few stripes so each spine has its own features
        for k in range(3):
            yy = int(y0 + (y1 - y0) * (0.2 + 0.25 * k))
            cv2.line(img, (x0 + 2, yy), (x1 - 2, yy), tuple(255 - c for c in col), 2)
    # marker leaning on the middle shelf books
    side = int(marker_cm * PX_PER_CM)
    mk = cv2.aruco.generateImageMarker(_DICT, 3, side)
    mx, my = int(40 * PX_PER_CM), int((10 + 2 * 36 - marker_cm - 1) * PX_PER_CM)
    pad = int(1.5 * PX_PER_CM)
    img[my - pad:my + side + pad, mx - pad:mx + side + pad] = 255
    img[my:my + side, mx:mx + side] = cv2.cvtColor(mk, cv2.COLOR_GRAY2BGR)
    # drop books hidden behind the marker card from ground truth: they are occluded
    mbox = (mx - pad, my - pad, mx + side + pad, my + side + pad)
    vis = []
    for b in books:
        bx = b.box_px
        ix = max(0, min(bx[2], mbox[2]) - max(bx[0], mbox[0])) * max(0, min(bx[3], mbox[3]) - max(bx[1], mbox[1]))
        if ix < 0.5 * (bx[2] - bx[0]) * (bx[3] - bx[1]):
            vis.append(b)
    return img, vis, unit_box


def views(scene: np.ndarray, n: int = 8, out=(1280, 900), seed: int = 0):
    """Pan across the scene left to right with mild perspective. Returns list of (img, H scene->view)."""
    rng = np.random.default_rng(seed)
    H_s, W_s = scene.shape[:2]
    vw, vh = out
    zoom = vh / H_s * 1.05
    span = W_s * zoom - vw
    res = []
    for i in range(n):
        tx = -span * i / max(1, n - 1)
        ty = -(H_s * zoom - vh) / 2
        H = np.array([[zoom, 0, tx], [0, zoom, ty], [0, 0, 1]], dtype=np.float64)
        persp = np.eye(3)
        persp[2, 0] = rng.uniform(-4e-5, 4e-5)
        persp[0, 1] = rng.uniform(-0.02, 0.02)
        Hf = persp @ H
        img = cv2.warpPerspective(scene, Hf, (vw, vh), borderValue=(120, 120, 120))
        res.append((img, Hf))
    return res


def project(H, box):
    x0, y0, x1, y1 = box
    pts = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).reshape(-1, 1, 2)
    q = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())


def mock_vision(frame_id: str, H, books: list[GTBook], unit_box, size, jitter: float = 1.5, seed: int = 0,
                drop_rate: float = 0.0) -> FrameVision:
    """What a perfect-ish reader would return for a view: GT boxes projected, jittered, clipped."""
    rng = np.random.default_rng(seed)
    vw, vh = size
    fv = FrameVision(frame_id=frame_id)
    for b in books:
        x0, y0, x1, y1 = project(H, b.box_px)
        full = (x1 - x0) * (y1 - y0)
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(vw, x1), min(vh, y1)
        if cx1 <= cx0 or cy1 <= cy0 or (cx1 - cx0) * (cy1 - cy0) < 0.5 * full:
            continue
        if rng.random() < drop_rate:
            continue
        j = rng.normal(0, jitter, 4)
        box = (max(0, cx0 + j[0]), max(0, cy0 + j[1]), min(vw, cx1 + j[2]), min(vh, cy1 + j[3]))
        flags = ["cut_at_edge"] if (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1) else []
        fv.spines.append(SpineObs(obs_id=f"{frame_id}-{b.id}", frame_id=frame_id, box=box, orientation=b.orientation,
                                  text=b.title.upper() if b.legible else "", title_read=b.title.upper() if b.legible else "",
                                  legible=b.legible, flags=flags))
    ub = project(H, unit_box)
    fv.shelf_units.append(ShelfUnitObs(obs_id=f"{frame_id}-u", frame_id=frame_id,
                                       box=(max(0, ub[0]), max(0, ub[1]), min(vw, ub[2]), min(vh, ub[3])), rows=3))
    fv.items.append(ItemObs(obs_id=f"{frame_id}-i", frame_id=frame_id, box=(max(0, ub[0]), max(0, ub[1]), min(vw, ub[2]), min(vh, ub[3])),
                            category="shelving", description="dark wood bookcase"))
    return fv
