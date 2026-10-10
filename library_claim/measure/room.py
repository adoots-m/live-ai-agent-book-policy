"""Stage: room dimensions, floor area and wall area from the same sweep.

No LiDAR in a browser, so:

1. **Metric depth per keyframe** from a pretrained monocular model
   (Depth-Anything-V2 Metric-Indoor, ~25M params, CPU-friendly).
2. **Depth scale correction** from the ArUco marker: wherever the marker is
   visible, its known size gives the true distance (pinhole model,
   ``z = fx * side / side_px``); the ratio to the model's depth at the marker
   rescales every depth map in the sweep (median over sightings).
3. **Camera motion** by RGB-D odometry between consecutive keyframes:
   ORB matches lifted to 3-D with the depth maps, rigid transform by RANSAC
   + Kabsch.
4. **Gravity** from the phone accelerometer (sent with every frame) defines
   "up", so the floor and ceiling are horizontal planes and walls vertical.
5. **Room fit**: floor = low height percentile, ceiling = high percentile
   (only if enough ceiling was seen), walls = outermost points projected on
   the floor plane. A rectangle is fitted with ``minAreaRect`` on trimmed
   extents; if the occupied outline fills < 90 % of that rectangle the
   outline polygon (L-shape etc.) is used instead and the report says so.

Each step is a function you can test on synthetic point clouds
(tests/test_room.py).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger(__name__)
M2_TO_FT2 = 10.7639
DEPTH_W = 518


# ------------------------------------------------------------------ depth backend
class DepthModel:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self._pipe = None

    def available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline("depth-estimation", model=self.model_id, device=-1)
        from PIL import Image
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out = self._pipe(Image.fromarray(rgb))
        d = out["predicted_depth"]
        d = d.squeeze().cpu().numpy() if hasattr(d, "cpu") else np.asarray(d)
        return cv2.resize(d.astype(np.float32), (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)


# ------------------------------------------------------------------ geometry helpers
def intrinsics(w: int, h: int, hfov_deg: float) -> np.ndarray:
    fx = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
    return np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], dtype=np.float64)


def backproject(depth: np.ndarray, K: np.ndarray, stride: int = 6, max_d: float = 8.0) -> np.ndarray:
    h, w = depth.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[vs, us]
    ok = (z > 0.2) & (z < max_d)
    us, vs, z = us[ok], vs[ok], z[ok]
    x = (us - K[0, 2]) * z / K[0, 0]
    y = (vs - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1)


def kabsch(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R, t minimising |R a + t - b|."""
    ca, cb = a.mean(0), b.mean(0)
    Hm = (a - ca).T @ (b - cb)
    U, _, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def ransac_rigid(a: np.ndarray, b: np.ndarray, iters: int = 300, thresh: float = 0.08, rng=None):
    rng = rng or np.random.default_rng(0)
    n = len(a)
    if n < 8:
        return None, None, 0
    best = (None, None, 0, None)
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        R, t = kabsch(a[idx], b[idx])
        err = np.linalg.norm((a @ R.T + t) - b, axis=1)
        inl = err < thresh
        if inl.sum() > best[2]:
            best = (R, t, int(inl.sum()), inl)
    if best[2] < 8:
        return None, None, best[2]
    R, t = kabsch(a[best[3]], b[best[3]])
    return R, t, best[2]


def up_from_imu(imu: dict | None, screen_angle: float = 0.0) -> np.ndarray | None:
    """Gravity 'up' in OpenCV camera coordinates (x right, y down, z forward)
    for the rear camera. ``accelerationIncludingGravity`` at rest points up in
    device coordinates (x right, y top of phone, z out of the screen)."""
    if not imu or imu.get("gx") is None:
        return None
    g = np.array([imu["gx"], imu["gy"], imu["gz"]], dtype=np.float64)
    n = np.linalg.norm(g)
    if n < 5:
        return None
    g /= n
    a = math.radians(screen_angle or 0.0)
    # rotate device x/y into the displayed (video) orientation
    gx = g[0] * math.cos(a) + g[1] * math.sin(a)
    gy = -g[0] * math.sin(a) + g[1] * math.cos(a)
    return np.array([gx, -gy, -g[2]])


def fit_plane_height(h: np.ndarray, low: bool, band: float = 0.05) -> tuple[float | None, int]:
    """Level of the floor (low=True) or ceiling along 'up'.

    A horizontal surface is a thin, dense band of heights; walls spread points
    evenly over height. Histogram heights in 2 cm bins; a bin is a *surface*
    if it holds >= 3x the median bin count (and >= 30 points). The floor is the
    lowest surface, the ceiling the highest. A shelf top or table also forms a
    band, which is why we take the extreme one, and why the ceiling must also
    be at least 1.9 m above the floor (checked by the caller).
    Returns (level, supporting points) or (None, 0)."""
    if len(h) < 50:
        return None, 0
    lo, hi = np.percentile(h, 0.1), np.percentile(h, 99.9)
    bins = np.arange(lo - 0.04, hi + 0.06, 0.02)
    counts, edges = np.histogram(h, bins=bins)
    if len(counts) < 5:
        return None, 0
    sm = np.convolve(counts, [0.25, 0.5, 0.25], mode="same")
    n = len(sm)
    order = range(n) if low else range(n - 1, -1, -1)
    step = 1 if low else -1
    k = None
    for i in order:
        a, b = (i + 3, i + 23) if low else (i - 22, i - 2)
        inward = sm[max(0, a):max(0, min(n, b))]
        if len(inward) < 5:
            break
        if sm[i] >= 30 and sm[i] >= 2.0 * max(1.0, float(np.median(inward))):
            k = i
            break
    if k is None:
        return None, 0
    while 0 <= k + step < n and sm[k + step] > sm[k]:
        k += step
    level = float((edges[k] + edges[k + 1]) / 2)
    support = int(((h > level - band) & (h < level + band)).sum())
    return level, support


def wall_position(x: np.ndarray, low: bool, search: float = 0.6, bin_w: float = 0.02) -> float:
    """Position of the outermost wall along one axis: the outermost density peak
    (>= 30 % of the strongest) within ``search`` metres of the extreme. A wall is
    a dense sheet of points; furniture and bookcase faces in front of it are
    further in, and depth noise forms thin tails outside it."""
    ext = np.percentile(x, 0.5 if low else 99.5)
    lo, hi = (ext - 0.15, ext + search) if low else (ext - search, ext + 0.15)
    sel = x[(x >= lo) & (x <= hi)]
    if len(sel) < 20:
        return float(ext)
    counts, edges = np.histogram(sel, bins=np.arange(lo, hi + bin_w, bin_w))
    sm = np.convolve(counts, [0.25, 0.5, 0.25], mode="same")
    thr = 0.3 * sm.max()
    idx = np.nonzero(sm >= thr)[0]
    k = idx[0] if low else idx[-1]
    # climb to the local maximum of that peak
    step = 1 if low else -1
    while 0 <= k + step < len(sm) and sm[k + step] >= sm[k]:
        k += step
    return float((edges[k] + edges[k + 1]) / 2)


@dataclass
class RoomFit:
    length_m: float | None = None
    width_m: float | None = None
    height_m: float | None = None
    floor_area_m2: float | None = None
    perimeter_m: float | None = None
    shape: str = ""
    shape_note: str = ""
    polygon_m: list = field(default_factory=list)
    confidence: float = 0.0
    issues: list[str] = field(default_factory=list)
    coverage: float = 0.0


def fit_room(points: np.ndarray, up: np.ndarray, cam_positions: np.ndarray | None = None,
             cell: float = 0.05) -> RoomFit:
    """Fit floor/ceiling/walls to a world point cloud with known 'up'."""
    fit = RoomFit()
    if len(points) < 500:
        fit.issues.append("too few 3-D points to fit the room")
        return fit
    up = up / np.linalg.norm(up)
    h = points @ up
    floor, fs = fit_plane_height(h, low=True)
    ceil, cs = fit_plane_height(h, low=False)
    if floor is None:
        fit.issues.append("floor not seen")
        return fit
    height = None
    if ceil is not None and cs >= 200 and 1.9 < ceil - floor < 6.0:
        height = ceil - floor
    else:
        fit.issues.append("ceiling not seen clearly; height unknown")
    # 2-D basis on the floor plane
    a = np.cross(up, [1.0, 0, 0])
    if np.linalg.norm(a) < 0.1:
        a = np.cross(up, [0, 0, 1.0])
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    top = (ceil - 0.25) if height else floor + 2.0
    wall = points[(h > floor + 0.3) & (h < top)]
    if len(wall) < 300:
        fit.issues.append("walls not seen")
        return fit
    xy = np.stack([wall @ a, wall @ b], axis=1).astype(np.float32)
    # occupancy grid, drop isolated cells (depth noise)
    mn = xy.min(0) - 0.5
    ij = ((xy - mn) / cell).astype(int)
    W, Hh = ij.max(0) + 2
    grid = np.zeros((Hh, W), np.uint16)
    np.add.at(grid, (ij[:, 1], ij[:, 0]), 1)
    occ = (grid >= 3).astype(np.uint8)
    # drop isolated cells (keep a cell if >= 2 of its 8 neighbours are occupied);
    # unlike a morphological opening this keeps 1-cell-thin wall lines
    nb = cv2.filter2D(occ, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - occ
    occ = ((occ > 0) & (nb >= 2)).astype(np.uint8)
    ys, xs = np.nonzero(occ)
    if len(xs) < 50:
        fit.issues.append("walls too sparse")
        return fit
    pts = (np.stack([xs, ys], 1).astype(np.float32) + 0.5) * cell + mn
    # rectangle from trimmed extents along the min-area-rect axes
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
    th = math.radians(ang)
    Rm = np.array([[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]])
    loc = (xy - [cx, cy]) @ Rm.T          # raw wall points, not grid cells
    lo = np.array([wall_position(loc[:, 0], True), wall_position(loc[:, 1], True)])
    hi = np.array([wall_position(loc[:, 0], False), wall_position(loc[:, 1], False)])
    dims = hi - lo
    L, Wd = float(max(dims)), float(min(dims))
    rect_area = L * Wd
    # outline polygon: close gaps, fill, largest contour
    k = max(3, int(0.6 / cell))
    closed = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    filled = closed.copy()
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    poly_area, poly, poly_per = None, [], None
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(filled, [c], -1, 1, thickness=-1)
        approx = cv2.approxPolyDP(c, 0.15 / cell, True).reshape(-1, 2).astype(np.float32)
        outer = ((approx + 0.5) * cell + mn).astype(np.float32)
        outer_area = float(cv2.contourArea(outer))
        outer_per = float(cv2.arcLength(outer, True))
        # The traced outline is the OUTSIDE of the wall band (wall + depth noise).
        # Band thickness = occupied area / outline length; the wall surface is the
        # band's centre, so shrink area by half a band along the perimeter.
        band = float(occ.sum()) * cell * cell / max(outer_per, 1e-6)
        poly_area = outer_area - outer_per * band / 2
        poly_per = outer_per - 4 * band       # offsetting an orthogonal polygon inward by d shortens it by 8d
        poly = outer.tolist()
    # angular coverage of walls around the room centre (needed to trust an outline)
    centre = np.array([cx, cy])
    angs = np.degrees(np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])) % 360
    fit.coverage = round(len(np.unique((angs // 10).astype(int))) / 36, 2)
    # Use the traced outline only if it is plausibly a closed room outline: an
    # L/T-shape removes at most ~40 % of its bounding rectangle, and the walls
    # must have been seen nearly all the way round.
    if poly_area and 0.6 * rect_area <= poly_area < 0.9 * rect_area and len(poly) >= 5 and fit.coverage >= 0.85:
        fit.shape = "orthogonal_polygon"
        fit.floor_area_m2 = round(poly_area, 2)
        fit.perimeter_m = round(poly_per, 2)
        fit.polygon_m = [(round(x, 2), round(y, 2)) for x, y in poly]
        fit.shape_note = (f"Outline fills {poly_area / rect_area:.0%} of the bounding rectangle, so the "
                          f"traced outline polygon ({len(poly)} vertices) is used for floor area and perimeter; "
                          f"length/width are the bounding rectangle.")
    else:
        fit.shape = "rectangle"
        fit.floor_area_m2 = round(rect_area, 2)
        fit.perimeter_m = round(2 * (L + Wd), 2)
        corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]) @ Rm + [cx, cy]
        fit.polygon_m = [(round(x, 2), round(y, 2)) for x, y in corners]
        fit.shape_note = ("Rectangle: axes from the minimum-area rectangle of wall points; each wall placed at the "
                          "outermost point-density peak (furniture in front of a wall is ignored).")
    fit.length_m, fit.width_m = round(L, 2), round(Wd, 2)
    fit.height_m = round(height, 2) if height else None
    if fit.coverage < 0.75:
        fit.issues.append(f"only {fit.coverage:.0%} of the wall directions were seen")
    fit.confidence = round(min(1.0, fit.coverage) * (1.0 if height else 0.6), 2)
    return fit


# ------------------------------------------------------------------ incremental builder
@dataclass
class _KF:
    frame_id: str
    K: np.ndarray
    depth: np.ndarray
    kp: list
    des: np.ndarray | None
    T: np.ndarray                  # camera -> world (4x4)
    up_cam: np.ndarray | None
    linked: bool


class RoomBuilder:
    """Accumulates depth + pose per keyframe during the sweep; fit at the end."""

    def __init__(self, depth_model: DepthModel, hfov_deg: float, marker_side_cm: float):
        self.depth_model = depth_model
        self.hfov = hfov_deg
        self.side_m = marker_side_cm / 100
        self.kfs: list[_KF] = []
        self.scale_samples: list[float] = []
        self.orb = cv2.ORB_create(1500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.broken_links = 0

    def add(self, frame_id: str, bgr: np.ndarray, imu: dict | None, marker_hits=None) -> None:
        s = DEPTH_W / bgr.shape[1]
        small = cv2.resize(bgr, (DEPTH_W, int(bgr.shape[0] * s)), interpolation=cv2.INTER_AREA)
        depth = self.depth_model(small)
        K = intrinsics(small.shape[1], small.shape[0], self.hfov)
        for hit in marker_hits or []:
            side_px = hit.side_px * s
            z_true = K[0, 0] * self.side_m / side_px
            c = (hit.corners * s).mean(0).astype(int)
            patch = depth[max(0, c[1] - 3):c[1] + 4, max(0, c[0] - 3):c[0] + 4]
            if patch.size and np.median(patch) > 0.1:
                self.scale_samples.append(float(z_true / np.median(patch)))
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(grey, None)
        up = up_from_imu(imu, (imu or {}).get("screen_angle", 0.0))
        T = np.eye(4)
        linked = False
        if self.kfs:
            prev = self.kfs[-1]
            rel = self._odometry(prev, kp, des, depth, K)
            if rel is not None:
                T = prev.T @ rel
                linked = True
            else:
                self.broken_links += 1
                T = prev.T.copy()   # assume the user mostly rotated; flagged in confidence
                if up is not None and prev.up_cam is not None:
                    T[:3, :3] = prev.T[:3, :3] @ _rot_between(up, prev.up_cam)
        self.kfs.append(_KF(frame_id, K, depth, kp, des, T, up, linked))

    def _odometry(self, prev: _KF, kp, des, depth, K):
        if des is None or prev.des is None or len(kp) < 30 or len(prev.kp) < 30:
            return None
        knn = self.bf.knnMatch(des, prev.des, k=2)
        good = [m for m, *r in knn if r and m.distance < 0.75 * r[0].distance]
        if len(good) < 20:
            return None
        uv1 = np.float32([kp[m.queryIdx].pt for m in good])
        uv0 = np.float32([prev.kp[m.trainIdx].pt for m in good])
        z1 = _bilinear(depth, uv1)
        z0 = _bilinear(prev.depth, uv0)
        ok = (z1 > 0.2) & (z1 < 8) & (z0 > 0.2) & (z0 < 8)
        # reject matches on depth discontinuities (bilinear depth is meaningless there)
        ok &= (_local_range(depth, uv1) < 0.05 * z1) & (_local_range(prev.depth, uv0) < 0.05 * z0)
        uv1, uv0, z1, z0 = uv1[ok], uv0[ok], z1[ok], z0[ok]
        a = np.c_[(uv1[:, 0] - K[0, 2]) * z1 / K[0, 0], (uv1[:, 1] - K[1, 2]) * z1 / K[1, 1], z1]
        b = np.c_[(uv0[:, 0] - prev.K[0, 2]) * z0 / prev.K[0, 0], (uv0[:, 1] - prev.K[1, 2]) * z0 / prev.K[1, 1], z0]
        if len(a) < 15:
            return None
        R, t, inl = ransac_rigid(a, b, iters=500, thresh=0.03 + 0.015 * float(np.median(a[:, 2])))
        if R is None or inl < 12 or np.linalg.norm(t) > 2.0:
            return None
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, t
        return T

    def item_dims_cm(self, frame_id: str, box_full, full_width: int) -> tuple[float | None, float | None]:
        """Approximate width/height of an object from its box and the median
        (marker-corrected) depth inside the box."""
        kf = next((k for k in self.kfs if k.frame_id == frame_id), None)
        if kf is None:
            return None, None
        s = kf.depth.shape[1] / full_width
        x0, y0, x1, y1 = [int(v * s) for v in box_full]
        patch = kf.depth[max(0, y0):y1, max(0, x0):x1]
        if patch.size < 4:
            return None, None
        z = float(np.median(patch)) * self.depth_scale()[0]
        fx = kf.K[0, 0]
        return round((x1 - x0) * z / fx * 100, 0), round((y1 - y0) * z / fx * 100, 0)

    def depth_scale(self) -> tuple[float, str]:
        if len(self.scale_samples) >= 1:
            return float(np.median(self.scale_samples)), f"marker-corrected monocular depth (n={len(self.scale_samples)})"
        return 1.0, "uncorrected monocular metric depth (marker never visible to the room stage)"

    def fit(self) -> tuple[RoomFit, dict]:
        info = {"keyframes": len(self.kfs), "broken_links": self.broken_links}
        if len(self.kfs) < 5:
            f = RoomFit()
            f.issues.append("fewer than 5 keyframes for the room stage")
            return f, info
        scale, method = self.depth_scale()
        info["depth_scale"] = round(scale, 3)
        info["scale_method"] = method
        clouds, ups, cams = [], [], []
        ref_up = next((kf.T[:3, :3] @ kf.up_cam for kf in self.kfs if kf.up_cam is not None), None)
        for kf in self.kfs:
            T = kf.T.copy()
            T[:3, 3] *= scale
            R = T[:3, :3]
            if ref_up is not None and kf.up_cam is not None:
                # The accelerometer gives this frame's absolute 'up'. Rotate the
                # odometry pose so it agrees: removes accumulated pitch/roll drift.
                R = _rot_between(R @ kf.up_cam, ref_up) @ R
                ups.append(R @ kf.up_cam)
            pts = backproject(kf.depth * scale, kf.K, stride=8)
            clouds.append(pts @ R.T + T[:3, 3])
            cams.append(T[:3, 3])
        cloud = np.concatenate(clouds)
        if ups:
            up = np.mean(ups, axis=0)
            info["up_source"] = "accelerometer (per-frame tilt correction)"
        else:
            up = np.array([0, -1.0, 0])   # camera y points down when phone is upright
            info["up_source"] = "assumed upright camera (no accelerometer data)"
        fit = fit_room(cloud, up, np.array(cams))
        link_ratio = 1 - self.broken_links / max(1, len(self.kfs) - 1)
        fit.confidence = round(fit.confidence * (0.5 + 0.5 * link_ratio) * (1.0 if self.scale_samples else 0.7), 2)
        if self.broken_links:
            fit.issues.append(f"camera tracking lost {self.broken_links} times; poses there assume pure rotation")
        return fit, info


def _bilinear(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    if len(uv) == 0:
        return np.zeros(0, np.float32)
    m = cv2.remap(img, uv[:, 0].reshape(-1, 1), uv[:, 1].reshape(-1, 1), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return m.ravel()


def _local_range(img: np.ndarray, uv: np.ndarray, r: int = 2) -> np.ndarray:
    """max - min of depth in a (2r+1)^2 window around each point."""
    mx = cv2.dilate(img, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
    mn = cv2.erode(img, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
    return _bilinear(mx, uv) - _bilinear(mn, uv)


def _rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation taking unit vector a onto b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))
