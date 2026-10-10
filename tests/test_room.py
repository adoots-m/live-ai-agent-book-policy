"""Room stage: plane/wall fitting on synthetic clouds, and the full RoomBuilder
(odometry + depth + gravity) on a ray-traced textured box room."""

import math

import cv2
import numpy as np

from library_claim.measure.room import RoomBuilder, fit_room, kabsch, ransac_rigid, up_from_imu

RNG = np.random.default_rng(0)


def _inside(poly, q):
    c = np.array(poly, np.float32).reshape(-1, 1, 2)
    return np.array([cv2.pointPolygonTest(c, (float(x), float(y)), False) >= 0 for x, y in q])


def room_cloud(poly, H, n=50000, ceiling=True, noise=0.03):
    pts = []
    P = np.array(poly + [poly[0]], float)
    per = sum(np.linalg.norm(P[i + 1] - P[i]) for i in range(len(poly)))
    for i in range(len(poly)):
        a, b = P[i], P[i + 1]
        k = int(n * np.linalg.norm(b - a) / per)
        t, z = RNG.random(k), RNG.random(k) * H
        xy = a + (b - a) * t[:, None]
        pts.append(np.c_[xy[:, 0], z, xy[:, 1]])
    mn, mx = np.min(poly, 0), np.max(poly, 0)
    for lvl in ([0, H] if ceiling else [0]):
        q = mn + (mx - mn) * RNG.random((20000, 2))
        q = q[_inside(poly, q)]
        pts.append(np.c_[q[:, 0], np.full(len(q), lvl), q[:, 1]])
    pts.append(np.c_[1 + RNG.random(3000) * 0.8, RNG.random(3000) * 0.9, 1 + RNG.random(3000) * 0.5])  # a sofa
    c = np.concatenate(pts)
    return c + RNG.normal(0, noise, c.shape)


UP = np.array([0, 1.0, 0])


def test_rectangle_room():
    r = fit_room(room_cloud([(0, 0), (4.2, 0), (4.2, 3.6), (0, 3.6)], 2.5), UP)
    assert r.shape == "rectangle"
    assert abs(r.length_m - 4.2) < 0.1 and abs(r.width_m - 3.6) < 0.1
    assert abs(r.height_m - 2.5) < 0.05
    assert abs(r.floor_area_m2 - 15.12) / 15.12 < 0.05


def test_l_shaped_room_uses_polygon():
    r = fit_room(room_cloud([(0, 0), (5, 0), (5, 2), (2.5, 2), (2.5, 4), (0, 4)], 2.6), UP)
    assert r.shape == "orthogonal_polygon"
    assert abs(r.floor_area_m2 - 15.0) / 15.0 < 0.08
    assert "polygon" in r.shape_note


def test_missing_ceiling_is_unknown_not_guessed():
    r = fit_room(room_cloud([(0, 0), (4.2, 0), (4.2, 3.6), (0, 3.6)], 2.5, ceiling=False), UP)
    assert r.height_m is None
    assert any("ceiling" in i for i in r.issues)


def test_kabsch_and_ransac_recover_motion():
    a = RNG.random((200, 3)) * 3
    th = math.radians(20)
    R = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]])
    t = np.array([0.3, 0.0, -0.2])
    b = a @ R.T + t
    b[:30] += RNG.random((30, 3))  # outliers
    R2, t2, inl = ransac_rigid(a, b)
    assert inl >= 160 and np.allclose(R2, R, atol=1e-2) and np.allclose(t2, t, atol=2e-2)
    R3, t3 = kabsch(a[30:], b[30:])
    assert np.allclose(R3, R, atol=1e-6)


def test_up_from_imu_portrait():
    up = up_from_imu({"gx": 0.0, "gy": 9.8, "gz": 0.0})
    assert np.allclose(up, [0, -1, 0])


# ----------------------------------------------------------------- ray-traced room
L, W, HGT = 4.0, 3.2, 2.5      # x, z extents and ceiling height; floor y=0


def _texture(p):
    """Procedural texture on world points (5 cm cells, hashed)."""
    q = np.floor(p / 0.05).astype(np.int64)
    h = (q[..., 0] * 73856093) ^ (q[..., 1] * 19349663) ^ (q[..., 2] * 83492791)
    return ((h & 0xFF).astype(np.uint8))


def render(cam_pos, yaw, pitch, w=518, h=388, hfov=67.0):
    fx = (w / 2) / math.tan(math.radians(hfov) / 2)
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    d_cam = np.stack([(us - w / 2) / fx, (vs - h / 2) / fx, np.ones_like(us, float)], -1)
    # camera -> world: OpenCV cam (x right, y down, z fwd) -> world (y up)
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    Rp = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    flip = np.diag([1, -1, 1])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    R = Ry @ flip @ Rp
    d = d_cam @ R.T
    o = np.asarray(cam_pos, float)
    ts = []
    for axis, lo, hi in ((0, 0, L), (1, 0, HGT), (2, 0, W)):
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - o[axis]) / d[..., axis]
            t2 = (hi - o[axis]) / d[..., axis]
        ts.append(np.where(t1 > 0, t1, np.inf))
        ts.append(np.where(t2 > 0, t2, np.inf))
    t = np.min(np.stack(ts), 0)
    p = o + d * t[..., None]
    img = _texture(p)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    depth = (t * d_cam[..., 2]).astype(np.float32)   # z-depth
    up_world = np.array([0, 1.0, 0])
    up_cam = R.T @ up_world
    return img, depth, up_cam


class FakeDepth:
    def __init__(self):
        self.next = None

    def __call__(self, img):
        return self.next


def test_room_builder_on_rendered_room():
    fd = FakeDepth()
    rb = RoomBuilder(fd, 67.0, 15.0)
    n = 72
    for i in range(n):
        yaw = 2 * math.pi * i / n
        pitch = 0.5 * math.sin(6 * math.pi * i / n)   # tilt smoothly up to the ceiling and down to the floor
        pos = (2.0 + 0.3 * math.cos(yaw), 1.4, 1.6 + 0.3 * math.sin(yaw))
        img, depth, up_cam = render(pos, yaw, pitch)
        fd.next = depth
        imu = {"gx": up_cam[0] * 9.8, "gy": -up_cam[1] * 9.8, "gz": -up_cam[2] * 9.8}
        rb.add(f"f{i:05d}", img, imu)
    fit, info = rb.fit()
    assert fit.length_m and abs(fit.length_m - L) / L < 0.1, (fit, info)
    assert abs(fit.width_m - W) / W < 0.1, (fit, info)
    assert fit.height_m and abs(fit.height_m - HGT) / HGT < 0.08, (fit, info)
    assert abs(fit.floor_area_m2 - L * W) / (L * W) < 0.10


def test_marker_corrects_biased_depth():
    """Depth model 20 % too far; marker sightings rescale it."""
    from library_claim.measure.room import intrinsics
    from library_claim.measure.scale import MarkerHit

    class Biased(FakeDepth):
        def __call__(self, img):
            return self.next * 1.2

    fd = Biased()
    rb = RoomBuilder(fd, 67.0, 15.0)
    n = 72
    K = intrinsics(518, 388, 67.0)
    for i in range(n):
        yaw = 2 * math.pi * i / n
        pitch = 0.5 * math.sin(6 * math.pi * i / n)
        pos = (2.0 + 0.3 * math.cos(yaw), 1.4, 1.6 + 0.3 * math.sin(yaw))
        img, depth, up_cam = render(pos, yaw, pitch)
        fd.next = depth
        hits = []
        if i % 9 == 0:
            z = float(depth[194, 259])
            side = K[0, 0] * 0.15 / z
            c = np.float32([[259 - side / 2, 194 - side / 2], [259 + side / 2, 194 - side / 2],
                            [259 + side / 2, 194 + side / 2], [259 - side / 2, 194 + side / 2]])
            hits = [MarkerHit(f"f{i:05d}", 0, c, side)]
        rb.add(f"f{i:05d}", img, {"gx": up_cam[0] * 9.8, "gy": -up_cam[1] * 9.8, "gz": -up_cam[2] * 9.8}, hits)
    fit, info = rb.fit()
    assert abs(info["depth_scale"] - 1 / 1.2) < 0.02
    assert abs(fit.floor_area_m2 - L * W) / (L * W) < 0.10
