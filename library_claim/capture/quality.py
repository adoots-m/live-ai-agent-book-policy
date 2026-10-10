"""Per-frame capture quality. Pure OpenCV, no model calls, ~5 ms per frame.

These numbers drive the agent's "slow down / step closer / tilt away from the
lamp" prompts, so they must be cheap enough to run on every frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..schemas import FrameQuality

# Thresholds were picked on phone footage at 640px working width; tune with
# scripts/tune_quality.py on your own frames.
BLUR_VAR = 60.0          # variance of Laplacian below this = blurry
GLARE_FRAC = 0.06        # >6 % of pixels saturated = glare patch
DARK_FRAC = 0.45         # >45 % of pixels near black = too dark
FAST_MOTION_PX = 45.0    # median feature shift between consecutive frames at 640px


def working_grey(img: np.ndarray, width: int = 640) -> np.ndarray:
    grey = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = grey.shape[:2]
    if w != width:
        grey = cv2.resize(grey, (width, max(1, int(h * width / w))), interpolation=cv2.INTER_AREA)
    return grey


def assess(img: np.ndarray, motion_px: float | None = None) -> FrameQuality:
    grey = working_grey(img)
    sharp = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    glare = float((grey >= 250).mean())
    dark = float((grey <= 25).mean())
    issues: list[str] = []
    if sharp < BLUR_VAR:
        issues.append("blur")
    if glare > GLARE_FRAC:
        issues.append("glare")
    if dark > DARK_FRAC:
        issues.append("dark")
    if motion_px is not None and motion_px > FAST_MOTION_PX:
        issues.append("fast")
    return FrameQuality(sharpness=round(sharp, 1), glare_frac=round(glare, 4), dark_frac=round(dark, 4),
                        motion_px=None if motion_px is None else round(motion_px, 1), issues=issues)


_ORB = cv2.ORB_create(nfeatures=800)
_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def features(grey: np.ndarray):
    return _ORB.detectAndCompute(grey, None)


def match_homography(feat_a, feat_b, min_matches: int = 25):
    """Homography mapping points in A to B, plus median displacement and inlier count."""
    kp_a, des_a = feat_a
    kp_b, des_b = feat_b
    if des_a is None or des_b is None or len(kp_a) < min_matches or len(kp_b) < min_matches:
        return None, None, 0
    matches = _MATCHER.match(des_a, des_b)
    if len(matches) < min_matches:
        return None, None, 0
    pa = np.float32([kp_a[m.queryIdx].pt for m in matches])
    pb = np.float32([kp_b[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0)
    if H is None:
        return None, None, 0
    inl = mask.ravel().astype(bool)
    if inl.sum() < min_matches:
        return None, None, int(inl.sum())
    disp = float(np.median(np.linalg.norm(pa[inl] - pb[inl], axis=1)))
    return H, disp, int(inl.sum())


def overlap_fraction(H: np.ndarray, w: int, h: int) -> float:
    """Share of frame A's area that lands inside frame B after warping by H."""
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    rect = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    inter, _ = cv2.intersectConvexConvex(warped.astype(np.float32), rect)
    return float(max(0.0, min(1.0, inter / float(w * h))))
