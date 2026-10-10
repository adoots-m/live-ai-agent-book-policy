"""Snap a reader's spine box to the actual spine edges in the image.

Vision-model boxes are good at *which* spine, but a few pixels off at the
edges. On a 2 cm spine seen at 7 px/cm that is a 20-30 % thickness error. We
search a small window around each long edge for the strongest gradient
profile (spines are separated by dark gaps or colour changes) and move the
edge there. Height edges are refined the same way along the other axis.
"""

from __future__ import annotations

import cv2
import numpy as np


def _best_edge(profile: np.ndarray, centre: int, radius: int) -> int:
    lo, hi = max(0, centre - radius), min(len(profile), centre + radius + 1)
    if hi - lo < 3:
        return centre
    seg = profile[lo:hi]
    k = int(np.argmax(seg))
    # only move if the peak is clearly stronger than what is at the original edge
    if seg[k] < 1.5 * (profile[min(max(centre, 0), len(profile) - 1)] + 1e-6):
        return centre
    return lo + k


def refine_box(grey: np.ndarray, box, orientation: str = "vertical") -> tuple[float, float, float, float]:
    h, w = grey.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w - 1, x1), min(h - 1, y1)
    bw, bh = x1 - x0, y1 - y0
    if bw < 4 or bh < 4:
        return box
    gx = np.abs(cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3))
    # profile across x, using the middle 60 % of the box height (avoid shelf boards / tops)
    ya, yb = y0 + int(0.2 * bh), y1 - int(0.2 * bh)
    px = gx[ya:yb].mean(axis=0)
    rx = max(2, int(0.15 * bw)) if orientation == "vertical" else max(2, int(0.06 * bw))
    nx0 = _best_edge(px, x0, rx)
    nx1 = _best_edge(px, x1, rx)
    xa, xb = x0 + int(0.2 * bw), x1 - int(0.2 * bw)
    py = gy[:, xa:xb].mean(axis=1) if xb > xa else gy[:, x0:x1 + 1].mean(axis=1)
    ry = max(2, int(0.03 * bh)) if orientation == "vertical" else max(2, int(0.25 * bh))
    ny0 = _best_edge(py, y0, ry)
    ny1 = _best_edge(py, y1, ry)
    if nx1 - nx0 < 0.5 * bw or ny1 - ny0 < 0.5 * bh:
        return box
    return float(nx0), float(ny0), float(nx1), float(ny1)
