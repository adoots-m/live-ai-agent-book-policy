"""Metric scale from a printed ArUco marker.

The policyholder leans one printed marker (scripts/make_marker.py, default 15 cm
square) against the book spines of each bookcase before the sweep. Its four
corners give an exact homography from image pixels to centimetres *on the
plane of the spines*, which removes perspective as well as scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_PARAMS = cv2.aruco.DetectorParameters()
_PARAMS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
_DETECTOR = cv2.aruco.ArucoDetector(_DICT, _PARAMS)


@dataclass
class MarkerHit:
    frame_id: str
    marker_id: int
    corners: np.ndarray            # 4x2 float32, TL TR BR BL, full-res pixels
    side_px: float

    @property
    def area_px(self) -> float:
        return float(cv2.contourArea(self.corners.astype(np.float32)))


def detect_markers(img: np.ndarray, frame_id: str) -> list[MarkerHit]:
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    corners, ids, _ = _DETECTOR.detectMarkers(grey)
    hits = []
    if ids is None:
        return hits
    for c, i in zip(corners, ids.ravel()):
        q = c.reshape(4, 2).astype(np.float32)
        side = float(np.mean([np.linalg.norm(q[k] - q[(k + 1) % 4]) for k in range(4)]))
        if side >= 18:  # below ~18 px the corner error dominates
            hits.append(MarkerHit(frame_id, int(i), q, side))
    return hits


def plane_from_marker(hit: MarkerHit, side_cm: float) -> np.ndarray:
    """Homography: image pixels -> centimetres on the marker plane (origin at marker TL)."""
    dst = np.float32([[0, 0], [side_cm, 0], [side_cm, side_cm], [0, side_cm]])
    return cv2.getPerspectiveTransform(hit.corners.astype(np.float32), dst)


def quad_dims_cm(P: np.ndarray, quad_px: np.ndarray) -> tuple[float, float]:
    """Width and height (cm) of an image quad TL TR BR BL mapped through P."""
    q = cv2.perspectiveTransform(quad_px.reshape(-1, 1, 2).astype(np.float32), P).reshape(-1, 2)
    w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    return float(w), float(h)


def render_marker_png(marker_id: int = 0, px: int = 1200) -> np.ndarray:
    img = cv2.aruco.generateImageMarker(_DICT, marker_id, px)
    border = px // 15   # ~1 cm white quiet zone at 15 cm; fits A4/Letter at 300 dpi
    return cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT, value=255)
