"""Stage: spine height / thickness in cm, shelf rows, and shelf run.

Scale sources, in order of preference (each result records which was used):

1. ``aruco_plane``      – marker visible somewhere in the same tracking segment.
                          frame px -(H_k)-> segment ref px -(inv H_m)-> marker frame px -(M)-> cm.
2. ``catalogue_dims``   – no marker, but other books in the same keyframe were
                          identified and their catalogue record lists a height
                          (Open Library ``physical_dimensions``). px/cm is the
                          median ratio over those books, leave-one-out.
3. ``stated_dimension`` – the policyholder said a bookcase dimension aloud
                          ("that bookcase is 180 tall"); px/cm from that unit's box.
4. none                 – dimensions left empty and the book goes to review.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import median

import numpy as np

from ..vision.tracker import Tracker, Track, project_quad
from .scale import MarkerHit, plane_from_marker, quad_dims_cm

PLAUSIBLE_H = (8.0, 60.0)      # cm; outside this a spine height is almost certainly a mis-read
PLAUSIBLE_T = (0.3, 15.0)
BORDER_PX = 4


@dataclass
class SpineMeasure:
    height_cm: float | None = None
    thickness_cm: float | None = None
    method: str = ""
    frame_id: str = ""
    issues: list[str] = field(default_factory=list)


def _touches_border(box, w, h) -> bool:
    return box[0] <= BORDER_PX or box[1] <= BORDER_PX or box[2] >= w - BORDER_PX or box[3] >= h - BORDER_PX


def _quad(box) -> np.ndarray:
    x0, y0, x1, y1 = box
    return np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def segment_planes(tracker: Tracker, markers: dict[str, list[MarkerHit]], side_cm: float) -> dict[int, np.ndarray]:
    """Per tracking segment: homography from segment-reference pixels to cm."""
    best: dict[int, MarkerHit] = {}
    for fid, hits in markers.items():
        g = tracker.geom.get(fid)
        if g is None:
            continue
        for h in hits:
            if g.segment not in best or h.area_px > best[g.segment].area_px:
                best[g.segment] = h
    planes = {}
    for seg, hit in best.items():
        g = tracker.geom[hit.frame_id]
        M = plane_from_marker(hit, side_cm)
        planes[seg] = M @ np.linalg.inv(g.H_to_ref)
    return planes


def _measure_with_plane(P_ref: np.ndarray, tracker: Tracker, obs) -> tuple[float, float]:
    g = tracker.geom[obs.frame_id]
    P = P_ref @ g.H_to_ref
    w, h = quad_dims_cm(P, _quad(obs.box))
    return w, h


def _orient(w_cm: float, h_cm: float, orientation: str) -> tuple[float, float]:
    # vertical book: box height = spine height, box width = thickness
    return (h_cm, w_cm) if orientation == "vertical" else (w_cm, h_cm)


def measure_spines(tracker: Tracker, markers: dict[str, list[MarkerHit]], side_cm: float,
                   frame_sizes: dict[str, tuple[int, int]],
                   catalogue_height_cm: dict[str, float] | None = None,
                   stated_units: list[dict] | None = None, load_grey=None) -> dict[str, SpineMeasure]:
    """``load_grey(frame_id) -> np.ndarray`` enables edge refinement (measure/refine.py)."""
    catalogue_height_cm = catalogue_height_cm or {}
    if load_grey is not None:
        _refine_all(tracker, load_grey, frame_sizes)
    planes = segment_planes(tracker, markers, side_cm)
    out: dict[str, SpineMeasure] = {}

    # Per-frame px/cm from catalogue heights (fallback 2), keyed by frame id.
    ratios: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for t in tracker.spines:
        known = catalogue_height_cm.get(t.id)
        if not known:
            continue
        for o in t.obs:
            w, h = frame_sizes[o.frame_id]
            if _touches_border(o.box, w, h):
                continue
            px = (o.box[3] - o.box[1]) if o.orientation == "vertical" else (o.box[2] - o.box[0])
            ratios[o.frame_id].append((t.id, px / known))

    # Per-frame px/cm from a stated bookcase dimension (fallback 3).
    stated_ratio: dict[str, float] = {}
    for st in stated_units or []:
        unit: Track | None = next((u for u in tracker.units if u.id == st.get("unit_id")), None)
        if unit is None and tracker.units:
            unit = max(tracker.units, key=lambda u: (u.mosaic_box[2] - u.mosaic_box[0]) * (u.mosaic_box[3] - u.mosaic_box[1]))
        if unit is None:
            continue
        for o in unit.obs:
            px = (o.box[3] - o.box[1]) if st.get("dimension") == "height" else (o.box[2] - o.box[0])
            stated_ratio[o.frame_id] = px / float(st["cm"])

    for t in tracker.spines:
        m = SpineMeasure()
        cands = sorted(t.obs, key=lambda o: _touches_border(o.box, *frame_sizes[o.frame_id]))
        clean = [o for o in cands if not _touches_border(o.box, *frame_sizes[o.frame_id])]
        if not clean:
            m.issues.append("spine cut by frame edge in every view")
        use = clean or cands
        if t.segment in planes:
            vals = [_orient(*_measure_with_plane(planes[t.segment], tracker, o), o.orientation) for o in use[:4]]
            m.height_cm = round(median(v[0] for v in vals), 1)
            m.thickness_cm = round(median(v[1] for v in vals), 1)
            m.method, m.frame_id = "aruco_plane", use[0].frame_id
        else:
            for o in use:
                rs = [r for tid, r in ratios.get(o.frame_id, []) if tid != t.id]
                if rs:
                    k = median(rs)
                    bw, bh = o.box[2] - o.box[0], o.box[3] - o.box[1]
                    h_cm, t_cm = _orient(bw / k, bh / k, o.orientation)
                    m.height_cm, m.thickness_cm = round(h_cm, 1), round(t_cm, 1)
                    m.method, m.frame_id = f"catalogue_dims(n={len(rs)})", o.frame_id
                    break
                if o.frame_id in stated_ratio:
                    k = stated_ratio[o.frame_id]
                    bw, bh = o.box[2] - o.box[0], o.box[3] - o.box[1]
                    h_cm, t_cm = _orient(bw / k, bh / k, o.orientation)
                    m.height_cm, m.thickness_cm = round(h_cm, 1), round(t_cm, 1)
                    m.method, m.frame_id = "stated_dimension", o.frame_id
                    break
        if m.height_cm is None:
            m.issues.append("no metric scale in this part of the sweep (marker not seen)")
        else:
            if not clean:
                m.height_cm = None  # the long axis is the one most likely to be clipped
            if m.height_cm is not None and not PLAUSIBLE_H[0] <= m.height_cm <= PLAUSIBLE_H[1]:
                m.issues.append(f"implausible spine height {m.height_cm} cm")
                m.height_cm = None
            if m.thickness_cm is not None and not PLAUSIBLE_T[0] <= m.thickness_cm <= PLAUSIBLE_T[1]:
                m.issues.append(f"implausible thickness {m.thickness_cm} cm")
                m.thickness_cm = None
        out[t.id] = m
    return out


def _refine_all(tracker: Tracker, load_grey, frame_sizes) -> None:
    """Replace each observation box by its edge-snapped version (in place, once)."""
    from .refine import refine_box
    by_frame: dict[str, list] = defaultdict(list)
    for t in tracker.spines:
        for o in t.obs:
            if not getattr(o, "_refined", False):
                by_frame[o.frame_id].append(o)
    for fid, obs in by_frame.items():
        try:
            grey = load_grey(fid)
        except Exception:
            continue
        for o in obs:
            o.box = refine_box(grey, o.box, o.orientation)
            object.__setattr__(o, "_refined", True)


def assign_rows(tracker: Tracker) -> dict[str, tuple[str, int]]:
    """Shelf label and left-to-right position for every spine track.

    Rows are found by clustering spine *bottoms* in mosaic coordinates: books on
    one shelf board share a baseline. Shelf label is U<unit>-R<row> where unit is
    the shelf-unit track containing the spine, or S<segment> if none."""
    result: dict[str, tuple[str, int]] = {}
    groups: dict[str, list[Track]] = defaultdict(list)
    for t in tracker.spines:
        cx = (t.mosaic_box[0] + t.mosaic_box[2]) / 2
        cy = (t.mosaic_box[1] + t.mosaic_box[3]) / 2
        unit = next((u for u in tracker.units if u.segment == t.segment and u.mosaic_box[0] <= cx <= u.mosaic_box[2]
                     and u.mosaic_box[1] <= cy <= u.mosaic_box[3]), None)
        groups[unit.id if unit else f"S{t.segment}"].append(t)
    for gname, tracks in groups.items():
        heights = [t.mosaic_box[3] - t.mosaic_box[1] for t in tracks if t.obs[0].orientation == "vertical"] or [50.0]
        gap = 0.35 * median(heights)
        tracks.sort(key=lambda t: t.mosaic_box[3])
        rows: list[list[Track]] = []
        for t in tracks:
            if rows and abs(t.mosaic_box[3] - median(r.mosaic_box[3] for r in rows[-1])) <= gap:
                rows[-1].append(t)
            else:
                rows.append([t])
        for ri, row in enumerate(rows):
            for pi, t in enumerate(sorted(row, key=lambda t: t.mosaic_box[0])):
                result[t.id] = (f"{gname}-R{ri + 1}", pi + 1)
    return result


def shelf_run_m(tracker: Tracker, measures: dict[str, SpineMeasure]) -> float | None:
    """Linear shelf length occupied: sum of upright thicknesses plus one footprint
    (widest book) per horizontal stack. None when nothing could be measured."""
    total, any_val = 0.0, False
    stacks: list[list[Track]] = []
    for t in tracker.spines:
        m = measures.get(t.id)
        if t.obs[0].orientation == "vertical":
            if m and m.thickness_cm:
                total += m.thickness_cm
                any_val = True
        else:
            for s in stacks:
                if s[0].segment == t.segment and min(s[0].mosaic_box[2], t.mosaic_box[2]) - max(s[0].mosaic_box[0], t.mosaic_box[0]) > 0:
                    s.append(t)
                    break
            else:
                stacks.append([t])
    for s in stacks:
        widths = [measures[t.id].height_cm for t in s if measures.get(t.id) and measures[t.id].height_cm]
        if widths:
            total += max(widths)
            any_val = True
    return round(total / 100, 2) if any_val else None


def measure_units(tracker: Tracker, markers: dict[str, list[MarkerHit]], side_cm: float) -> list[dict]:
    """Face width/height of each shelving unit in cm (for shelved wall area)."""
    planes = segment_planes(tracker, markers, side_cm)
    units = []
    for u in tracker.units:
        rec = {"id": u.id, "segment": u.segment, "width_cm": None, "height_cm": None, "method": "",
               "frame_id": max(u.obs, key=lambda o: (o.box[2] - o.box[0]) * (o.box[3] - o.box[1])).frame_id}
        if u.segment in planes:
            import cv2
            # Each edge = median over the views where that edge is not cut by the frame border.
            edges = {"l": [], "t": [], "r": [], "b": []}
            for o in u.obs:
                P = planes[u.segment] @ tracker.geom[o.frame_id].H_to_ref
                q = cv2.perspectiveTransform(_quad(o.box).reshape(-1, 1, 2), P).reshape(-1, 2)
                fw, fh = tracker.geom[o.frame_id].width, tracker.geom[o.frame_id].height
                if o.box[0] > BORDER_PX:
                    edges["l"].append(q[:, 0].min())
                if o.box[1] > BORDER_PX:
                    edges["t"].append(q[:, 1].min())
                if o.box[2] < fw - BORDER_PX:
                    edges["r"].append(q[:, 0].max())
                if o.box[3] < fh - BORDER_PX:
                    edges["b"].append(q[:, 1].max())
            if not all(edges.values()):
                rec["method"] = "unit edge cut by frame border in every view"
                units.append(rec)
                continue
            w = float(median(edges["r"]) - median(edges["l"]))
            h = float(median(edges["b"]) - median(edges["t"]))
            if 20 <= w <= 600 and 20 <= h <= 400:
                rec.update(width_cm=round(w, 1), height_cm=round(h, 1), method="aruco_plane")
        units.append(rec)
    return units


__all__ = ["SpineMeasure", "measure_spines", "assign_rows", "shelf_run_m", "measure_units", "project_quad"]
