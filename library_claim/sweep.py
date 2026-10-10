"""Sweep orchestrator: owns one sweep's state and runs the stages.

During the sweep (live):
    frame -> FrameStore (save + quality) -> keyframe? -> Tracker geometry
          -> marker detection -> RoomBuilder (depth, background)
          -> vision reader (async) -> Tracker association -> identification (async)
          -> pricing (async, as soon as a book is identified)
After the sweep (finish):
    drain queues -> cross-segment merge -> corrections -> measurement
    -> re-identify with measured heights -> pricing -> item consolidation + pricing
    -> room fit -> packet (totals in code) -> report

Each stage writes its own JSON under ``<sweep>/stages/`` so it can be inspected
and re-run on saved frames (``python -m library_claim.cli process <sweep_dir>``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import cv2
import numpy as np

from .capture.frames import FrameStore
from .config import SETTINGS, Locale, locale_for
from .identify.catalogue import Resolution, resolve
from .llm import Ledger, call_json
from .measure import spines as spine_measure
from .measure.room import DepthModel, RoomBuilder
from .measure.scale import MarkerHit, detect_markers
from .packet.build import build_packet
from .pricing.books import BookPrice, price_book
from .pricing.items import price_item
from .schemas import FrameVision
from .vision.reader import read_frame
from .vision.tracker import Tracker

log = logging.getLogger(__name__)
ROOM_MIN_GAP_S = 0.8   # depth model runs on at most ~1 keyframe per 0.8 s (CPU budget)
RARITY_FLAGS = {"signed", "leather", "antique", "first_edition_marked", "gilt"}

Notify = Callable[[dict], Awaitable[None]]


@dataclass
class Correction:
    kind: str           # not_mine | first_edition | signed | rare | is_print | is_original | known_dimension | note
    scope: str          # book | shelf | unit | item
    frame_id: str       # latest frame when the user spoke
    t: float
    text: str = ""
    value: Any = None


@dataclass
class SweepSession:
    country: str = field(default_factory=lambda: SETTINGS.default_country)
    id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4])
    device: str = ""
    notify: Notify | None = None
    use_depth: bool = True

    def __post_init__(self):
        self.locale: Locale = locale_for(self.country)
        self.dir: Path = SETTINGS.data_dir / self.id
        (self.dir / "stages" / "vision").mkdir(parents=True, exist_ok=True)
        self.store = FrameStore(self.dir)
        self.tracker = Tracker()
        self.ledger = Ledger()
        self.vision: dict[str, FrameVision] = {}
        self.markers: dict[str, list[MarkerHit]] = {}
        depth = DepthModel(SETTINGS.depth_model)
        self.room = RoomBuilder(depth, SETTINGS.camera_hfov_deg, SETTINGS.marker_side_cm) if (self.use_depth and depth.available()) else None
        self.room_disabled_reason = "" if self.room else "depth model unavailable (pip install -r requirements-depth.txt)"
        self.corrections: list[Correction] = []
        self.ids: dict[str, Resolution] = {}
        self._id_text: dict[str, str] = {}
        self.prices: dict[str, BookPrice] = {}
        self._price_tasks: dict[str, asyncio.Task] = {}
        self._tasks: set[asyncio.Task] = set()
        self._kf_queue: asyncio.Queue = asyncio.Queue()
        self._room_queue: asyncio.Queue = asyncio.Queue()
        self._sem_vision = asyncio.Semaphore(SETTINGS.vision_concurrency)
        self._sem_price = asyncio.Semaphore(SETTINGS.pricing_concurrency)
        self._lock = asyncio.Lock()
        self.t0: float | None = None
        self.started_at = ""
        self.ended_at_mono: float | None = None
        self.duration_s = 0.0
        self.asked_art: set[str] = set()
        self.art_answers: dict[str, bool] = {}   # item track id -> is_print
        self.events: list[dict] = []
        self.finished = False
        self.packet = None
        self._workers: list[asyncio.Task] = []
        self.last_frame_id = ""
        self._last_room_t = -1e9

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.t0 = time.monotonic()
        self.started_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        self._workers = [asyncio.create_task(self._keyframe_worker()), asyncio.create_task(self._room_worker())]

    def elapsed(self) -> float:
        return 0.0 if self.t0 is None else time.monotonic() - self.t0

    def set_country(self, country: str) -> Locale:
        self.locale = locale_for(country)
        self.country = self.locale.country
        return self.locale

    def _spawn(self, coro, name: str = "") -> asyncio.Task:
        t = asyncio.create_task(coro, name=name or None)
        self._tasks.add(t)
        t.add_done_callback(self._done)
        return t

    def _done(self, t: asyncio.Task) -> None:
        self._tasks.discard(t)
        if not t.cancelled() and t.exception():
            log.error("background task failed", exc_info=t.exception())

    async def _emit(self, payload: dict) -> None:
        if self.notify:
            try:
                await self.notify(payload)
            except Exception:
                log.debug("notify failed", exc_info=True)

    # ------------------------------------------------------------------ frames
    async def add_frame(self, jpeg: bytes, imu: dict | None = None, t: float | None = None):
        """Add one camera frame. ``t`` overrides the clock (offline replay of a recording)."""
        if self.t0 is None:
            self.start()
        rec = await asyncio.to_thread(self.store.add, jpeg, self.elapsed() if t is None else t, imu)
        if t is not None:
            self.duration_s = max(self.duration_s, t)
        self.last_frame_id = rec.id
        if rec.keyframe:
            await self._kf_queue.put(rec.id)
        return rec

    async def _keyframe_worker(self) -> None:
        while True:
            fid = await self._kf_queue.get()
            try:
                if fid is None:
                    return
                img = await asyncio.to_thread(self.store.load, fid)
                await asyncio.to_thread(self.tracker.add_keyframe, fid, img)
                hits = await asyncio.to_thread(detect_markers, img, fid)
                if hits:
                    self.markers[fid] = hits
                t_kf = self.store.frames[fid].t
                if self.room is not None and (hits or t_kf - self._last_room_t >= ROOM_MIN_GAP_S):
                    self._last_room_t = t_kf
                    await self._room_queue.put((fid, img, hits))
                self._spawn(self._vision(fid, img))
            except Exception:
                log.exception("keyframe %s failed", fid)
            finally:
                self._kf_queue.task_done()

    async def _room_worker(self) -> None:
        while True:
            job = await self._room_queue.get()
            try:
                if job is None:
                    return
                fid, img, hits = job
                imu = self.store.frames[fid].imu.model_dump()
                await asyncio.to_thread(self.room.add, fid, img, imu, hits)
            except Exception:
                log.exception("room stage failed on %s", job[0] if job else "?")
            finally:
                self._room_queue.task_done()

    async def _vision(self, fid: str, img: np.ndarray) -> None:
        cache = self.dir / "stages" / "vision" / f"{fid}.json"
        async with self._sem_vision:
            if cache.exists():
                fv = FrameVision.model_validate_json(cache.read_text(encoding="utf-8"))
            else:
                fv = await read_frame(img, fid, self.ledger)
                cache.write_text(fv.model_dump_json(indent=1), encoding="utf-8")
        async with self._lock:
            self.vision[fid] = fv
            self.tracker.add_vision(fv)
        for t in self.tracker.spines:
            b = t.best_spine()
            if b.legible and self._id_text.get(t.id) != b.text:
                self._id_text[t.id] = b.text
                self._spawn(self._identify(t.id, b))
        await self._emit({"type": "inventory", **self.live_inventory()})

    async def _identify(self, track_id: str, spine, measured_h: float | None = None) -> None:
        res = await resolve(spine.text, spine.title_read, spine.author_read, spine.publisher_read,
                            self.locale.country, measured_h)
        self.ids[track_id] = res
        if res.accepted:
            await self._emit({"type": "logged", "kind": "book", "id": track_id, "title": res.title, "author": res.author})
            self._schedule_price(track_id)

    def _price_key(self, track_id: str) -> str:
        r = self.ids[track_id]
        return f"{self.locale.country}|{r.isbn or (r.title + '|' + r.author).lower()}"

    def _schedule_price(self, track_id: str) -> None:
        key = self._price_key(track_id)
        if key in self._price_tasks:
            return
        r = self.ids[track_id]
        flags = sorted(set(self.tracker_flags(track_id)) & RARITY_FLAGS)

        async def job():
            async with self._sem_price:
                return await price_book(r.title, r.author, r.edition, r.isbn, self.locale, self.ledger, flags)
        self._price_tasks[key] = self._spawn(job(), name=f"price:{key}")

    def tracker_flags(self, track_id: str) -> list[str]:
        t = next((t for t in self.tracker.spines if t.id == track_id), None)
        flags = set()
        if t:
            for o in t.obs:
                flags.update(o.flags)
        for c in self.corrections:
            if c.kind in ("first_edition", "signed", "rare") and track_id in (c.value or []):
                flags.add({"first_edition": "first_edition_marked"}.get(c.kind, c.kind if c.kind != "rare" else "antique"))
        return sorted(flags)

    # ------------------------------------------------------------------ live feedback
    def capture_status(self, window_s: float = 2.0) -> dict:
        """Aggregate quality over the last ``window_s`` seconds for the agent."""
        now = self.elapsed()
        recent = [self.store.frames[i] for i in self.store.order[-20:] if now - self.store.frames[i].t <= window_s]
        issues: dict[str, int] = {}
        for r in recent:
            for k in (r.quality.issues if r.quality else []):
                issues[k] = issues.get(k, 0) + 1
        n = max(1, len(recent))
        problems = [k for k, v in issues.items() if v / n >= 0.6]
        latest_v = self.vision.get(max(self.vision, key=lambda k: self.store.frames[k].t)) if self.vision else None
        too_far = False
        if latest_v and latest_v.spines:
            w = self.store.frames[latest_v.frame_id].width
            thin = [s for s in latest_v.spines if (s.box[2] - s.box[0]) < 0.012 * w and s.orientation == "vertical"]
            too_far = latest_v.unreadable_fraction > 0.5 and len(thin) > 0.4 * len(latest_v.spines)
        seg = self.tracker.segment
        geom = dict(self.tracker.geom)   # snapshot: the keyframe worker thread may be inserting
        seg_has_marker = any(geom[f].segment == seg for f in list(self.markers) if f in geom)
        return {"t": round(now, 1), "frames": len(recent), "problems": problems, "too_far": too_far,
                "unreadable_fraction": latest_v.unreadable_fraction if latest_v else None,
                "reader_feedback": latest_v.capture_feedback if latest_v else "",
                "segment": seg, "segment_has_marker": seg_has_marker,
                "keyframes_in_segment": sum(1 for g in list(self.tracker.geom.values()) if g.segment == seg),
                "pending_keyframes": self._kf_queue.qsize(), **self.tracker.live_counts()}

    def missed_rows(self) -> bool:
        """True if the bookcase most recently seen reports more shelf rows than
        rows with any spines captured (and we have been on it for a while)."""
        seg = self.tracker.segment
        units = [u for u in self.tracker.units if u.segment == seg and len(u.obs) >= 4]
        if not units:
            return False
        u = max(units, key=lambda u: len(u.obs))
        expected = max((o.rows for o in u.obs), default=0)
        rows = spine_measure.assign_rows(self.tracker)
        got = {r for tid, (r, _) in rows.items() if r.startswith(u.id + "-")}
        return expected >= 2 and len(got) < expected

    def peek_art_questions(self) -> list[dict]:
        """Art items not yet asked about (the caller marks one asked when it asks)."""
        return [{"item_id": t.id, "description": t.obs[0].description} for t in self.tracker.items
                if t.obs[0].is_art and t.id not in self.asked_art and t.id not in self.art_answers]

    def live_inventory(self) -> dict:
        books = []
        for t in self.tracker.spines:
            b = t.best_spine()
            r = self.ids.get(t.id)
            books.append({"id": t.id, "title": r.title if r and r.accepted else "", "author": r.author if r and r.accepted else "",
                          "spine_text": b.text[:80], "legible": b.legible, "frame": b.frame_id, "orientation": b.orientation})
        items = [{"id": t.id, "category": t.obs[0].category, "description": t.obs[0].description, "frame": t.obs[0].frame_id}
                 for t in self.tracker.items]
        return {"counts": self.tracker.live_counts(), "books": books[-400:], "items": items}

    # ------------------------------------------------------------------ corrections
    def record_correction(self, kind: str, scope: str, text: str = "", value: Any = None) -> dict:
        c = Correction(kind=kind, scope=scope, frame_id=self.last_frame_id, t=round(self.elapsed(), 2), text=text, value=value)
        self.corrections.append(c)
        return {"recorded": True, "kind": kind, "scope": scope, "frame": c.frame_id,
                "note": "Applied when the packet is built, against the shelf/item in view when you spoke."}

    def answer_art(self, item_id: str, is_print: bool) -> dict:
        self.art_answers[item_id] = is_print
        return {"recorded": True, "item_id": item_id, "is_print": is_print}

    def _nearest_keyframe(self, frame_id: str) -> str | None:
        if not self.tracker.order:
            return None
        t = self.store.frames[frame_id].t if frame_id in self.store.frames else 0
        return min(self.tracker.order, key=lambda f: abs(self.store.frames[f].t - t))

    def _resolve_corrections(self) -> tuple[set[str], set[str], dict]:
        """Map each spoken correction onto concrete track ids using the frame in view."""
        excluded_books: set[str] = set()
        excluded_items: set[str] = set()
        applied = {}
        rows = spine_measure.assign_rows(self.tracker)
        for i, c in enumerate(self.corrections):
            kf = self._nearest_keyframe(c.frame_id)
            if kf is None:
                continue
            fr = self.store.frames[kf]
            cx, cy = fr.width / 2, fr.height / 2
            in_view = [t for t in self.tracker.spines if any(o.frame_id == kf for o in t.obs)]
            centre = min(in_view, key=lambda t: _dist_to(next(o for o in t.obs if o.frame_id == kf).box, cx, cy), default=None)
            targets: list[str] = []
            if c.scope == "book" and centre:
                targets = [centre.id]
            elif c.scope == "shelf" and centre:
                shelf = rows.get(centre.id, ("", 0))[0]
                targets = [tid for tid, (s, _) in rows.items() if s == shelf]
            elif c.scope == "unit" and centre:
                prefix = rows.get(centre.id, ("", 0))[0].split("-R")[0]
                targets = [tid for tid, (s, _) in rows.items() if s.startswith(prefix + "-")]
            elif c.scope == "item":
                items = [t for t in self.tracker.items if any(o.frame_id == kf for o in t.obs)]
                it = min(items, key=lambda t: _dist_to(next(o for o in t.obs if o.frame_id == kf).box, cx, cy), default=None)
                targets = [it.id] if it else []
            c.value = targets if c.kind != "known_dimension" else c.value
            applied[f"c{i}"] = {"kind": c.kind, "scope": c.scope, "frame": kf, "targets": targets, "said": c.text}
            if c.kind == "not_mine":
                (excluded_items if c.scope == "item" else excluded_books).update(targets)
            if c.kind in ("is_print", "is_original") and c.scope == "item":
                for t in targets:
                    self.art_answers[t] = c.kind == "is_print"
        return excluded_books, excluded_items, applied

    # ------------------------------------------------------------------ finish
    async def finish(self) -> Any:
        if self.finished:
            return self.packet
        self.ended_at_mono = time.monotonic()
        # live: wall-clock duration; offline replay: timestamp of the last frame
        self.duration_s = round(self.duration_s or self.elapsed(), 1)
        t_end = time.perf_counter()
        L = self.ledger
        with L.stage("drain_keyframes"):
            await self._kf_queue.join()
            await self._kf_queue.put(None)
            while True:  # vision tasks spawn identification tasks; loop until both are drained
                pending = [t for t in self._tasks if not t.done() and not t.get_name().startswith("price:")]
                if not pending:
                    break
                await asyncio.gather(*pending, return_exceptions=True)
        with L.stage("tracking_merge"):
            merged = self.tracker.merge_across_segments()
        excluded_books, excluded_items, applied = self._resolve_corrections()
        sizes = {f: (r.width, r.height) for f, r in self.store.frames.items()}
        stated = [c.value for c in self.corrections if c.kind == "known_dimension" and isinstance(c.value, dict)]
        with L.stage("measure_spines"):
            cat_h = {tid: r.height_cm for tid, r in self.ids.items() if r.accepted and r.height_cm}
            refine = (lambda fid: cv2.cvtColor(self.store.load(fid), cv2.COLOR_BGR2GRAY)) if os.getenv("LCA_EDGE_REFINE", "1") != "0" else None
            measures = await asyncio.to_thread(spine_measure.measure_spines, self.tracker, self.markers, SETTINGS.marker_side_cm,
                                               sizes, cat_h, stated, refine)
            rows = spine_measure.assign_rows(self.tracker)
            units = spine_measure.measure_units(self.tracker, self.markers, SETTINGS.marker_side_cm)
        with L.stage("identify"):
            jobs = []
            for t in self.tracker.spines:
                b = t.best_spine()
                if not b.legible:
                    continue
                r = self.ids.get(t.id)
                mh = measures[t.id].height_cm if t.id in measures else None
                if r is None or self._id_text.get(t.id) != b.text or (r.accepted and not r.isbn and mh):
                    self._id_text[t.id] = b.text
                    jobs.append(self._identify(t.id, b, mh))
            await asyncio.gather(*jobs, return_exceptions=True)
            for tid, r in self.ids.items():
                if r.accepted:
                    self._schedule_price(tid)
        with L.stage("price_books"):
            results = await asyncio.gather(*self._price_tasks.values(), return_exceptions=True)
            for key, res in zip(list(self._price_tasks), results):
                if isinstance(res, BookPrice):
                    self.prices[key] = res
        with L.stage("room_fit"):
            if self.room is not None:
                await self._room_queue.join()
                await self._room_queue.put(None)
                room_fit, room_info = await asyncio.to_thread(self.room.fit)
            else:
                room_fit, room_info = None, {"disabled": self.room_disabled_reason}
        with L.stage("consolidate_items"):
            groups = await self._consolidate_items()
        with L.stage("price_items"):
            item_prices = await self._price_items(groups, excluded_items)
        with L.stage("build_packet"):
            self.packet = build_packet(self, measures=measures, rows=rows, units=units, merged=merged,
                                       excluded_books=excluded_books, excluded_items=excluded_items,
                                       room_fit=room_fit, room_info=room_info, item_groups=groups,
                                       item_prices=item_prices, corrections=applied)
        self.packet.meta["time_to_packet_s"] = round(time.perf_counter() - t_end, 1)
        self.packet.meta["cost"] = L.report()
        self._write_stage("corrections", applied)
        from .packet.report import write_outputs
        write_outputs(self.packet, self.dir)
        self.finished = True
        for w in self._workers:
            w.cancel()
        await self._emit({"type": "packet_built", "sweep_id": self.id})
        return self.packet

    async def _consolidate_items(self) -> list[list[str]]:
        """Group item tracks that are the same physical object seen in different
        tracking segments. The model returns only groupings of ids we gave it."""
        tracks = self.tracker.items
        if len({t.segment for t in tracks}) <= 1 or len(tracks) < 2:
            return [[t.id] for t in tracks]
        listing = [{"id": t.id, "category": t.obs[0].category, "description": t.obs[0].description,
                    "segment": t.segment, "seen_at_s": round(self.store.frames[t.obs[0].frame_id].t, 1)} for t in tracks]
        prompt = ("These are object sightings from one continuous walk around ONE room. Sightings in different "
                  "'segment's may be the same physical object seen again. Group ids that are clearly the same object. "
                  "When unsure, keep them separate. Return JSON {\"groups\": [[id, ...], ...]} covering every id once.\n"
                  + json.dumps(listing))
        valid = {t.id for t in tracks}
        try:
            data = await call_json("consolidate_items", self.ledger, [prompt])
            groups = [[i for i in g if i in valid] for g in data.get("groups", [])]
            seen = {i for g in groups for i in g}
            groups = [g for g in groups if g] + [[i] for i in valid - seen]
            # an id in two groups -> keep first occurrence only
            used, clean = set(), []
            for g in groups:
                g2 = [i for i in g if i not in used]
                used.update(g2)
                if g2:
                    clean.append(g2)
            self._write_stage("item_groups", clean)
            return clean
        except Exception:
            log.exception("item consolidation failed; keeping tracks separate")
            return [[t.id] for t in tracks]

    async def _price_items(self, groups: list[list[str]], excluded: set[str]) -> dict[str, tuple]:
        by_id = {t.id: t for t in self.tracker.items}
        out: dict[str, tuple] = {}

        async def one(gid: str, members: list[str]):
            by_id[members[0]]
            o = max((ob for m in members for ob in by_id[m].obs), key=lambda ob: len(ob.brand_model) * 10 + len(ob.description))
            dims = self.item_dims(members)
            is_print = next((self.art_answers[m] for m in members if m in self.art_answers), None)
            async with self._sem_price:
                out[gid] = await price_item(o.category, o.description, o.brand_model, o.material,
                                            {"w": dims[0], "h": dims[1]}, self.locale, self.ledger, o.is_art, is_print)
        await asyncio.gather(*(one(g[0], g) for g in groups if g[0] not in excluded and not any(m in excluded for m in g)),
                             return_exceptions=True)
        return out

    def item_dims(self, members: list[str]) -> tuple[float | None, float | None]:
        if self.room is None:
            return None, None
        by_id = {t.id: t for t in self.tracker.items}
        ws, hs = [], []
        for m in members:
            for o in by_id[m].obs:
                w, h = self.room.item_dims_cm(o.frame_id, o.box, self.store.frames[o.frame_id].width)
                if w and h:
                    ws.append(w)
                    hs.append(h)
        if not ws:
            return None, None
        return float(np.median(ws)), float(np.median(hs))

    def summary(self) -> dict:
        p = self.packet
        if p is None:
            return {}
        t = p.totals
        r = p.room
        return {"books": t.book_count, "identified": t.books_identified, "unidentified": t.books_unidentified,
                "needs_appraisal": t.books_needs_appraisal, "shelf_run_m": t.shelf_run_m,
                "books_replacement": t.books_replacement_cost, "books_used": t.books_used_value,
                "items": len(p.items), "items_low": t.items_replacement_cost_low, "items_high": t.items_replacement_cost_high,
                "currency": t.currency, "floor_m2": r.floor_area_m2, "wall_m2": r.wall_area_m2,
                "review_items": len(p.review_queue), "excluded": t.excluded_from_totals,
                "time_to_packet_s": p.meta.get("time_to_packet_s")}

    def _write_stage(self, name: str, data: Any) -> None:
        (self.dir / "stages" / f"{name}.json").write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")


def _dist_to(box, cx, cy) -> float:
    return ((box[0] + box[2]) / 2 - cx) ** 2 + ((box[1] + box[3]) / 2 - cy) ** 2
