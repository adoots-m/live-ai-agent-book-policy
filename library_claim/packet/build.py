"""Assemble claim_packet.json from the stage outputs. Deterministic: no model calls."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import SETTINGS
from ..measure.room import M2_TO_FT2, RoomFit
from ..schemas import (Book, ClaimPacket, Dimensions, Item, ItemCost, Money, ReviewEntry, Room, Sweep,
                       UsedMoney)
from ..vision.tracker import box_iou
from .totals import compute_totals

if TYPE_CHECKING:
    from ..sweep import SweepSession

RARITY = {"signed", "leather", "antique", "first_edition_marked", "gilt"}
ANTIQUARIAN_BEFORE = 1950
LOW_ID_CONF = 0.85


def build_packet(s: "SweepSession", *, measures, rows, units, merged, excluded_books, excluded_items,
                 room_fit: RoomFit | None, room_info: dict, item_groups, item_prices, corrections) -> ClaimPacket:
    loc = s.locale
    review: list[ReviewEntry] = []
    books: list[Book] = []
    excluded_meta = []

    def rv(ref, reason, kind):
        review.append(ReviewEntry(ref_id=ref, reason=reason, kind=kind))

    ordered = sorted(s.tracker.spines, key=lambda t: (rows.get(t.id, ("~", 0))[0], rows.get(t.id, ("", 0))[1]))
    for t in ordered:
        b = t.best_spine()
        refs = list(dict.fromkeys(s.store.frame_ref(o.frame_id) for o in t.obs))
        if t.id in excluded_books:
            excluded_meta.append({"id": t.id, "frame_ref": s.store.frame_ref(b.frame_id), "spine_text": b.text,
                                  "reason": "policyholder said these are not theirs"})
            continue
        m = measures.get(t.id)
        shelf, pos = rows.get(t.id, ("", 0))
        book = Book(id=t.id, shelf=shelf, position=pos, frame_ref=s.store.frame_ref(b.frame_id), frame_refs=refs,
                    frame_box=tuple(round(v, 1) for v in b.box), spine_text=b.text, orientation=b.orientation,
                    spine_height_cm=m.height_cm if m else None, spine_thickness_cm=m.thickness_cm if m else None,
                    scale_method=m.method if m else "")
        if m and m.frame_id:
            book.measure_frame_ref = s.store.frame_ref(m.frame_id)
        for issue in (m.issues if m else []):
            rv(t.id, issue, "measurement")
        flags = set(s.tracker_flags(t.id))
        book.flags = sorted(flags)
        r = s.ids.get(t.id)
        if r is not None and r.accepted:
            book.status = "identified"
            book.title, book.author, book.edition, book.publisher, book.isbn = r.title, r.author, r.edition, r.publisher, r.isbn
            book.id_confidence, book.id_source = round(r.confidence, 2), r.source_url
            if r.confidence < LOW_ID_CONF:
                rv(t.id, f"identification confidence {r.confidence:.2f}: spine '{b.text[:60]}' -> '{r.title}'", "identification")
            if r.year and r.year < ANTIQUARIAN_BEFORE:
                flags.add("antique")
                book.flags = sorted(flags)
        else:
            book.status = "unidentified"
            if b.legible:
                rv(t.id, f"legible spine '{b.text[:60]}' not confidently matched: {r.reason if r else 'not looked up'}", "identification")
            else:
                n = len(t.obs)
                rv(t.id, f"spine unreadable in {n} view{'s' if n != 1 else ''}; logged with dimensions only"
                   + (f" (partial text '{b.text[:40]}')" if b.text else ""), "identification")
        for c in [c for c in s.corrections if c.kind == "note" and t.id in (c.value or [])]:
            book.user_notes.append(c.text)
        rare = flags & RARITY
        if rare:
            book.status = "needs_appraisal"
            book.excluded_reason = "rarity indicators: " + ", ".join(sorted(rare))
            rv(t.id, f"possible rare/collectible ({', '.join(sorted(rare))}); not auto-priced", "appraisal")
        elif book.status == "identified":
            p = s.prices.get(s._price_key(t.id))
            if p is None:
                book.excluded_reason = "pricing did not complete"
                rv(t.id, "pricing did not complete; excluded from totals", "pricing")
            elif p.appraisal_reason:
                book.status = "needs_appraisal"
                book.excluded_reason = p.appraisal_reason
                rv(t.id, p.appraisal_reason + "; not auto-priced", "appraisal")
            else:
                book.replacement_cost = p.replacement if p.replacement.amount is not None else Money(currency=loc.currency)
                book.used_value = p.used if p.used.amount is not None else UsedMoney(currency=loc.currency)
                for msg in p.review:
                    rv(t.id, msg, "pricing")
                if book.replacement_cost.amount is None and book.used_value.amount is None:
                    book.excluded_reason = "no verifiable price found"
                    rv(t.id, "no verifiable price in any market; excluded from totals", "pricing")
                elif book.replacement_cost.amount is None:
                    rv(t.id, "no verifiable NEW price; only used value counted", "pricing")
                elif book.used_value.amount is None:
                    rv(t.id, "no verifiable USED price; only replacement cost counted", "pricing")
        if book.status == "unidentified":
            book.excluded_reason = "unidentified: not priced"
        books.append(book)

    # ------------------------------------------------------------------ items
    by_id = {t.id: t for t in s.tracker.items}
    items: list[Item] = []
    for g in item_groups:
        members = [m for m in g if m in by_id]
        if not members:
            continue
        gid = members[0]
        obs = [o for m in members for o in by_id[m].obs]
        best = max(obs, key=lambda o: (o.box[2] - o.box[0]) * (o.box[3] - o.box[1]))
        if any(m in excluded_items for m in members):
            excluded_meta.append({"id": gid, "frame_ref": s.store.frame_ref(best.frame_id), "description": best.description,
                                  "reason": "policyholder said this is not theirs"})
            continue
        brand = next((o.brand_model for o in obs if o.brand_model), "")
        it = Item(id=gid, category=best.category, description=best.description, brand_model=brand, material=best.material,
                  frame_ref=s.store.frame_ref(best.frame_id), frame_refs=list(dict.fromkeys(s.store.frame_ref(o.frame_id) for o in obs)),
                  frame_box=tuple(round(v, 1) for v in best.box))
        w, h = s.item_dims(members)
        it.dimensions_cm = Dimensions(w=w, h=h, d=None, method="box x marker-corrected monocular depth" if w else "")
        if best.category == "shelving":
            u = _matching_unit(s, by_id[gid], units)
            if u and u.get("width_cm"):
                it.dimensions_cm = Dimensions(w=u["width_cm"], h=u["height_cm"], d=None, method="aruco_plane (bookcase face)")
        if not w and best.category != "shelving":
            rv(gid, "item dimensions unknown (no depth for this view)", "measurement")
        status, cost, notes = item_prices.get(gid, ("unpriced", ItemCost(currency=loc.currency), ["pricing did not run"]))
        it.status, it.replacement_cost = status, cost
        is_art = any(o.is_art for o in obs)
        conf = 0.5 + 0.2 * (len(obs) > 1) + 0.2 * bool(brand) + 0.1 * (status in ("priced", "range"))
        it.confidence = round(min(conf, 0.95) if not is_art else 0.5, 2)
        if status == "needs_appraisal":
            it.excluded_reason = "art/portrait: needs appraisal"
        elif status == "unpriced":
            it.excluded_reason = "no verifiable price"
        for n in notes:
            rv(gid, n, "appraisal" if status == "needs_appraisal" else "pricing")
        if len(g) > 1:
            it.user_notes.append(f"merged sightings {', '.join(g)} (same object seen in different parts of the sweep)")
        items.append(it)

    # ------------------------------------------------------------------ room
    room = Room()
    shelved = [u for u in units if u.get("width_cm") and u.get("height_cm")]
    if shelved:
        a = sum(u["width_cm"] * u["height_cm"] for u in shelved) / 1e4
        room.shelved_wall_area_m2 = round(a, 2)
        room.shelved_wall_area_ft2 = round(a * M2_TO_FT2, 1)
    if len(shelved) < len(units):
        rv("room", f"{len(units) - len(shelved)} shelving unit(s) without metric scale; shelved wall area is a lower bound", "room")
    room.shelving_units = units
    if room_fit is not None:
        room.length_m, room.width_m, room.height_m = room_fit.length_m, room_fit.width_m, room_fit.height_m
        room.floor_area_m2 = room_fit.floor_area_m2
        room.perimeter_m = room_fit.perimeter_m
        room.shape, room.shape_note, room.polygon_m = room_fit.shape, room_fit.shape_note, room_fit.polygon_m
        if room.floor_area_m2:
            room.floor_area_ft2 = round(room.floor_area_m2 * M2_TO_FT2, 1)
        if room_fit.perimeter_m and room_fit.height_m:
            wa = room_fit.perimeter_m * room_fit.height_m
            room.wall_area_m2 = round(wa, 2)
            room.wall_area_ft2 = round(wa * M2_TO_FT2, 1)
            room.shape_note += " Wall area is gross (perimeter x height), no deduction for doors/windows."
        room.scale_method = room_info.get("scale_method", "") + f"; up axis: {room_info.get('up_source', '')}"
        room.confidence = room_fit.confidence
        kfs = s.room.kfs if s.room else []
        step = max(1, len(kfs) // 12)
        room.frame_refs = [s.store.frame_ref(k.frame_id) for k in kfs[::step]]
        for issue in room_fit.issues:
            rv("room", issue, "room")
    else:
        rv("room", "room geometry not computed: " + room_info.get("disabled", "unknown reason"), "room")

    for fid in s.tracker.segment_breaks:
        rv(f"frame:{fid}", f"camera tracking restarted at {s.store.frame_ref(fid)}; unreadable spines either side of this "
           "point may be counted twice", "capture")
    if not s.markers:
        rv("scale", "reference marker never detected; spine sizes rely on catalogue dimensions only", "measurement")

    totals = compute_totals(books, items, loc.currency, _shelf_run(s, measures, excluded_books))
    vision_tok = s.ledger.tokens_in.get("vision_read", 0)
    packet = ClaimPacket(
        sweep=Sweep(id=s.id, captured_at=s.started_at, device=s.device, duration_s=s.duration_s,
                    country=loc.country, currency=loc.currency),
        room=room, books=books, items=items, totals=totals, review_queue=review,
        meta={
            "models": {"live": SETTINGS.live_model, "vision": SETTINGS.vision_model, "pricing": SETTINGS.pricing_model,
                       "depth": SETTINGS.depth_model if s.room else None},
            "marker_side_cm": SETTINGS.marker_side_cm,
            "keyframes": len(s.tracker.order), "frames": len(s.store.order), "segments": s.tracker.segment + 1,
            "segment_breaks": s.tracker.segment_breaks, "merged_across_segments": merged,
            "marker_frames": sorted(s.markers), "excluded_by_policyholder": excluded_meta,
            "corrections": corrections, "room_stage": room_info, "vision_tokens_in": vision_tok,
            "appraisal_threshold": loc.appraisal_threshold,
            "unknown_value_policy": "null = the system does not know; never a default",
        })
    return packet


def _shelf_run(s, measures, excluded):
    from ..measure.spines import shelf_run_m
    keep = [t for t in s.tracker.spines if t.id not in excluded]
    orig = s.tracker.spines
    s.tracker.spines = keep
    try:
        return shelf_run_m(s.tracker, measures)
    finally:
        s.tracker.spines = orig


def _matching_unit(s, item_track, units):
    best, best_iou = None, 0.2
    for u in units:
        ut = next((t for t in s.tracker.units if t.id == u["id"]), None)
        if ut and ut.segment == item_track.segment:
            iou = box_iou(ut.mosaic_box, item_track.mosaic_box)
            if iou > best_iou:
                best, best_iou = u, iou
    return best
