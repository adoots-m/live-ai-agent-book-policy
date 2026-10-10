"""End-to-end sweep on a synthetic bookcase with every model call mocked.

Checks the things the brief makes rules of: totals computed from lines,
every frame_ref opens, unreadable spines are not given titles, prices carry a
source and date, rare items are not auto-priced, corrections apply.
"""

import asyncio
import json

import cv2
import pytest

from synth import make_scene, mock_vision, views


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LCA_DATA_DIR", str(tmp_path))
    import importlib
    import library_claim.config as cfg
    importlib.reload(cfg)
    import library_claim.sweep as sweep_mod
    monkeypatch.setattr(sweep_mod, "SETTINGS", cfg.Settings(data_dir=tmp_path))
    import library_claim.packet.build as build_mod
    monkeypatch.setattr(build_mod, "SETTINGS", cfg.Settings(data_dir=tmp_path))
    return tmp_path, sweep_mod


def test_synthetic_sweep(env, monkeypatch):
    tmp, sweep_mod = env
    scene, books, unit_box = make_scene(5, width_cm=200)
    vs = views(scene, 14, seed=3)
    H_by_frame = {}

    async def fake_read(img, fid, ledger, tiles=None):
        H = H_by_frame[fid]
        fv = mock_vision(fid, H, books, unit_box, (img.shape[1], img.shape[0]), seed=int(fid[1:]))
        if fid == "f00000":     # a framed portrait appears once
            from library_claim.schemas import ItemObs
            fv.items.append(ItemObs(obs_id="p1", frame_id=fid, box=(10, 10, 200, 260), category="art",
                                    description="framed oil portrait of a woman", is_art=True))
        return fv

    from library_claim.identify.catalogue import Resolution

    async def fake_resolve(text, title, author, pub, country="US", measured_h=None):
        # pretend every 5th legible title is not in any catalogue
        if hash(title) % 5 == 0:
            return Resolution(False, 0.4, reason="no catalogue match")
        return Resolution(True, 0.93, title=title.title(), author="A. Writer", source_url="https://openlibrary.org/works/OL1W")

    from library_claim.pricing.books import BookPrice
    from library_claim.schemas import ItemCost, Money, UsedMoney

    async def fake_price(title, author, edition, isbn, loc, ledger, flags=None):
        if flags:
            return BookPrice(appraisal_reason="rarity indicators: " + ",".join(flags))
        return BookPrice(replacement=Money(amount=14.99, currency=loc.currency, source="Shop", url="https://shop/x",
                                           retrieved_at="2026-10-09T00:00:00+00:00", verification="page_text"),
                         used=UsedMoney(amount=5.5, currency=loc.currency, source="Used", url="https://used/x",
                                        retrieved_at="2026-10-09T00:00:00+00:00", verification="page_text",
                                        condition_assumed="Good"))

    async def fake_item(cat, desc, brand, mat, dims, loc, ledger, is_art, is_print):
        if is_art and not is_print:
            return "needs_appraisal", ItemCost(currency=loc.currency), ["art needs appraisal"]
        return "range", ItemCost(low=150.0, high=320.0, currency=loc.currency, source="2 offers", url="https://x",
                                 retrieved_at="2026-10-09"), []

    monkeypatch.setattr(sweep_mod, "read_frame", fake_read)
    monkeypatch.setattr(sweep_mod, "resolve", fake_resolve)
    monkeypatch.setattr(sweep_mod, "price_book", fake_price)
    monkeypatch.setattr(sweep_mod, "price_item", fake_item)

    async def run():
        s = sweep_mod.SweepSession(country="US", device="synthetic", use_depth=False)
        s.start()
        for i, (img, H) in enumerate(vs):
            H_by_frame[f"f{i:05d}"] = H
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            await s.add_frame(jpg.tobytes(), {"gx": 0, "gy": 9.8, "gz": 0}, t=i * 0.5)
            if i == 8:
                s.record_correction("signed", "book", "that one is signed")
        await s.finish()
        return s

    s = asyncio.run(run())
    p = json.loads((s.dir / "claim_packet.json").read_text())
    kf = s.tracker.order
    # every GT book visible in a keyframe should be counted once
    seen_gt = {o.obs_id.split("-", 1)[1] for t in s.tracker.spines for o in t.obs}
    assert abs(p["totals"]["book_count"] - len(seen_gt)) <= max(1, 0.05 * len(seen_gt)), (p["totals"]["book_count"], len(seen_gt))
    # totals add up from lines, computed in code
    assert p["meta"]["totals_check"] == []
    rc = round(sum(b["replacement_cost"]["amount"] or 0 for b in p["books"] if b["status"] == "identified"), 2)
    assert p["totals"]["books_replacement_cost"] == rc
    # frame refs open
    for b in p["books"]:
        assert (s.dir / b["frame_ref"].split("#")[0]).exists()
    # unreadable spines carry no title
    for b in p["books"]:
        if b["status"] == "unidentified":
            assert b["title"] == "" and b["replacement_cost"]["amount"] is None
    # identified + priced lines carry source + date
    for b in p["books"]:
        if b["replacement_cost"]["amount"] is not None:
            assert b["replacement_cost"]["url"] and b["replacement_cost"]["retrieved_at"]
    # the 'signed' correction produced a needs_appraisal line, not priced
    signed = [b for b in p["books"] if "signed" in b["flags"]]
    assert signed and all(b["status"] == "needs_appraisal" and b["replacement_cost"]["amount"] is None for b in signed)
    # art defaults to appraisal
    art = [i for i in p["items"] if i["category"] == "art"]
    assert art and art[0]["status"] == "needs_appraisal"
    # measurements exist via the marker for most books
    with_h = [b for b in p["books"] if b["spine_height_cm"]]
    assert len(with_h) > 0.6 * len(p["books"])
    assert all(b["scale_method"] == "aruco_plane" for b in with_h)
    # review queue explains unidentified lines
    assert any(r["kind"] == "identification" for r in p["review_queue"])
    assert (s.dir / "report.html").exists()
    assert list(s.dir.glob("claim_packet_*.zip"))
    # height accuracy against synthetic truth
    gt = {b.id: b for b in books}
    errs = []
    for t in s.tracker.spines:
        g = gt[sorted({o.obs_id.split('-', 1)[1] for o in t.obs})[0]]
        line = next((b for b in p["books"] if b["id"] == t.id), None)
        if line and line["spine_height_cm"]:
            errs.append(abs(line["spine_height_cm"] - g.height_cm) / g.height_cm)
    assert sorted(errs)[len(errs) // 2] < 0.05
    print("books", p["totals"]["book_count"], "keyframes", len(kf), "median h err", sorted(errs)[len(errs) // 2])
