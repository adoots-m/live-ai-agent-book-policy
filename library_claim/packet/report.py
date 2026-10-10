"""Write claim_packet.json, report.html, evidence crops and a zip per sweep."""

from __future__ import annotations

import html
import json
import zipfile
from pathlib import Path

import cv2

from ..schemas import ClaimPacket
from .totals import check_totals


def _crop(sweep_dir: Path, frame_ref: str, box, out: Path) -> bool:
    if not frame_ref or not box:
        return False
    img = cv2.imread(str(sweep_dir / frame_ref.split("#")[0]))
    if img is None:
        return False
    x0, y0, x1, y1 = [int(v) for v in box]
    pad = 6
    crop = img[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
    if crop.size == 0:
        return False
    h, w = crop.shape[:2]
    s = 160 / max(h, w)
    if s < 1:
        crop = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))))
    out.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out), crop))


def _money(m, sym="") -> str:
    if not m or m.get("amount") is None:
        return "<span class=na>—</span>"
    conv = f"<div class=small>converted: {html.escape(m.get('converted_from', ''))}</div>" if m.get("converted") else ""
    link = f"<a href='{html.escape(m.get('url', ''))}' target=_blank>{html.escape(m.get('source', '')[:50])}</a>"
    return (f"<b>{sym}{m['amount']:.2f}</b><div class=small>{link}<br>{html.escape(m.get('retrieved_at', '')[:10])} · "
            f"{html.escape(m.get('verification', ''))}{(' · ' + html.escape(m.get('condition_assumed', ''))) if m.get('condition_assumed') else ''}</div>{conv}")


def _num(v, unit="") -> str:
    return "<span class=na>—</span>" if v is None else f"{v}{unit}"


def render_html(p: dict, crops: dict[str, str]) -> str:
    e = html.escape
    t, r, sw = p["totals"], p["room"], p["sweep"]
    cur = sw["currency"]
    rows_b = []
    for b in p["books"]:
        crop = crops.get(b["id"])
        img = f"<a href='{e(b['frame_ref'].split('#')[0])}' target=_blank><img src='{e(crop)}'></a>" if crop else ""
        title = (f"<b>{e(b['title'])}</b><div class=small>{e(b['author'])}</div><div class=small>{e(b['edition'][:90])}"
                 f"{(' · ISBN ' + e(b['isbn'])) if b['isbn'] else ''}</div>") if b["title"] else f"<i>spine: {e(b['spine_text'][:70]) or 'unreadable'}</i>"
        rows_b.append(f"<tr class='{b['status']}'><td>{e(b['id'])}</td><td>{e(b['shelf'])}·{b['position']}</td><td>{img}</td>"
                      f"<td>{title}</td><td>{e(b['status'])}<div class=small>{e(b.get('excluded_reason', ''))}</div></td>"
                      f"<td>{b['id_confidence']:.2f}</td><td>{_num(b['spine_height_cm'])} × {_num(b['spine_thickness_cm'])}"
                      f"<div class=small>{e(b['scale_method'])}</div></td><td>{_money(b['replacement_cost'])}</td>"
                      f"<td>{_money(b['used_value'])}</td><td class=small><a href='{e(b['frame_ref'].split('#')[0])}'>{e(b['frame_ref'])}</a></td></tr>")
    rows_i = []
    for i in p["items"]:
        c = i["replacement_cost"]
        crop = crops.get(i["id"])
        img = f"<a href='{e(i['frame_ref'].split('#')[0])}' target=_blank><img src='{e(crop)}'></a>" if crop else ""
        price = ("<span class=na>—</span>" if c.get("low") is None else
                 f"<b>{c['low']:.2f}{'' if c['low'] == c['high'] else ' – ' + format(c['high'], '.2f')}</b>"
                 f"<div class=small><a href='{e(c.get('url', ''))}' target=_blank>{e(c.get('source', '')[:60])}</a> · {e(c.get('retrieved_at', '')[:10])}</div>")
        d = i["dimensions_cm"]
        rows_i.append(f"<tr class='{i['status']}'><td>{e(i['id'])}</td><td>{img}</td><td>{e(i['category'])}</td>"
                      f"<td>{e(i['description'])}<div class=small>{e(i['brand_model'])} {e(i['material'])}</div></td>"
                      f"<td>{_num(d.get('w'))} × {_num(d.get('h'))}<div class=small>{e(d.get('method', ''))}</div></td>"
                      f"<td>{e(i['status'])}</td><td>{price}</td><td>{i['confidence']:.2f}</td></tr>")
    rq = "".join(f"<tr><td>{e(x['ref_id'])}</td><td>{e(x.get('kind', ''))}</td><td>{e(x['reason'])}</td></tr>" for x in p["review_queue"])
    cost = p["meta"].get("cost", {})
    stages = "".join(f"<tr><td>{e(k)}</td><td>{v.get('calls')}</td><td>{v.get('wall_ms')}</td><td>{v.get('model_latency_ms_p50')}</td>"
                     f"<td>{v.get('cost_usd')}</td></tr>" for k, v in cost.get("stages", {}).items())
    errs = p["meta"].get("totals_check", [])
    return f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>
<title>Contents claim {e(sw['id'])}</title><style>
body{{font:14px/1.4 system-ui,sans-serif;margin:24px;color:#1b1b1b;background:#fff}} table{{border-collapse:collapse;width:100%;margin:8px 0 24px}}
td,th{{border-bottom:1px solid #ddd;padding:6px;vertical-align:top;text-align:left}} th{{background:#f4f4f4}} .small{{font-size:11px;color:#555}}
.na{{color:#aaa}} img{{max-height:80px;max-width:80px}} .unidentified td{{background:#fafafa}} .needs_appraisal td{{background:#fff6e0}}
.kpi{{display:inline-block;margin:0 18px 12px 0}} .kpi b{{font-size:20px;display:block}} h2{{margin-top:32px}}
</style></head><body>
<h1>Library contents claim packet</h1>
<p>Sweep <b>{e(sw['id'])}</b> · captured {e(sw['captured_at'])} · {sw['duration_s']} s · {e(sw['device'])} · {e(sw['country'])} / {e(cur)}</p>
<div class=kpi><b>{t['book_count']}</b>books ({t['books_identified']} identified, {t['books_unidentified']} unidentified, {t['books_needs_appraisal']} appraisal)</div>
<div class=kpi><b>{_num(t['shelf_run_m'], ' m')}</b>shelf run</div>
<div class=kpi><b>{cur} {t['books_replacement_cost']:.2f}</b>books replacement ({t['books_priced_replacement']} priced)</div>
<div class=kpi><b>{cur} {t['books_used_value']:.2f}</b>books used value ({t['books_priced_used']} priced)</div>
<div class=kpi><b>{cur} {t['items_replacement_cost_low']:.2f} – {t['items_replacement_cost_high']:.2f}</b>other contents</div>
<div class=kpi><b>{t['excluded_from_totals']}</b>lines excluded from totals</div>
<p class=small>Totals are computed in code from the lines below ({e(t['computed_by'])}). Self-check: {'OK' if not errs else e('; '.join(errs))}.
Lines with no verifiable price, unidentified books and appraisal items are excluded, not estimated.</p>
<h2>Room</h2>
<table><tr><th>Length</th><th>Width</th><th>Height</th><th>Floor</th><th>Walls (gross)</th><th>Shelved wall</th><th>Shape</th><th>Scale</th><th>Conf.</th></tr>
<tr><td>{_num(r['length_m'], ' m')}</td><td>{_num(r['width_m'], ' m')}</td><td>{_num(r['height_m'], ' m')}</td>
<td>{_num(r['floor_area_m2'], ' m²')}<div class=small>{_num(r['floor_area_ft2'], ' ft²')}</div></td>
<td>{_num(r['wall_area_m2'], ' m²')}<div class=small>{_num(r['wall_area_ft2'], ' ft²')}</div></td>
<td>{_num(r['shelved_wall_area_m2'], ' m²')}<div class=small>{_num(r['shelved_wall_area_ft2'], ' ft²')}</div></td>
<td>{e(r['shape'])}<div class=small>{e(r['shape_note'])}</div></td><td class=small>{e(r['scale_method'])}</td><td>{r['confidence']}</td></tr></table>
<h2>Books ({len(p['books'])})</h2>
<table><tr><th>ID</th><th>Shelf</th><th>Spine</th><th>Identification</th><th>Status</th><th>ID conf.</th><th>H × T (cm)</th><th>Replacement</th><th>Used</th><th>Frame</th></tr>
{''.join(rows_b)}</table>
<h2>Other contents ({len(p['items'])})</h2>
<table><tr><th>ID</th><th>View</th><th>Category</th><th>Description</th><th>W × H (cm)</th><th>Status</th><th>Replacement ({e(cur)})</th><th>Conf.</th></tr>
{''.join(rows_i)}</table>
<h2>Review queue ({len(p['review_queue'])})</h2>
<table><tr><th>Ref</th><th>Kind</th><th>Reason</th></tr>{rq}</table>
<h2>Cost and latency</h2>
<p>Total model cost this sweep: <b>${cost.get('total_cost_usd', 0)}</b> (live session {cost.get('live_session_s', 0)} s).
Time from end of sweep to packet: <b>{p['meta'].get('time_to_packet_s')} s</b>.</p>
<table><tr><th>Stage</th><th>Model calls</th><th>Wall ms</th><th>Model p50 ms</th><th>USD</th></tr>{stages}</table>
<p class=small>Models: {e(json.dumps(p['meta'].get('models', {})))}</p>
</body></html>"""


def write_outputs(packet: ClaimPacket, sweep_dir: Path) -> dict[str, Path]:
    errs = check_totals(packet)
    packet.meta["totals_check"] = errs
    data = json.loads(packet.model_dump_json())
    jp = sweep_dir / "claim_packet.json"
    jp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    crops = {}
    for b in data["books"]:
        rel = f"crops/{b['id']}.jpg"
        if _crop(sweep_dir, b["frame_ref"], b["frame_box"], sweep_dir / rel):
            crops[b["id"]] = rel
    for i in data["items"]:
        rel = f"crops/{i['id']}.jpg"
        if _crop(sweep_dir, i["frame_ref"], i["frame_box"], sweep_dir / rel):
            crops[i["id"]] = rel
    hp = sweep_dir / "report.html"
    hp.write_text(render_html(data, crops), encoding="utf-8")
    zp = sweep_dir / f"claim_packet_{packet.sweep.id}.zip"
    refs = {r.split("#")[0] for b in data["books"] for r in b["frame_refs"] + [b["frame_ref"]] if r}
    refs |= {r.split("#")[0] for i in data["items"] for r in i["frame_refs"] + [i["frame_ref"]] if r}
    refs |= {r.split("#")[0] for r in data["room"]["frame_refs"]}
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jp, "claim_packet.json")
        z.write(hp, "report.html")
        for rel in sorted(refs | set(crops.values())):
            f = sweep_dir / rel
            if f.exists():
                z.write(f, rel)
    return {"json": jp, "html": hp, "zip": zp}
