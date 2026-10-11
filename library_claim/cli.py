"""Command line entry points.

    python -m library_claim.cli marker                      # printable reference marker
    python -m library_claim.cli calibrate-fov photo.jpg --distance-cm 100
    python -m library_claim.cli process sweeps/<id>         # re-run all stages on saved frames
    python -m library_claim.cli process my_sweep.mp4        # run on a recorded video (no live agent)
    python -m library_claim.cli reprice sweeps/<id> --country GB --limit 10
    python -m library_claim.cli doctor [--live] [--depth]  # pre-flight checks
    python -m library_claim.cli try-photo shelf.jpg         # read + identify + measure one photo
    python -m library_claim.cli gt-template ground_truth.xlsx
    python -m library_claim.cli score sweeps/<id> ground_truth.xlsx
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
from pathlib import Path

import cv2


def cmd_marker(a):
    from .measure.scale import render_marker_png
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    side_px = int(round(a.side_cm / 2.54 * 300))      # 300 dpi
    img = render_marker_png(a.id, side_px)
    cv2.putText(img, f"ArUco 4x4 id {a.id} - black square must measure {a.side_cm:g} cm. Measure after printing and set "
                     f"LCA_MARKER_SIDE_CM.", (20, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    cv2.imwrite(str(out), img)
    print(f"Wrote {out}. Print at 100% scale (300 dpi) on A4/Letter, then measure the black square with a ruler.")


def cmd_calibrate(a):
    """Horizontal FOV from one photo of the marker held flat-on at a tape-measured distance."""
    import math
    from .measure.scale import detect_markers
    from .config import SETTINGS
    img = cv2.imread(a.photo)
    hits = detect_markers(img, "cal")
    if not hits:
        sys.exit("Marker not found in the photo.")
    side_px = max(h.side_px for h in hits)
    fx = side_px * (a.distance_cm / 100) / (SETTINGS.marker_side_cm / 100)
    hfov = math.degrees(2 * math.atan(img.shape[1] / 2 / fx))
    print(f"fx = {fx:.0f}px at width {img.shape[1]} -> horizontal FOV {hfov:.1f} deg. Put LCA_CAMERA_HFOV_DEG={hfov:.1f} in .env")


async def _process(a):
    from .sweep import SweepSession
    src = Path(a.source)
    s = SweepSession(country=a.country, device=f"offline:{src.name}", use_depth=not a.no_depth)
    print("sweep dir:", s.dir)
    s.start()
    if src.is_dir():
        idx = [json.loads(l) for l in (src / "frames.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        if a.reuse_vision and (src / "stages" / "vision").exists():
            shutil.copytree(src / "stages" / "vision", s.dir / "stages" / "vision", dirs_exist_ok=True)
        for rec in idx:
            await s.add_frame((src / rec["path"]).read_bytes(), rec.get("imu"), t=rec["t"])
    else:
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(1, int(round(fps / a.fps)))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                h, w = frame.shape[:2]
                sc = min(1.0, 1600 / max(h, w))
                if sc < 1:
                    frame = cv2.resize(frame, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
                await s.add_frame(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes(), None, t=i / fps)
            i += 1
    await s.finish()
    print(json.dumps(s.summary(), indent=2))
    print("report:", s.dir / "report.html")


async def _reprice(a):
    from .config import locale_for
    from .llm import Ledger
    from .pricing.books import price_book
    d = Path(a.sweep)
    p = json.loads((d / "claim_packet.json").read_text(encoding="utf-8"))
    loc = locale_for(a.country)
    books = [b for b in p["books"] if b["status"] == "identified"][: a.limit]
    ledger = Ledger()
    res = await asyncio.gather(*(price_book(b["title"], b["author"], b["edition"], b["isbn"], loc, ledger) for b in books))
    rows = []
    for b, r in zip(books, res):
        rows.append({"id": b["id"], "title": b["title"], "home_country": p["sweep"]["country"],
                     "home_replacement": b["replacement_cost"], "home_used": b["used_value"],
                     "country": loc.country, "replacement": r.replacement.model_dump(), "used": r.used.model_dump(),
                     "appraisal": r.appraisal_reason, "review": r.review})
    out = {"home": {"country": p["sweep"]["country"], "currency": p["sweep"]["currency"]},
           "second": {"country": loc.country, "currency": loc.currency}, "books": rows, "cost": ledger.report()}
    jp = d / f"locale_comparison_{loc.country}.json"
    jp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    from .packet.locale_report import render
    hp = d / f"locale_comparison_{loc.country}.html"
    hp.write_text(render(out), encoding="utf-8")
    print("wrote", jp, "and", hp)


def cmd_usage():
    """Sum usage_log.csv: per sweep, and month to date (compare with AI Studio > Dashboard > Usage)."""
    import csv
    from .config import SETTINGS
    p = SETTINGS.data_dir / "usage_log.csv"
    if not p.exists():
        print("no sweeps finished yet")
        return
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    print(f"{'sweep':24} {'calls':>6} {'search':>6} {'tok in':>9} {'tok out':>9} {'live s':>7} {'USD':>7}")
    month = rows[-1]["finished_at"][:7]
    tot = 0.0
    for r in rows:
        print(f"{r['sweep_id']:24} {r['model_calls']:>6} {r['search_calls']:>6} {r['tokens_in']:>9} {r['tokens_out']:>9} "
              f"{float(r['live_session_s']):>7.0f} {float(r['est_cost_usd']):>7.2f}")
        if r["finished_at"].startswith(month):
            tot += float(r["est_cost_usd"])
    searches = sum(int(r["search_calls"]) for r in rows if r["finished_at"].startswith(month))
    print(f"\n{month}: estimated ${tot:.2f}; {searches} grounded searches (first 5,000/month are free on the paid tier, "
          "so the real bill can be lower than this estimate)")


def main(argv=None):
    logging.basicConfig(level=logging.WARNING if (argv or sys.argv[1:2]) and (argv or sys.argv[1:])[0] in ("doctor", "try-photo", "usage") else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from .config import SETTINGS
    ap = argparse.ArgumentParser(prog="library_claim")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("marker")
    m.add_argument("--id", type=int, default=0)
    m.add_argument("--side-cm", type=float, default=SETTINGS.marker_side_cm)
    m.add_argument("--out", default="markers/marker_0.png")
    c = sub.add_parser("calibrate-fov")
    c.add_argument("photo")
    c.add_argument("--distance-cm", type=float, required=True)
    pr = sub.add_parser("process")
    pr.add_argument("source", help="saved sweep directory or a video file")
    pr.add_argument("--country", default=SETTINGS.default_country)
    pr.add_argument("--fps", type=float, default=3.0)
    pr.add_argument("--no-depth", action="store_true")
    pr.add_argument("--reuse-vision", action="store_true", help="reuse cached vision reads (only changes downstream stages)")
    rp = sub.add_parser("reprice")
    rp.add_argument("sweep")
    rp.add_argument("--country", required=True)
    rp.add_argument("--limit", type=int, default=10)
    sub.add_parser("usage")
    dr = sub.add_parser("doctor")
    dr.add_argument("--live", action="store_true")
    dr.add_argument("--depth", action="store_true")
    tp = sub.add_parser("try-photo")
    tp.add_argument("photo")
    tp.add_argument("--country", default=SETTINGS.default_country)
    g = sub.add_parser("gt-template")
    g.add_argument("out")
    sc = sub.add_parser("score")
    sc.add_argument("sweep")
    sc.add_argument("gt")
    a = ap.parse_args(argv)
    if a.cmd == "marker":
        cmd_marker(a)
    elif a.cmd == "calibrate-fov":
        cmd_calibrate(a)
    elif a.cmd == "process":
        asyncio.run(_process(a))
    elif a.cmd == "reprice":
        asyncio.run(_reprice(a))
    elif a.cmd == "usage":
        cmd_usage()
    elif a.cmd == "doctor":
        from .doctor import doctor
        sys.exit(doctor(a.live, a.depth))
    elif a.cmd == "try-photo":
        from .doctor import try_photo
        try_photo(a.photo, a.country)
    elif a.cmd == "gt-template":
        from .eval.score import write_template
        write_template(Path(a.out))
        print("wrote", a.out)
    elif a.cmd == "score":
        from .eval.score import score, write_results
        res = score(Path(a.sweep), Path(a.gt))
        out = write_results(res, Path(a.gt), Path(a.sweep))
        for k, v in res["bars"].items():
            if "pass" in v:
                print(f"{k:22s} {'PASS' if v['pass'] else 'FAIL'}  {v.get('bar', '')}")
        print("wrote", out)


if __name__ == "__main__":
    main()
