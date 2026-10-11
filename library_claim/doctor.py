"""Pre-flight checks before a demo run, and a one-photo smoke test.

    python -m library_claim.cli doctor               # every external dependency, one line each
    python -m library_claim.cli doctor --live        # also opens a Gemini Live session (~10 s)
    python -m library_claim.cli doctor --depth       # also downloads + runs the depth model (~100 MB first time)
    python -m library_claim.cli try-photo shelf.jpg  # read, identify and measure one photo of a shelf
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .config import ROOT, SETTINGS

OK, BAD, SKIP = "PASS", "FAIL", "SKIP"
results: list[tuple[str, str, str]] = []
led_ref: list = []
led_total: list[float] = []


def report(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""), flush=True)


async def _timed(coro, timeout: float):
    t0 = time.perf_counter()
    res = await asyncio.wait_for(coro, timeout)
    return res, int((time.perf_counter() - t0) * 1000)


def check_local() -> None:
    import sys
    v = sys.version_info
    report("python >= 3.11", OK if v >= (3, 11) else BAD, f"{v.major}.{v.minor}.{v.micro}")
    report(".env file", OK if (ROOT / ".env").exists() else BAD, "" if (ROOT / ".env").exists() else "copy .env.example .env")
    key = SETTINGS.google_api_key
    report("GOOGLE_API_KEY set", OK if key and key != "replace-me" else BAD, "" if key and key != "replace-me" else "put your key in .env")
    try:
        cv2.aruco.ArucoDetector
        report("opencv with aruco", OK, cv2.__version__)
    except AttributeError:
        report("opencv with aruco", BAD, "pip uninstall opencv-python; pip install opencv-contrib-python-headless")
    mk = ROOT / "markers" / "marker_0.png"
    if mk.exists():
        from .measure.scale import detect_markers
        hits = detect_markers(cv2.imread(str(mk)), "m")
        report("printable marker detectable", OK if hits else BAD, str(mk.relative_to(ROOT)))
    else:
        report("printable marker", BAD, "run: python -m library_claim.cli marker")
    report("marker side setting", OK, f"LCA_MARKER_SIDE_CM={SETTINGS.marker_side_cm} (must equal your ruler measurement)")
    report("cost guards", OK, f"LCA_MAX_USD_PER_SWEEP=${SETTINGS.max_usd_per_sweep:.2f}, LCA_MAX_VISION_CALLS={SETTINGS.max_vision_calls}; "
           f"rates in ${SETTINGS.price_in_per_m}/{SETTINGS.price_out_per_m} per 1M, search ${SETTINGS.search_price_per_call}")
    report("camera FOV setting", OK, f"LCA_CAMERA_HFOV_DEG={SETTINGS.camera_hfov_deg} (calibrate with calibrate-fov)")
    try:
        import cryptography  # noqa: F401
        report("https certificate support", OK)
    except ImportError:
        report("https certificate support", BAD, "pip install cryptography")
    from .measure.room import DepthModel
    report("depth stack (torch + transformers)", OK if DepthModel(SETTINGS.depth_model).available() else BAD,
           "" if DepthModel(SETTINGS.depth_model).available() else "pip install -r requirements-depth.txt (room area will be empty)")


async def check_free_apis() -> None:
    from .http import get_json
    try:
        d, ms = await _timed(get_json("https://openlibrary.org/search.json", {"q": "the hobbit tolkien", "limit": 1,
                                                                             "fields": "title,author_name"}), 20)
        ok = bool(d and d.get("docs"))
        report("Open Library search", OK if ok else BAD, f"{ms} ms" + ("" if ok else " (no results / blocked)"))
    except Exception as e:
        report("Open Library search", BAD, repr(e))
    try:
        d, ms = await _timed(get_json("https://www.googleapis.com/books/v1/volumes", {"q": "intitle:hobbit", "maxResults": 1,
                                                                                       "country": SETTINGS.default_country}), 20)
        ok = bool(d and d.get("items"))
        report("Google Books search", OK if ok else BAD, f"{ms} ms" + ("" if ok else " (no results / quota)"))
    except Exception as e:
        report("Google Books search", BAD, repr(e))
    from .pricing.fx import rate
    try:
        r, ms = await _timed(rate("GBP", "USD"), 20)
        report("ECB exchange rate (Frankfurter)", OK if r else BAD, f"GBP->USD {r[0]:.4f} on {r[1]}, {ms} ms" if r else "unreachable")
    except Exception as e:
        report("ECB exchange rate (Frankfurter)", BAD, repr(e))


async def check_gemini(live: bool) -> None:
    if not SETTINGS.google_api_key or SETTINGS.google_api_key == "replace-me":
        for n in ("vision model", "pricing search", "live model"):
            report(n, SKIP, "no API key")
        return
    from .llm import Ledger, call_json, client, grounded_search
    names = []
    try:
        names = [m.name.split("/")[-1] for m in client().models.list()]
    except Exception as e:
        report("list Gemini models", BAD, repr(e)[:160])
    for label, model in (("vision model", SETTINGS.vision_model), ("pricing model", SETTINGS.pricing_model),
                         ("live model", SETTINGS.live_model)):
        if names and model not in names:
            near = [n for n in names if ("live" in n) == ("live" in model) and "gemini" in n][:6]
            report(f"{label} '{model}' exists", BAD, f"not in your model list; set LCA_*_MODEL in .env. Available: {', '.join(near)}")
        elif names:
            report(f"{label} '{model}' exists", OK)
    led = Ledger()
    led_ref.append(led)
    try:
        img = np.full((480, 640, 3), 235, np.uint8)
        cv2.putText(img, "THE HOBBIT", (40, 250), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (20, 20, 20), 5)
        from google.genai import types
        part = types.Part.from_bytes(data=cv2.imencode(".jpg", img)[1].tobytes(), mime_type="image/jpeg")
        out, ms = await _timed(call_json("doctor", led, [part, 'Return JSON {"text": "<the words in the image>"}']), 60)
        ok = "hobbit" in json.dumps(out).lower()
        report("vision model reads an image (JSON mode)", OK if ok else BAD, f"{ms} ms, got {json.dumps(out)[:60]}")
    except Exception as e:
        report("vision model reads an image (JSON mode)", BAD, repr(e)[:200])
    try:
        g, ms = await _timed(grounded_search("doctor", led, "What is the current list price in USD of the paperback "
                                             "'The Hobbit' by J.R.R. Tolkien at a US retailer? One sentence."), 90)
        report("Google Search grounding", OK if g.chunks else BAD,
               f"{ms} ms, {len(g.chunks)} sources, {len(g.supports)} supported sentences" if g.chunks else
               "no grounding sources returned (pricing will drop every offer)")
    except Exception as e:
        report("Google Search grounding", BAD, repr(e)[:200])
    led_total.append(led.cost_so_far())
    if live:
        await check_live()
    else:
        report("Gemini Live session", SKIP, "run with --live to test")


async def check_live() -> None:
    from google.genai import types
    from .live.agent import live_config
    from .llm import client
    try:
        t0 = time.perf_counter()
        got_audio = got_text = False
        async with client().aio.live.connect(model=SETTINGS.live_model, config=live_config(SETTINGS.default_country)) as s:
            await s.send_client_content(turns=types.Content(role="user", parts=[types.Part(text="Say the word ready.")]), turn_complete=True)

            async def drain():
                nonlocal got_audio, got_text
                async for r in s.receive():
                    sc = r.server_content
                    if sc and sc.model_turn:
                        for p in sc.model_turn.parts or []:
                            if p.inline_data and (p.inline_data.mime_type or "").startswith("audio/pcm"):
                                got_audio = True
                    if sc and sc.output_transcription and sc.output_transcription.text:
                        got_text = True
                    if r.tool_call:
                        got_text = True
                    if sc and sc.turn_complete:
                        return
            await asyncio.wait_for(drain(), 25)
        ms = int((time.perf_counter() - t0) * 1000)
        report("Gemini Live session (voice + tools config)", OK if got_audio else BAD,
               f"{ms} ms, audio={'yes' if got_audio else 'no'}, transcript={'yes' if got_text else 'no'}")
    except Exception as e:
        report("Gemini Live session (voice + tools config)", BAD, repr(e)[:240])


def check_depth() -> None:
    from .measure.room import DepthModel
    dm = DepthModel(SETTINGS.depth_model)
    if not dm.available():
        report("depth model run", SKIP, "torch/transformers not installed")
        return
    try:
        t0 = time.perf_counter()
        dm(np.full((388, 518, 3), 128, np.uint8))           # first call downloads + loads
        t1 = time.perf_counter()
        d = dm(np.random.default_rng(0).integers(0, 255, (388, 518, 3), dtype=np.uint8))
        t2 = time.perf_counter()
        report("depth model run", OK, f"load {t1 - t0:.1f} s, {(t2 - t1) * 1000:.0f} ms/frame, depth range {d.min():.2f}-{d.max():.2f} m")
    except Exception as e:
        report("depth model run", BAD, repr(e)[:200])


async def _doctor(live: bool, depth: bool) -> None:
    print("\nLocal setup")
    check_local()
    print("\nFree data sources")
    await check_free_apis()
    print("\nGemini")
    await check_gemini(live)
    print("\nDepth model")
    if depth:
        await asyncio.to_thread(check_depth)
    else:
        report("depth model run", SKIP, "run with --depth to download and time it")


def doctor(live: bool = False, depth: bool = False) -> int:
    asyncio.run(_doctor(live, depth))
    bad = [r for r in results if r[1] == BAD]
    if led_total:
        print(f"\nThis doctor run cost about ${led_total[0]:.3f}.")
    print(f"\n{len(results) - len(bad)} ok / {len(bad)} failed" + ("" if not bad else ":  " + "; ".join(r[0] for r in bad)))
    return 1 if bad else 0


async def _try_photo(path: str, country: str) -> None:
    """Run reader + marker + measurement + identification on ONE photo and print a table."""
    from .identify.catalogue import resolve
    from .llm import Ledger
    from .measure.refine import refine_box
    from .measure.scale import detect_markers, plane_from_marker, quad_dims_cm
    from .vision.reader import read_frame
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    h, w = img.shape[:2]
    s = min(1.0, 1600 / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    led = Ledger()
    t0 = time.perf_counter()
    fv = await read_frame(img, "photo", led)
    print(f"reader: {len(fv.spines)} spines ({sum(s.legible for s in fv.spines)} legible), {len(fv.items)} items, "
          f"{len(fv.shelf_units)} bookcases in {time.perf_counter() - t0:.1f} s; feedback: {fv.capture_feedback!r}")
    hits = detect_markers(img, "photo")
    P = plane_from_marker(max(hits, key=lambda x: x.area_px), SETTINGS.marker_side_cm) if hits else None
    print("marker:", f"found, side {hits[0].side_px:.0f} px" if hits else "NOT found (no cm measurements; lean it on the spines)")
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ids = await asyncio.gather(*(resolve(sp.text, sp.title_read, sp.author_read, sp.publisher_read, country) if sp.legible
                                 else asyncio.sleep(0, result=None) for sp in fv.spines))
    vis = img.copy()
    print(f"\n{'#':>3} {'H cm':>6} {'T cm':>5}  {'status':12} spine text -> match")
    for i, (sp, r) in enumerate(sorted(zip(fv.spines, ids), key=lambda z: (z[0].row or 0, z[0].box[0]))):
        hcm = tcm = None
        if P is not None:
            b = refine_box(grey, sp.box, sp.orientation)
            q = np.float32([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
            bw, bh = quad_dims_cm(P, q)
            hcm, tcm = (bh, bw) if sp.orientation == "vertical" else (bw, bh)
        status = "identified" if r and r.accepted else ("no match" if sp.legible else "unreadable")
        match = f"{r.title} / {r.author} ({r.confidence:.2f})" if r and r.accepted else (r.reason if r else "")
        print(f"{i:>3} {hcm or 0:6.1f} {tcm or 0:5.1f}  {status:12} {sp.text[:38]!r} -> {match[:60]}")
        col = (0, 200, 0) if status == "identified" else ((0, 200, 255) if sp.legible else (0, 0, 255))
        x0, y0, x1, y1 = [int(v) for v in sp.box]
        cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
        cv2.putText(vis, str(i), (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    for it in fv.items:
        x0, y0, x1, y1 = [int(v) for v in it.box]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 120, 0), 2)
        cv2.putText(vis, it.category, (x0, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)
    out = Path(path).with_name(Path(path).stem + "_annotated.jpg")
    cv2.imwrite(str(out), vis)
    print("\nitems: " + "; ".join(f"{i.category}: {i.description}" for i in fv.items))
    print(f"annotated image: {out}   (green = identified, orange = legible but unmatched, red = unreadable)")
    print("cost:", json.dumps(led.report()["total_cost_usd"]), "USD")


def try_photo(path: str, country: str) -> None:
    asyncio.run(_try_photo(path, country))
