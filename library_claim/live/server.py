"""FastAPI server: phone browser <-> this server <-> Gemini Live, plus the sweep pipeline.

Run:  python -m library_claim.live   (prints an https URL with an access key)
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import secrets
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from ..config import SETTINGS, locale_for
from ..llm import client
from ..sweep import SweepSession
from .agent import Director, live_config

log = logging.getLogger("library_claim.live")
STATIC = Path(__file__).parent / "static"
TOKEN = os.getenv("LCA_ACCESS_TOKEN") or secrets.token_urlsafe(9)
SWEEPS: dict[str, SweepSession] = {}
LIVE_FRAME_EVERY_S = 1.0
MAX_PENDING_FRAMES = 4

app = FastAPI(title="Library Contents Claim Agent")


def _check(k: str | None) -> None:
    if not k or not secrets.compare_digest(k, TOKEN):
        raise HTTPException(status_code=403, detail="Missing or wrong access key (?k=...) - see the server console.")


@app.get("/")
def index(k: str | None = None):
    _check(k)
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8").replace("__TOKEN__", k))


@app.get("/static/{name}")
def static(name: str):
    p = (STATIC / name).resolve()
    if p.parent != STATIC.resolve() or not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/health")
def health():
    return {"ok": True, "api_key": bool(SETTINGS.google_api_key), "live_model": SETTINGS.live_model,
            "vision_model": SETTINGS.vision_model, "country": SETTINGS.default_country}


@app.get("/sweeps/{sweep_id}/{path:path}")
def sweep_file(sweep_id: str, path: str, k: str | None = None, request: Request = None):
    _check(k or request.cookies.get("lca_k"))
    base = (SETTINGS.data_dir / sweep_id).resolve()
    p = (base / (path or "report.html")).resolve()
    if base not in p.parents or not p.is_file():
        raise HTTPException(404)
    resp = FileResponse(p)
    resp.set_cookie("lca_k", k or request.cookies.get("lca_k", ""), httponly=True, samesite="strict")
    return resp


def _downscale_jpeg(jpeg: bytes, width: int = 768) -> bytes:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jpeg
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()


@app.websocket("/ws")
async def live_ws(ws: WebSocket):
    if not secrets.compare_digest(ws.query_params.get("k", ""), TOKEN):
        await ws.close(code=1008)
        return
    await ws.accept()
    from google.genai import types
    send_lock = asyncio.Lock()

    async def send(payload: dict):
        async with send_lock:
            with contextlib.suppress(Exception):
                await ws.send_json(payload)

    session = SweepSession(country=SETTINGS.default_country, device=ws.headers.get("user-agent", "")[:160], notify=send)
    SWEEPS[session.id] = session
    state = {"active": False, "last_live_frame": 0.0, "pending_frames": 0, "finish": None, "connected_at": time.monotonic()}
    director = Director()
    live_q: asyncio.Queue = asyncio.Queue()   # text notices to the live model

    async def run_tool(fc, live):
        name, args = fc.name, dict(fc.args or {})
        result: dict
        try:
            if name == "confirm_locale":
                loc = session.set_country(str(args.get("country", "")))
                result = {"country": loc.country, "currency": loc.currency}
                await send({"type": "locale", "country": loc.country, "currency": loc.currency})
            elif name == "start_sweep":
                result = await start_sweep()
            elif name == "end_sweep":
                result = await end_sweep()
            elif name == "get_sweep_status":
                result = session.capture_status() if state["active"] else {"active": False}
            elif name == "record_correction":
                result = session.record_correction(str(args.get("kind")), str(args.get("scope")), str(args.get("said", "")))
            elif name == "record_known_dimension":
                result = session.record_correction("known_dimension", "unit", str(args.get("said", "")),
                                                   {"dimension": args.get("dimension"), "cm": float(args.get("centimetres", 0))})
            elif name == "answer_art_question":
                result = session.answer_art(str(args.get("item_id")), bool(args.get("is_print")))
            else:
                result = {"error": f"unknown tool {name}"}
        except ValueError as exc:
            result = {"error": str(exc)}
        except Exception as exc:
            log.exception("tool %s failed", name)
            result = {"error": f"{name} failed: {exc}"}
        await send({"type": "tool", "name": name, "args": args, "result": result})
        await live.send_tool_response(function_responses=[types.FunctionResponse(
            id=fc.id, name=name, response=result, scheduling=types.FunctionResponseScheduling.WHEN_IDLE)])

    async def start_sweep():
        if state["active"]:
            return {"started": True, "already": True}
        if session.finished:
            return {"error": "This sweep is finished. Reload the page for a new sweep."}
        state["active"] = True
        session.start()
        await send({"type": "sweep", "active": True, "sweep_id": session.id})
        return {"started": True, "sweep_id": session.id}

    async def end_sweep():
        if not state["active"]:
            return {"error": "No sweep in progress."}
        state["active"] = False
        await send({"type": "sweep", "active": False, "sweep_id": session.id, "processing": True})

        async def finish():
            try:
                session.ledger.live_seconds = time.monotonic() - state["connected_at"]
                await session.finish()
                summary = session.summary()
                await live_q.put("PACKET READY: " + json.dumps(summary) + " Read back a short summary now.")
                await send({"type": "packet_ready", "sweep_id": session.id, "summary": summary,
                            "report": f"/sweeps/{session.id}/report.html?k={TOKEN}",
                            "zip": f"/sweeps/{session.id}/claim_packet_{session.id}.zip?k={TOKEN}",
                            "json": f"/sweeps/{session.id}/claim_packet.json?k={TOKEN}"})
            except Exception as exc:
                log.exception("finish failed")
                await send({"type": "error", "message": f"Packet build failed: {exc}"})
                await live_q.put("PACKET FAILED: tell the policyholder the packet could not be built and the frames are saved.")
        state["finish"] = asyncio.create_task(finish())
        return {"status": "processing", "note": "You will receive PACKET READY in a few minutes."}

    async def browser_to_live(live):
        while True:
            msg = await ws.receive_text()
            m = json.loads(msg)
            kind = m.get("type")
            if kind == "audio":
                await live.send_realtime_input(audio=types.Blob(data=base64.b64decode(m["data"]), mime_type="audio/pcm;rate=16000"))
            elif kind == "frame":
                jpeg = base64.b64decode(m["data"])
                now = time.monotonic()
                if now - state["last_live_frame"] >= LIVE_FRAME_EVERY_S:
                    state["last_live_frame"] = now
                    small = await asyncio.to_thread(_downscale_jpeg, jpeg)
                    await live.send_realtime_input(video=types.Blob(data=small, mime_type="image/jpeg"))
                if state["active"]:
                    if state["pending_frames"] >= MAX_PENDING_FRAMES:
                        continue   # drop: the server is behind; keyframes come from later frames
                    state["pending_frames"] += 1

                    async def store(j=jpeg, imu=m.get("imu")):
                        try:
                            await session.add_frame(j, imu)
                        except Exception:
                            log.exception("frame rejected")
                        finally:
                            state["pending_frames"] -= 1
                    asyncio.create_task(store())
            elif kind == "text":
                await live.send_client_content(turns=types.Content(role="user", parts=[types.Part(text=str(m.get("text", ""))[:2000])]), turn_complete=True)
            elif kind == "control":
                act = m.get("action")
                if act == "start_sweep":
                    await start_sweep()
                    await live_q.put("The policyholder pressed Start sweep. Acknowledge in a few words.")
                elif act == "end_sweep":
                    await end_sweep()
                    await live_q.put("The policyholder pressed End sweep. Tell them the team is finishing the packet.")
                elif act == "set_country":
                    loc = session.set_country(str(m.get("country", "")))
                    await send({"type": "locale", "country": loc.country, "currency": loc.currency})
            elif kind == "close":
                return

    async def live_to_browser(live):
        pending = {"user": "", "agent": ""}
        while True:
            async for r in live.receive():
                if getattr(r, "usage_metadata", None):
                    session.ledger.add_live_usage(r.usage_metadata)
                if r.tool_call and r.tool_call.function_calls:
                    for fc in r.tool_call.function_calls:
                        asyncio.create_task(run_tool(fc, live))
                sc = r.server_content
                if not sc:
                    continue
                if sc.interrupted:
                    await send({"type": "interrupted"})
                for who, tr in (("user", sc.input_transcription), ("agent", sc.output_transcription)):
                    if tr and tr.text:
                        pending[who] += tr.text
                        await send({"type": "transcript", "who": who, "text": pending[who], "final": False})
                if sc.model_turn:
                    for part in sc.model_turn.parts or []:
                        if part.inline_data and (part.inline_data.mime_type or "").startswith("audio/pcm"):
                            await send({"type": "audio", "data": base64.b64encode(part.inline_data.data).decode()})
                if sc.turn_complete:
                    for who in pending:
                        if pending[who]:
                            await send({"type": "transcript", "who": who, "text": pending[who], "final": True})
                            pending[who] = ""

    async def notices(live):
        while True:
            try:
                text = await asyncio.wait_for(live_q.get(), timeout=1.0)
                await live.send_client_content(turns=types.Content(role="user", parts=[types.Part(text="[APP NOTICE] " + text)]), turn_complete=True)
                await send({"type": "notice", "text": text})
                continue
            except asyncio.TimeoutError:
                pass
            if not state["active"]:
                continue
            st = session.capture_status()
            await send({"type": "status", **st})
            arts = session.peek_art_questions()
            notice = director.next_notice(st, arts, session.missed_rows())
            if notice:
                if notice.startswith("ASK:") and arts:
                    session.asked_art.add(arts[0]["item_id"])
                await live.send_client_content(turns=types.Content(role="user", parts=[types.Part(text="[APP NOTICE] " + notice)]), turn_complete=True)
                await send({"type": "notice", "text": notice})

    try:
        if not SETTINGS.google_api_key:
            await send({"type": "error", "message": "GOOGLE_API_KEY missing on the server (.env)."})
            return
        async with client().aio.live.connect(model=SETTINGS.live_model, config=live_config(session.country)) as live:
            await send({"type": "ready", "sweep_id": session.id, "country": session.locale.country, "currency": session.locale.currency,
                        "model": SETTINGS.live_model})
            await live.send_client_content(turns=types.Content(role="user", parts=[types.Part(
                text="[APP NOTICE] The policyholder has connected with camera and microphone on. Greet them and start step 1.")]),
                turn_complete=True)
            tasks = [asyncio.create_task(browser_to_live(live)), asyncio.create_task(live_to_browser(live)),
                     asyncio.create_task(notices(live))]
            done, rest = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in rest:
                t.cancel()
            for t in done:
                if t.exception() and not isinstance(t.exception(), WebSocketDisconnect):
                    raise t.exception()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.exception("live session ended with error")
        await send({"type": "error", "message": f"Live session ended: {exc}"})
    finally:
        if state["active"]:
            session.ledger.live_seconds = time.monotonic() - state["connected_at"]
            # Connection dropped mid-sweep: still build the packet from what was captured.
            state["active"] = False
            with contextlib.suppress(Exception):
                await session.finish()
        elif state["finish"] is not None:
            with contextlib.suppress(Exception):
                await state["finish"]
        with contextlib.suppress(Exception):
            await ws.close()


@app.get("/api/sweeps")
def list_sweeps(k: str | None = None):
    _check(k)
    out = []
    for d in sorted(SETTINGS.data_dir.glob("2*"), reverse=True):
        if (d / "claim_packet.json").exists():
            out.append({"id": d.name, "report": f"/sweeps/{d.name}/report.html?k={k}"})
    return JSONResponse(out)


__all__ = ["app", "TOKEN", "locale_for"]
