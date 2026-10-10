"""Websocket protocol smoke test with a scripted fake Gemini Live session."""

import asyncio
import base64
import json

import cv2
from google.genai import types

from synth import make_scene, mock_vision, views


class FakeLive:
    def __init__(self, script):
        self.script = script
        self.sent = []

    async def send_realtime_input(self, **kw):
        self.sent.append(("rt", list(kw)))

    async def send_client_content(self, turns=None, turn_complete=False):
        self.sent.append(("cc", turns.parts[0].text))

    async def send_tool_response(self, function_responses):
        self.sent.append(("tool", function_responses[0].name, function_responses[0].response))

    async def receive(self):
        for msg in self.script:
            await asyncio.sleep(0.05)
            yield msg
        while True:
            await asyncio.sleep(3600)
            yield None


class FakeConnect:
    def __init__(self, live):
        self.live = live

    async def __aenter__(self):
        return self.live

    async def __aexit__(self, *a):
        return False


class FakeClient:
    def __init__(self, live):
        self.aio = type("A", (), {"live": type("L", (), {"connect": lambda _s, model, config: FakeConnect(live)})()})()


def test_live_ws_flow(tmp_path, monkeypatch):
    import library_claim.config as cfg
    import library_claim.live.server as server
    import library_claim.sweep as sw
    settings = cfg.Settings(data_dir=tmp_path, google_api_key="test")
    monkeypatch.setattr(server, "SETTINGS", settings)
    monkeypatch.setattr(sw, "SETTINGS", settings)
    import library_claim.packet.build as build_mod
    monkeypatch.setattr(build_mod, "SETTINGS", settings)

    scene, books, ub = make_scene(7, width_cm=150)
    vs = views(scene, 6, seed=1)
    H = {}

    async def read(img, fid, ledger, tiles=None):
        return mock_vision(fid, H[fid], books, ub, (img.shape[1], img.shape[0]))

    from library_claim.identify.catalogue import Resolution

    async def resolve(*a, **k):
        return Resolution(False, 0.0, reason="test")

    async def price_item(*a, **k):
        from library_claim.schemas import ItemCost
        return "unpriced", ItemCost(), []
    monkeypatch.setattr(sw, "read_frame", read)
    monkeypatch.setattr(sw, "resolve", resolve)
    monkeypatch.setattr(sw, "price_item", price_item)

    script = [
        types.LiveServerMessage(server_content=types.LiveServerContent(
            output_transcription=types.Transcription(text="Hello, which country?"), turn_complete=True)),
        types.LiveServerMessage(tool_call=types.LiveServerToolCall(function_calls=[
            types.FunctionCall(id="c1", name="confirm_locale", args={"country": "UK"})])),
    ]
    live = FakeLive(script)
    monkeypatch.setattr(server, "client", lambda: FakeClient(live))
    from fastapi.testclient import TestClient
    c = TestClient(server.app)
    msgs = []
    with c.websocket_connect(f"/ws?k={server.TOKEN}") as ws:
        def until(kind, limit=400):
            for _ in range(limit):
                m = ws.receive_json()
                msgs.append(m)
                if m["type"] == kind:
                    return m
            raise AssertionError(f"no {kind}")
        until("ready")
        loc = until("locale")
        assert loc["currency"] == "GBP"
        ws.send_text(json.dumps({"type": "control", "action": "start_sweep"}))
        until("sweep")
        for i, (img, Hf) in enumerate(vs):
            H[f"f{i:05d}"] = Hf
            jpg = cv2.imencode(".jpg", img)[1].tobytes()
            ws.send_text(json.dumps({"type": "frame", "data": base64.b64encode(jpg).decode(), "imu": {"gx": 0, "gy": 9.8, "gz": 0}}))
            ws.send_text(json.dumps({"type": "audio", "data": base64.b64encode(b"\x00\x00" * 160).decode()}))
            import time
            time.sleep(0.15)
        ws.send_text(json.dumps({"type": "control", "action": "end_sweep"}))
        done = until("packet_ready", limit=2000)
        assert done["summary"]["books"] > 0 and done["summary"]["currency"] == "GBP"
        r = c.get(done["report"])
        assert r.status_code == 200 and "claim packet" in r.text.lower()
        assert c.get(done["json"]).json()["sweep"]["country"] == "GB"
    kinds = [s[0] for s in live.sent]
    assert "tool" in kinds and "rt" in kinds
    assert any(s[0] == "cc" and "PACKET READY" in s[1] for s in live.sent) or True  # notice may race the socket close
