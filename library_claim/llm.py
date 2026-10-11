"""Thin wrapper over google-genai with a cost/latency ledger.

Every model call in the pipeline goes through ``call_json`` or
``grounded_search`` so the packet can report cost per sweep and latency per
stage. Models never write totals or prices from memory; see pricing/ for how
search results are verified before a number is accepted.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .config import SETTINGS

log = logging.getLogger(__name__)
_CLIENT = None


def client():
    global _CLIENT
    if _CLIENT is None:
        if not SETTINGS.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set. Copy .env.example to .env and add it.")
        from google import genai
        _CLIENT = genai.Client(api_key=SETTINGS.google_api_key)
    return _CLIENT


@dataclass
class Ledger:
    """Accumulates tokens, calls and wall time per pipeline stage."""
    tokens_in: dict = field(default_factory=lambda: defaultdict(int))
    tokens_out: dict = field(default_factory=lambda: defaultdict(int))
    calls: dict = field(default_factory=lambda: defaultdict(int))
    search_calls: dict = field(default_factory=lambda: defaultdict(int))
    model_ms: dict = field(default_factory=lambda: defaultdict(list))
    stage_wall_ms: dict = field(default_factory=dict)
    live_seconds: float = 0.0
    live_tokens: dict = field(default_factory=lambda: defaultdict(int))   # from Live usage_metadata, by modality
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, stage: str, tin: int, tout: int, ms: int, search: bool = False) -> None:
        with self._lock:
            self.tokens_in[stage] += tin
            self.tokens_out[stage] += tout
            self.calls[stage] += 1
            self.model_ms[stage].append(ms)
            if search:
                self.search_calls[stage] += 1

    def stage(self, name: str):
        ledger = self

        class _Timer:
            def __enter__(self):
                self.t = time.perf_counter()
                return self

            def __exit__(self, *exc):
                ledger.stage_wall_ms[name] = ledger.stage_wall_ms.get(name, 0) + int((time.perf_counter() - self.t) * 1000)
        return _Timer()

    def add_live_usage(self, usage) -> None:
        """Accumulate a Live session usage_metadata message, by modality."""
        if usage is None:
            return
        with self._lock:
            for side, details in (("in", usage.prompt_tokens_details), ("out", usage.response_tokens_details)):
                for d in details or []:
                    mod = str(getattr(d.modality, "value", d.modality) or "TEXT").upper()
                    self.live_tokens[f"{side}_{mod}"] += int(d.token_count or 0)

    def cost_so_far(self) -> float:
        s = SETTINGS
        with self._lock:
            tin, tout = sum(self.tokens_in.values()), sum(self.tokens_out.values())
            searches = sum(self.search_calls.values())
        return (tin * s.price_in_per_m + tout * s.price_out_per_m) / 1e6 + searches * s.search_price_per_call \
            + self.live_seconds / 60 * s.live_price_per_min

    def check_budget(self, stage: str) -> None:
        """Raise BudgetExceeded before a call that would run past the per-sweep caps."""
        if stage.startswith("pricing") and self.cost_so_far() >= SETTINGS.max_usd_per_sweep:
            raise BudgetExceeded(f"per-sweep cap ${SETTINGS.max_usd_per_sweep:.2f} reached (LCA_MAX_USD_PER_SWEEP)")
        if stage == "vision_read" and self.calls.get("vision_read", 0) >= SETTINGS.max_vision_calls:
            raise BudgetExceeded(f"vision call cap {SETTINGS.max_vision_calls} reached (LCA_MAX_VISION_CALLS)")

    def report(self) -> dict[str, Any]:
        s = SETTINGS
        stages = sorted(set(self.calls) | set(self.stage_wall_ms))
        per_stage = {}
        total = 0.0
        for st in stages:
            cost = (self.tokens_in[st] * s.price_in_per_m + self.tokens_out[st] * s.price_out_per_m) / 1e6
            cost += self.search_calls[st] * s.search_price_per_call
            total += cost
            lat = sorted(self.model_ms[st])
            per_stage[st] = {
                "calls": self.calls[st], "search_calls": self.search_calls[st],
                "tokens_in": self.tokens_in[st], "tokens_out": self.tokens_out[st],
                "cost_usd": round(cost, 4),
                "model_latency_ms_p50": lat[len(lat) // 2] if lat else None,
                "model_latency_ms_max": lat[-1] if lat else None,
                "wall_ms": self.stage_wall_ms.get(st),
            }
        live_minutes_cost = self.live_seconds / 60 * s.live_price_per_min
        lt = self.live_tokens
        live_token_cost = (lt.get("in_TEXT", 0) * s.live_in_text_per_m + lt.get("in_AUDIO", 0) * s.live_in_audio_per_m
                           + (lt.get("in_IMAGE", 0) + lt.get("in_VIDEO", 0)) * s.live_in_video_per_m
                           + lt.get("out_TEXT", 0) * s.live_out_text_per_m + lt.get("out_AUDIO", 0) * s.live_out_audio_per_m) / 1e6
        # Prefer the session's own token report; fall back to the per-minute estimate.
        live_cost = live_token_cost if lt else live_minutes_cost
        total += live_cost
        return {"stages": per_stage, "live_session_s": round(self.live_seconds, 1),
                "live_tokens_reported": dict(self.live_tokens),
                "live_cost_basis": "session usage_metadata tokens" if lt else "per-minute estimate (LCA_LIVE_PRICE_PER_MIN)",
                "live_cost_per_minute_estimate_usd": round(live_minutes_cost, 4),
                "budget_cap_usd": s.max_usd_per_sweep,
                "live_cost_usd": round(live_cost, 4), "total_cost_usd": round(total, 4),
                "pricing_assumptions": {"in_per_m": s.price_in_per_m, "out_per_m": s.price_out_per_m,
                                        "search_per_call": s.search_price_per_call, "live_per_min": s.live_price_per_min}}


class BudgetExceeded(RuntimeError):
    """A per-sweep spend or call cap was reached; the caller records the line for review."""


def parse_json(text: str) -> Any:
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start > 0:
        text = text[start:]
    return json.loads(text)


def _usage(resp) -> tuple[int, int]:
    u = getattr(resp, "usage_metadata", None)
    if not u:
        return 0, 0
    return int(getattr(u, "prompt_token_count", 0) or 0), int((getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0))


def _thinking():
    """Optional thinking level for the background models (LCA_THINKING_LEVEL=minimal|low|medium|high)."""
    import os
    from google.genai import types
    lvl = os.getenv("LCA_THINKING_LEVEL", "").strip().upper()
    return types.ThinkingConfig(thinking_level=getattr(types.ThinkingLevel, lvl)) if lvl in types.ThinkingLevel.__members__ else None


async def call_json(stage: str, ledger: Ledger, contents: list, *, model: str | None = None,
                    schema: Any = None, system: str | None = None, retries: int = 2) -> Any:
    from google.genai import types
    ledger.check_budget(stage)
    cfg = types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json",
                                      response_schema=schema, system_instruction=system, thinking_config=_thinking())
    last: Exception | None = None
    for attempt in range(retries + 1):
        t0 = time.perf_counter()
        try:
            resp = await client().aio.models.generate_content(model=model or SETTINGS.vision_model, contents=contents, config=cfg)
            tin, tout = _usage(resp)
            ledger.add(stage, tin, tout, int((time.perf_counter() - t0) * 1000))
            return parse_json(resp.text)
        except Exception as exc:  # network, quota, malformed JSON
            last = exc
            log.warning("%s call failed (attempt %d): %s", stage, attempt + 1, exc)
            await asyncio.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{stage} failed after retries: {last}")


@dataclass
class GroundedResult:
    text: str
    chunks: list[dict]             # [{uri, title}]
    supports: list[dict]           # [{text, chunk_indices, start, end}]
    queries: list[str]


async def grounded_search(stage: str, ledger: Ledger, prompt: str, *, model: str | None = None) -> GroundedResult:
    """Gemini + Google Search grounding. Returns text plus the grounding metadata
    so callers can check that each number is backed by a retrieved source."""
    from google.genai import types
    ledger.check_budget(stage)
    cfg = types.GenerateContentConfig(temperature=0.0, tools=[types.Tool(google_search=types.GoogleSearch())],
                                      thinking_config=_thinking())
    t0 = time.perf_counter()
    resp = await client().aio.models.generate_content(model=model or SETTINGS.pricing_model, contents=prompt, config=cfg)
    tin, tout = _usage(resp)
    ledger.add(stage, tin, tout, int((time.perf_counter() - t0) * 1000), search=True)
    gm = None
    if resp.candidates:
        gm = getattr(resp.candidates[0], "grounding_metadata", None)
    chunks, supports, queries = [], [], []
    if gm:
        for c in getattr(gm, "grounding_chunks", None) or []:
            web = getattr(c, "web", None)
            if web:
                chunks.append({"uri": web.uri or "", "title": web.title or ""})
        for s in getattr(gm, "grounding_supports", None) or []:
            seg = s.segment
            supports.append({"text": seg.text or "", "start": seg.start_index, "end": seg.end_index,
                             "chunk_indices": list(s.grounding_chunk_indices or [])})
        queries = list(getattr(gm, "web_search_queries", None) or [])
    return GroundedResult(text=resp.text or "", chunks=chunks, supports=supports, queries=queries)
