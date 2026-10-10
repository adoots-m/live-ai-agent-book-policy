"""Shared async HTTP client with an on-disk cache (so a re-price or re-run does
not hammer free APIs, and so a test fixture can stand in for the network)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path

import httpx

from .config import SETTINGS

log = logging.getLogger(__name__)
UA = "LibraryClaimAgent/0.1 (insurance contents inventory research prototype)"
_client: httpx.AsyncClient | None = None
CACHE_DIR = SETTINGS.data_dir / "_http_cache"
CACHE_TTL_S = 24 * 3600


def _cli() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=SETTINGS.http_timeout_s, follow_redirects=True,
                                    headers={"User-Agent": UA, "Accept-Language": "en"})
    return _client


def _key(url: str, params: dict | None) -> Path:
    h = hashlib.sha256((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()[:32]
    return CACHE_DIR / f"{h}.json"


async def get_json(url: str, params: dict | None = None, retries: int = 2) -> dict | None:
    p = _key(url, params)
    if p.exists() and time.time() - p.stat().st_mtime < CACHE_TTL_S:
        return json.loads(p.read_text(encoding="utf-8"))
    for attempt in range(retries + 1):
        try:
            r = await _cli().get(url, params=params)
            if r.status_code == 429:
                await asyncio.sleep(2 * (attempt + 1))
                continue
            if r.status_code >= 400:
                log.info("GET %s -> %s", url, r.status_code)
                return None
            data = r.json()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data), encoding="utf-8")
            return data
        except (httpx.HTTPError, ValueError) as exc:
            log.info("GET %s failed: %s", url, exc)
            await asyncio.sleep(1 + attempt)
    return None


async def get_page(url: str, max_bytes: int = 2_000_000) -> tuple[str, str, str]:
    """Fetch a page; return (final_url, text, raw_html). Empty strings on failure."""
    try:
        r = await _cli().get(url, headers={"Accept": "text/html,application/xhtml+xml"})
        if r.status_code >= 400:
            return str(r.url), "", ""
        html = r.text[:max_bytes]
        return str(r.url), html_to_text(html), html
    except httpx.HTTPError:
        return url, "", ""


def html_to_text(html: str) -> str:
    import re
    html = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
    # keep JSON-LD / meta price attributes, which retailers often use instead of visible text
    metas = " ".join(re.findall(r'(?i)(?:content|value)="([^"]{1,40})"', html))
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    text = re.sub(r"&pound;|&#163;", "£", text)
    text = re.sub(r"&amp;", "&", text)
    return re.sub(r"\s+", " ", text + " " + metas)
