"""Price evidence verification.

The pricing agent (Gemini + Google Search) proposes offers. An offer becomes a
price only if a retrievable source carries the same number:

* ``page_text`` – we fetched the offer URL ourselves and the amount appears on
  the page next to the currency (or as a schema.org/meta ``price``), and the
  page mentions the product.
* ``grounded_snippet`` – the page could not be fetched (bot wall), but the
  sentence containing the amount is attributed by Gemini's grounding metadata
  to a retrieved search result whose URL we keep. Accepted, but the line goes
  to the review queue.

Anything else is discarded: a number the model wrote with no source is a
fabricated price.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from ..http import get_page
from ..identify.catalogue import coverage
from ..llm import GroundedResult

SYMBOLS = {"USD": ["$", "US$", "USD"], "GBP": ["£", "GBP"], "EUR": ["€", "EUR"], "INR": ["₹", "Rs", "INR", "Rs."]}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_amount(s) -> float | None:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    t = re.sub(r"[^\d.,]", "", str(s))
    if not t:
        return None
    if "," in t and "." in t:
        t = t.replace(",", "") if t.rfind(".") > t.rfind(",") else t.replace(".", "").replace(",", ".")
    elif "," in t:
        head, _, tail = t.rpartition(",")
        t = head.replace(",", "") + "." + tail if len(tail) == 2 else t.replace(",", "")
    try:
        return float(t)
    except ValueError:
        return None


def amount_patterns(amount: float, currency: str) -> list[re.Pattern]:
    whole, cents = f"{amount:.2f}".split(".")
    whole_grp = f"{int(whole):,}"
    nums = {f"{whole}[.,]{cents}", f"{whole_grp.replace(',', '[,. ]?')}[.,]{cents}"}
    if cents == "00":
        nums |= {whole, whole_grp.replace(",", "[,. ]?")}
    syms = [re.escape(x) for x in SYMBOLS.get(currency, [currency])]
    sym = "(?:" + "|".join(syms) + ")"
    pats = []
    for n in nums:
        pats.append(re.compile(rf"{sym}\s?{n}(?!\d)"))
        pats.append(re.compile(rf"(?<![\d.,]){n}\s?{sym}"))
    return pats


def amount_on_page(amount: float, currency: str, text: str, html: str) -> str:
    """Return the matching snippet if the amount is on the page, else ''."""
    for p in amount_patterns(amount, currency):
        m = p.search(text)
        if m:
            a, b = max(0, m.start() - 60), min(len(text), m.end() + 40)
            return text[a:b].strip()
    whole, cents = f"{amount:.2f}".split(".")
    m = re.search(rf'(?i)"?(?:price|lowPrice|highPrice)"?\s*[:=]\s*"?{whole}\.{cents}\b', html or "")
    if m:
        return html[max(0, m.start() - 40): m.end() + 20]
    return ""


@dataclass
class Offer:
    retailer: str
    url: str
    amount: float
    currency: str
    condition: str
    format: str
    quote: str


@dataclass
class VerifiedOffer(Offer):
    verification: str = ""
    evidence: str = ""
    final_url: str = ""
    retrieved_at: str = ""


def offers_from_json(data: dict) -> list[Offer]:
    out = []
    for o in (data or {}).get("offers") or []:
        amt = parse_amount(o.get("price"))
        if amt is None or amt <= 0:
            continue
        out.append(Offer(retailer=str(o.get("retailer") or ""), url=str(o.get("url") or ""), amount=amt,
                         currency=str(o.get("currency") or "").upper(), condition=str(o.get("condition") or ""),
                         format=str(o.get("format") or ""), quote=str(o.get("quote") or "")))
    return out


def _supporting_chunks(g: GroundedResult, amount: float) -> list[dict]:
    whole, cents = f"{amount:.2f}".split(".")
    pat = re.compile(rf"{whole}[.,]{cents}" if cents != "00" else rf"\b{whole}\b")
    hits = []
    for s in g.supports:
        if pat.search(s["text"]):
            for i in s["chunk_indices"]:
                if 0 <= i < len(g.chunks):
                    hits.append(g.chunks[i])
    return hits


async def verify_offer(o: Offer, g: GroundedResult, product_words: str, expected_currency: str) -> VerifiedOffer | None:
    if o.currency and o.currency != expected_currency:
        return None
    vo = VerifiedOffer(**o.__dict__, retrieved_at=now_iso())
    vo.currency = expected_currency
    urls = [o.url] if o.url.startswith("http") else []
    urls += [c["uri"] for c in _supporting_chunks(g, o.amount)]
    resolved: dict[str, str] = {}
    for u in dict.fromkeys(urls):
        final, text, html = await get_page(u)
        resolved[u] = final
        if not text:
            continue
        snippet = amount_on_page(o.amount, expected_currency, text, html)
        if snippet and coverage(product_words, text) >= 0.6:
            vo.verification, vo.evidence, vo.final_url = "page_text", snippet[:240], final
            return vo
    chunks = _supporting_chunks(g, o.amount)
    if chunks:
        seg = next((s["text"] for s in g.supports if f"{o.amount:.2f}" in s["text"] or f"{o.amount:.2f}".replace(".", ",") in s["text"]), "")
        vo.verification = "grounded_snippet"
        vo.evidence = (seg or o.quote)[:240]
        vo.final_url = resolved.get(chunks[0]["uri"], chunks[0]["uri"])
        vo.retailer = vo.retailer or chunks[0]["title"]
        return vo
    return None


async def verify_all(offers: list[Offer], g: GroundedResult, product_words: str, currency: str) -> list[VerifiedOffer]:
    res = await asyncio.gather(*(verify_offer(o, g, product_words, currency) for o in offers[:6]))
    return [r for r in res if r is not None]
