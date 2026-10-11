"""Stage: price one identified book in the policyholder's market.

Two figures per book:
* replacement_cost – a NEW copy (same edition if known, else the current
  in-print edition of the same work) from a retailer in the country.
* used_value       – a USED copy in "Good" condition from a used marketplace in
  the country.

Each figure is the median of *verified* offers (pricing/verify.py) and points
at the offer closest to that median. No local offer -> try the fallback market
and convert at the ECB rate (labelled converted). Nothing -> empty + excluded.
Any verified offer above the locale's appraisal threshold, or any rarity flag,
stops auto-pricing: the line becomes needs_appraisal with the evidence kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

from ..config import LOCALES, Locale
from ..llm import BudgetExceeded, Ledger, grounded_search, parse_json
from ..schemas import Money, UsedMoney
from . import fx
from .verify import VerifiedOffer, offers_from_json, verify_all

FALLBACK_MARKET = {"US": "GB", "GB": "US", "IN": "US", "DE": "GB"}
CONDITION_USED = "Good (used; typical shelf wear, no damage noted)"

PROMPT = """Search the web for current {kind} prices of this book in {country_name}.
Book: "{title}" by {author}{edition}{isbn}
Look on {sources} (or other retailers that ship within {country_name}).
{condition_rule}
Return ONLY JSON:
{{"offers": [{{"retailer": "...", "url": "<the product page URL>", "price": "<number as shown>",
  "currency": "{currency}", "condition": "new|used", "format": "hardcover|paperback|...",
  "quote": "<the exact text from the page or result that shows the price>"}}]}}
List at most 5 offers you actually found in search results. Prices must be in {currency}
for sellers in {country_name}. Exclude shipping. If you find none, return {{"offers": []}}.
Never estimate or recall a price."""

COUNTRY_NAMES = {"US": "the United States", "GB": "the United Kingdom", "IN": "India", "DE": "Germany"}


@dataclass
class BookPrice:
    replacement: Money = field(default_factory=Money)
    used: UsedMoney = field(default_factory=UsedMoney)
    candidates: list[dict] = field(default_factory=list)
    appraisal_reason: str = ""
    review: list[str] = field(default_factory=list)


def _pick(offers: list[VerifiedOffer]) -> tuple[float, VerifiedOffer]:
    m = round(median(o.amount for o in offers), 2)
    return m, min(offers, key=lambda o: abs(o.amount - m))


async def _search(kind: str, loc: Locale, title: str, author: str, edition: str, isbn: str, ledger: Ledger):
    rule = ("Only NEW copies (not used, not marketplace resellers' used listings, not ebooks or audiobooks)."
            if kind == "new" else
            "Only USED copies in good / very good condition. Ignore new copies, ebooks, collectible or signed copies.")
    prompt = PROMPT.format(kind="new retail" if kind == "new" else "used", country_name=COUNTRY_NAMES.get(loc.country, loc.country),
                           title=title, author=author or "unknown author",
                           edition=f", edition: {edition.split(' [')[0]}" if edition else "", isbn=f", ISBN {isbn}" if isbn else "",
                           sources=", ".join(loc.used_sources if kind == "used" else loc.new_sources),
                           condition_rule=rule, currency=loc.currency)
    g = await grounded_search(f"pricing_books_{kind}", ledger, prompt)
    try:
        offers = offers_from_json(parse_json(g.text))
    except Exception:
        offers = []
    words = f"{title} {author.split(',')[0] if author else ''}"
    verified = await verify_all(offers, g, words, loc.currency)
    return offers, verified


async def _figure(kind: str, loc: Locale, title, author, edition, isbn, ledger, out: BookPrice) -> tuple[Money | None, list[VerifiedOffer]]:
    offers, verified = await _search(kind, loc, title, author, edition, isbn, ledger)
    out.candidates += [{"kind": kind, "market": loc.country, "retailer": o.retailer, "amount": o.amount,
                        "currency": loc.currency, "url": o.final_url or o.url, "verification": o.verification,
                        "evidence": o.evidence} for o in verified]
    dropped = len(offers) - len(verified)
    if dropped:
        out.review.append(f"{dropped} {kind} offer(s) proposed by search were discarded: amount not found on a retrievable page")
    if verified:
        amt, src = _pick(verified)
        m = Money(amount=amt, currency=loc.currency, source=f"{src.retailer} (median of {len(verified)} verified {kind} offers)",
                  url=src.final_url or src.url, retrieved_at=src.retrieved_at, evidence=src.evidence,
                  verification=src.verification)
        if src.verification == "grounded_snippet":
            out.review.append(f"{kind} price backed by search snippet; page itself could not be fetched")
        return m, verified
    # fallback market + FX conversion
    other = LOCALES.get(FALLBACK_MARKET.get(loc.country, "US"))
    if other is None or other.country == loc.country:
        return None, []
    _, verified2 = await _search(kind, other, title, author, edition, isbn, ledger)
    if not verified2:
        return None, []
    amt, src = _pick(verified2)
    r = await fx.rate(other.currency, loc.currency)
    if r is None:
        out.review.append(f"{kind} price only found in {other.country} and FX rate unavailable")
        return None, verified2
    rate, date, url = r
    m = Money(amount=round(amt * rate, 2), currency=loc.currency,
              source=f"{src.retailer} ({other.country} market, converted)", url=src.final_url or src.url,
              retrieved_at=src.retrieved_at, converted=True,
              converted_from=f"{other.currency} {amt:.2f} @ {rate:.4f} (ECB {date}, {url})",
              evidence=src.evidence, verification=src.verification)
    out.review.append(f"no {loc.country} {kind} price found; converted from {other.country}")
    return m, verified2


async def price_book(title: str, author: str, edition: str, isbn: str, loc: Locale, ledger: Ledger,
                     rarity_flags: list[str] | None = None) -> BookPrice:
    out = BookPrice()
    if rarity_flags:
        out.appraisal_reason = "rarity indicators: " + ", ".join(rarity_flags)
        return out
    try:
        new, v_new = await _figure("new", loc, title, author, edition, isbn, ledger, out)
        used, v_used = await _figure("used", loc, title, author, edition, isbn, ledger, out)
    except BudgetExceeded as exc:
        out.review.append(f"not priced: {exc}")
        return out
    top = max([o.amount for o in v_new + v_used] + [0])
    if new and new.converted:
        top = max(top, new.amount or 0)
    if top > loc.appraisal_threshold:
        out.appraisal_reason = f"verified offer {loc.symbol}{top:.2f} exceeds appraisal threshold {loc.symbol}{loc.appraisal_threshold:.0f}"
        return out
    if new:
        out.replacement = new
    if used:
        out.used = UsedMoney(**used.model_dump(), condition_assumed=CONDITION_USED)
    return out
