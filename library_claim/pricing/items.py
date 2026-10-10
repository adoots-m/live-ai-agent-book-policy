"""Stage: price non-book contents at local replacement cost.

* Brand/model legible -> search that model; status ``priced`` (low/high = the
  verified offers' min/max for that model).
* Otherwise -> search comparable new items matching the visible description and
  size; status ``range`` (25th-75th percentile of >= 2 verified offers).
* Art, portraits, sculpture -> ``needs_appraisal`` unless the policyholder said
  it is a print/poster, in which case it is priced as a framed print.
"""

from __future__ import annotations

from statistics import quantiles

from ..config import Locale
from ..llm import Ledger, grounded_search, parse_json
from ..schemas import ItemCost, Money
from .books import COUNTRY_NAMES
from .verify import offers_from_json, verify_all

PROMPT = """Search the web for the current price to buy NEW, in {country_name}, {what}.
Size seen: {size}. Material: {material}.
Return ONLY JSON:
{{"offers": [{{"retailer": "...", "url": "<product page URL>", "price": "<number as shown>",
  "currency": "{currency}", "condition": "new", "format": "<product name>",
  "quote": "<exact text from the page or result showing the price>"}}]}}
At most 5 offers from retailers selling in {country_name}, in {currency}. Never estimate or recall a price.
If nothing comparable is found, return {{"offers": []}}."""


async def price_item(category: str, description: str, brand_model: str, material: str, dims: dict,
                     loc: Locale, ledger: Ledger, is_art: bool, user_says_print: bool | None) -> tuple[str, ItemCost, list[str]]:
    review: list[str] = []
    if is_art and not user_says_print:
        return "needs_appraisal", ItemCost(currency=loc.currency), ["art/portrait: original vs print not confirmed; needs appraisal"]
    size = ", ".join(f"{k} {v:.0f} cm" for k, v in dims.items() if v) or "unknown"
    what = (f"a {brand_model} ({description})" if brand_model else
            f"an item comparable to: {description}" + (" (a framed print / poster, not an original)" if is_art else ""))
    g = await grounded_search("pricing_items", ledger, PROMPT.format(country_name=COUNTRY_NAMES.get(loc.country, loc.country),
                                                                      what=what, size=size, material=material or "unknown",
                                                                      currency=loc.currency))
    try:
        offers = offers_from_json(parse_json(g.text))
    except Exception:
        offers = []
    words = brand_model or " ".join(description.split()[-2:])
    verified = await verify_all(offers, g, words, loc.currency)
    if len(offers) > len(verified):
        review.append(f"{len(offers) - len(verified)} proposed offer(s) discarded: price not on a retrievable page")
    if not verified:
        return "unpriced", ItemCost(currency=loc.currency), review + ["no verifiable local price found"]
    amts = sorted(o.amount for o in verified)
    srcs = [Money(amount=o.amount, currency=loc.currency, source=o.retailer, url=o.final_url or o.url,
                  retrieved_at=o.retrieved_at, evidence=o.evidence, verification=o.verification) for o in verified]
    first = srcs[0]
    if any(o.verification == "grounded_snippet" for o in verified):
        review.append("some item prices backed by search snippets only")
    if brand_model:
        cost = ItemCost(low=amts[0], high=amts[-1], currency=loc.currency, source=f"{len(srcs)} verified offers for {brand_model}",
                        url=first.url, retrieved_at=first.retrieved_at, sources=srcs)
        return "priced", cost, review
    if len(amts) < 2:
        review.append("range rests on a single comparable offer")
        lo = hi = amts[0]
    else:
        q = quantiles(amts, n=4, method="inclusive")
        lo, hi = round(q[0], 2), round(q[2], 2)
    cost = ItemCost(low=lo, high=hi, currency=loc.currency, source=f"interquartile range of {len(srcs)} verified comparable offers",
                    url=first.url, retrieved_at=first.retrieved_at, sources=srcs)
    return "range", cost, review
