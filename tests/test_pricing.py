"""Pricing integrity: a number is accepted only with a retrievable source that carries it."""

import asyncio
import json

import pytest

from library_claim.config import locale_for
from library_claim.llm import GroundedResult, Ledger
from library_claim.pricing import books as pb
from library_claim.pricing import verify as pv


def test_parse_amount_formats():
    assert pv.parse_amount("$12.99") == 12.99
    assert pv.parse_amount("£1,299.00") == 1299.0
    assert pv.parse_amount("12,99 €") == 12.99
    assert pv.parse_amount("1.299,50") == 1299.5
    assert pv.parse_amount("") is None


def test_amount_on_page_needs_currency_or_price_attribute():
    assert pv.amount_on_page(12.99, "USD", "Paperback $12.99 Add to cart", "")
    assert pv.amount_on_page(8.49, "GBP", "Price: £8.49 in stock", "")
    assert not pv.amount_on_page(12.99, "USD", "Page 12.99 of a manual", "")      # bare number is not a price
    assert pv.amount_on_page(12.99, "USD", "no visible", '{"@type":"Offer","price":"12.99"}')
    assert not pv.amount_on_page(12.99, "USD", "Paperback $112.99", "")


def _g(text, supports=(), chunks=()):
    return GroundedResult(text=text, chunks=list(chunks), supports=list(supports), queries=[])


@pytest.fixture()
def pages(monkeypatch):
    store = {}

    async def fake_get_page(url, max_bytes=0):
        return store.get(url, (url, "", ""))
    monkeypatch.setattr(pv, "get_page", fake_get_page)
    return store


def test_verify_page_text(pages):
    pages["https://shop.example/hobbit"] = ("https://shop.example/hobbit", "The Hobbit by J.R.R. Tolkien paperback $14.99", "")
    o = pv.Offer("Shop", "https://shop.example/hobbit", 14.99, "USD", "new", "paperback", "$14.99")
    v = asyncio.run(pv.verify_offer(o, _g(""), "The Hobbit Tolkien", "USD"))
    assert v and v.verification == "page_text" and "$14.99" in v.evidence


def test_verify_rejects_number_not_on_page(pages):
    pages["https://shop.example/hobbit"] = ("https://shop.example/hobbit", "The Hobbit by Tolkien $9.99", "")
    o = pv.Offer("Shop", "https://shop.example/hobbit", 14.99, "USD", "new", "", "")
    assert asyncio.run(pv.verify_offer(o, _g(""), "The Hobbit Tolkien", "USD")) is None


def test_verify_grounded_snippet_when_page_blocked(pages):
    g = _g("Shop sells it for $14.99.", supports=[{"text": "Shop sells it for $14.99.", "chunk_indices": [0], "start": 0, "end": 10}],
           chunks=[{"uri": "https://vertexaisearch.example/redirect/1", "title": "shop.example"}])
    o = pv.Offer("Shop", "https://blocked.example/x", 14.99, "USD", "new", "", "")
    v = asyncio.run(pv.verify_offer(o, g, "The Hobbit", "USD"))
    assert v and v.verification == "grounded_snippet"


def test_verify_rejects_wrong_currency(pages):
    o = pv.Offer("Shop", "https://x", 14.99, "GBP", "new", "", "")
    assert asyncio.run(pv.verify_offer(o, _g(""), "x", "USD")) is None


def _offers(*items):
    return json.dumps({"offers": [{"retailer": r, "url": u, "price": p, "currency": c} for r, u, p, c in items]})


def test_price_book_median_and_fallback_conversion(monkeypatch, pages):
    pages["https://us.new/1"] = ("https://us.new/1", "Dune Frank Herbert paperback $18.00", "")
    pages["https://us.new/2"] = ("https://us.new/2", "Dune Frank Herbert $20.00", "")
    pages["https://uk.used/1"] = ("https://uk.used/1", "Dune Herbert used good £4.50", "")

    async def fake_search(stage, ledger, prompt, model=None):
        if "new retail" in prompt and "United States" in prompt:
            return _g(_offers(("A", "https://us.new/1", "18.00", "USD"), ("B", "https://us.new/2", "20.00", "USD"),
                              ("Hallucinated", "https://nowhere/x", "3.00", "USD")))
        if "used" in prompt and "United States" in prompt:
            return _g('{"offers": []}')
        if "used" in prompt and "United Kingdom" in prompt:
            return _g(_offers(("UKUsed", "https://uk.used/1", "4.50", "GBP")))
        return _g('{"offers": []}')

    async def fake_rate(src, dst):
        return 1.25, "2026-10-09", "https://api.frankfurter.dev/v1/latest?from=GBP&to=USD"
    monkeypatch.setattr(pb, "grounded_search", fake_search)
    monkeypatch.setattr(pb.fx, "rate", fake_rate)
    r = asyncio.run(pb.price_book("Dune", "Frank Herbert", "", "", locale_for("US"), Ledger()))
    assert r.replacement.amount == 19.0 and not r.replacement.converted
    assert r.replacement.url.startswith("https://us.new/")
    assert any("discarded" in m for m in r.review)          # the hallucinated $3 offer was dropped
    assert r.used.converted and r.used.amount == round(4.50 * 1.25, 2)
    assert "GBP 4.50" in r.used.converted_from and r.used.condition_assumed


def test_price_book_above_threshold_goes_to_appraisal(monkeypatch, pages):
    pages["https://rare/1"] = ("https://rare/1", "Ulysses James Joyce first edition $4,500.00", "")

    async def fake_search(stage, ledger, prompt, model=None):
        return _g(_offers(("Rare", "https://rare/1", "4500.00", "USD")))
    monkeypatch.setattr(pb, "grounded_search", fake_search)
    r = asyncio.run(pb.price_book("Ulysses", "James Joyce", "", "", locale_for("US"), Ledger()))
    assert r.appraisal_reason and r.replacement.amount is None and r.used.amount is None


def test_rarity_flags_skip_pricing(monkeypatch):
    async def boom(*a, **k):
        raise AssertionError("must not search")
    monkeypatch.setattr(pb, "grounded_search", boom)
    r = asyncio.run(pb.price_book("X", "Y", "", "", locale_for("GB"), Ledger(), ["signed"]))
    assert "signed" in r.appraisal_reason


def test_locale_is_a_setting():
    assert locale_for("uk").currency == "GBP" and locale_for("US").currency == "USD"
    with pytest.raises(ValueError):
        locale_for("Narnia")
