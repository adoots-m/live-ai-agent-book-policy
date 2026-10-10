"""Totals are computed here, in code, from the packet lines. Never by a model."""

from __future__ import annotations

from ..schemas import Book, Item, Totals


def compute_totals(books: list[Book], items: list[Item], currency: str, shelf_run_m: float | None) -> Totals:
    t = Totals(currency=currency, shelf_run_m=shelf_run_m)
    t.book_count = len(books)
    t.books_identified = sum(1 for b in books if b.status == "identified")
    t.books_unidentified = sum(1 for b in books if b.status == "unidentified")
    t.books_needs_appraisal = sum(1 for b in books if b.status == "needs_appraisal")
    excluded = 0
    for b in books:
        rc, uv = b.replacement_cost.amount, b.used_value.amount
        counted = False
        if b.status == "identified" and rc is not None:
            t.books_replacement_cost += rc
            t.books_priced_replacement += 1
            counted = True
        if b.status == "identified" and uv is not None:
            t.books_used_value += uv
            t.books_priced_used += 1
            counted = True
        if not counted:
            excluded += 1
    for i in items:
        c = i.replacement_cost
        if i.status in ("priced", "range") and c.low is not None and c.high is not None:
            t.items_replacement_cost_low += c.low
            t.items_replacement_cost_high += c.high
        else:
            excluded += 1
    t.excluded_from_totals = excluded
    t.books_replacement_cost = round(t.books_replacement_cost, 2)
    t.books_used_value = round(t.books_used_value, 2)
    t.items_replacement_cost_low = round(t.items_replacement_cost_low, 2)
    t.items_replacement_cost_high = round(t.items_replacement_cost_high, 2)
    return t


def check_totals(packet) -> list[str]:
    """Re-derive totals from the lines and report any mismatch (used by tests and the report)."""
    again = compute_totals(packet.books, packet.items, packet.totals.currency, packet.totals.shelf_run_m)
    errs = []
    for k, v in again.model_dump().items():
        if getattr(packet.totals, k) != v:
            errs.append(f"totals.{k}: packet={getattr(packet.totals, k)!r} recomputed={v!r}")
    return errs
