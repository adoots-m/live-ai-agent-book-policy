"""Identification refuses to guess; the reader parser drops invented titles."""

import asyncio

from library_claim.identify import catalogue as cat
from library_claim.vision.reader import parse_reader_output


def test_reader_drops_title_not_in_transcription():
    raw = {"spines": [
        {"box_2d": [100, 100, 900, 130], "text": "TOLKIEN THE HOBBIT", "title": "The Hobbit", "author": "Tolkien", "legible": True},
        {"box_2d": [100, 140, 900, 170], "text": "THE LO", "title": "The Lord of the Rings", "legible": True},
        {"box_2d": [100, 180, 900, 200], "text": "", "title": "", "legible": False},
        {"box_2d": [0, 0, 0, 0], "text": "degenerate"}]}
    fv = parse_reader_output(raw, "f00001", 1000, 1000)
    assert len(fv.spines) == 3
    a, b, c = fv.spines
    assert a.legible and a.title_read == "The Hobbit" and a.author_read == "Tolkien"
    assert not b.legible and b.title_read == ""          # model 'completed' a partial title -> dropped
    assert not c.legible
    assert a.box[0] == 100 and a.box[2] == 130            # [ymin,xmin,ymax,xmax] -> x0,y0,x1,y1


def test_coverage_and_author():
    assert cat.coverage("The Hobbit", "TOLKIEN THE HOBBIT HARPERCOLLINS") == 1.0
    assert cat.coverage("The Silmarillion", "TOLKIEN THE HOBBIT") == 0.0
    assert cat.author_hit(["J. R. R. Tolkien"], "TOLKIEN THE HOBBIT")


def _cand(title, authors, source="openlibrary", pubs=(), work="/works/OL1W"):
    return cat.Candidate(title=title, authors=list(authors), source=source, url="https://openlibrary.org" + work,
                         work_key=work, publishers=list(pubs))


def test_resolve_accepts_clear_match(monkeypatch):
    async def ol(q):
        return [_cand("The Hobbit", ["J. R. R. Tolkien"], pubs=["HarperCollins"]), _cand("The Hobbit Companion", ["David Day"])]

    async def gb(q, c):
        return []

    async def eds(work, pub, h):
        return None
    monkeypatch.setattr(cat, "search_openlibrary", ol)
    monkeypatch.setattr(cat, "search_googlebooks", gb)
    monkeypatch.setattr(cat, "pick_edition", eds)
    r = asyncio.run(cat.resolve("TOLKIEN THE HOBBIT", "THE HOBBIT", "TOLKIEN", "", "US"))
    assert r.accepted and r.title == "The Hobbit" and r.confidence >= 0.75
    assert r.edition == "" and r.isbn == ""                # no publisher on spine -> no edition claimed


def test_resolve_refuses_weak_match(monkeypatch):
    async def ol(q):
        return [_cand("Atlas Shrugged", ["Ayn Rand"])]

    async def gb(q, c):
        return []
    monkeypatch.setattr(cat, "search_openlibrary", ol)
    monkeypatch.setattr(cat, "search_googlebooks", gb)
    r = asyncio.run(cat.resolve("ATLAS", "ATLAS", "", "", "US"))
    assert not r.accepted and "scored" in r.reason


def test_resolve_refuses_ambiguous(monkeypatch):
    async def ol(q):
        return [_cand("Collected Poems", ["Anne Smith"], work="/works/A"), _cand("Collected Poems", ["Anne Smith"], work="/works/B")]

    async def gb(q, c):
        return [_cand("Poems Collected Edition", ["Anne Smith"], source="googlebooks")]
    monkeypatch.setattr(cat, "search_openlibrary", ol)
    monkeypatch.setattr(cat, "search_googlebooks", gb)
    # same title twice is the same work -> accepted; a different title at equal score would be refused
    r = asyncio.run(cat.resolve("SMITH COLLECTED POEMS", "COLLECTED POEMS", "SMITH", "", "US"))
    assert r.accepted


def test_edition_picked_by_publisher_and_height(monkeypatch):
    async def fake_get_json(url, params=None):
        return {"entries": [
            {"key": "/books/OL1M", "publishers": ["Penguin"], "isbn_13": ["9780000000001"], "physical_dimensions": "19.8 x 12.9 x 2 centimeters", "physical_format": "Paperback"},
            {"key": "/books/OL2M", "publishers": ["Penguin"], "isbn_13": ["9780000000002"], "physical_dimensions": "24 x 16 x 3 centimeters", "physical_format": "Hardcover"},
            {"key": "/books/OL3M", "publishers": ["Vintage"], "isbn_13": ["9780000000003"]}]}
    monkeypatch.setattr(cat, "get_json", fake_get_json)
    ed = asyncio.run(cat.pick_edition("/works/OL1W", "PENGUIN", 19.5))
    assert ed["isbn"] == "9780000000001" and "height" in ed["basis"]
    ed2 = asyncio.run(cat.pick_edition("/works/OL1W", "PENGUIN", None))
    assert ed2["isbn"] == "" and "not determinable" in ed2["basis"]


def test_parse_height():
    assert cat.parse_height("23 x 15 x 2 centimeters") == 23.0
    assert cat.parse_height("9.2 x 6.1 x 1 inches") == round(9.2 * 2.54, 1)
    assert cat.parse_height("24.00 cm") == 24.0
