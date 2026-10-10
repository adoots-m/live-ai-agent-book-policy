"""Stage: resolve a spine transcription to a catalogue record.

Inputs are only what the reader transcribed. Candidates come from Open Library
and Google Books (both keyless). A candidate is accepted only if its title is
*found in the spine text* (token coverage) and the author or a long title
corroborates it. Edition is chosen only when the spine's publisher matches an
edition, and among those, the one whose catalogue height agrees with the
measured spine height. Otherwise the edition stays empty.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from ..http import get_json

OL_SEARCH = "https://openlibrary.org/search.json"
OL_WORK_EDITIONS = "https://openlibrary.org{work}/editions.json"
GB_SEARCH = "https://www.googleapis.com/books/v1/volumes"

STOP = {"the", "a", "an", "of", "and", "&", "in", "on", "to", "for", "by", "with", "de", "la", "le"}
ACCEPT = 0.75


def tokens(s: str) -> list[str]:
    return [t for t in re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).split() if t]


def _tok_in(t: str, bag: list[str]) -> bool:
    return any(t == b or (len(t) > 3 and SequenceMatcher(None, t, b).ratio() >= 0.84) for b in bag)


def coverage(phrase: str, text: str) -> float:
    """Share of meaningful tokens of ``phrase`` that appear in ``text`` (fuzzy for OCR slips)."""
    pt = [t for t in tokens(phrase) if t not in STOP] or tokens(phrase)
    if not pt:
        return 0.0
    bag = tokens(text)
    return sum(1 for t in pt if _tok_in(t, bag)) / len(pt)


def author_hit(authors: list[str], text: str) -> bool:
    bag = tokens(text)
    for a in authors or []:
        parts = [p for p in tokens(a) if len(p) > 1]
        if parts and _tok_in(parts[-1], bag):
            return True
    return False


@dataclass
class Candidate:
    title: str
    authors: list[str]
    source: str                     # openlibrary | googlebooks
    url: str
    work_key: str = ""
    publishers: list[str] = field(default_factory=list)
    isbns: list[str] = field(default_factory=list)
    first_year: int | None = None
    height_cm: float | None = None
    gb_id: str = ""
    score: float = 0.0
    title_cov: float = 0.0
    author_ok: bool = False


@dataclass
class Resolution:
    accepted: bool
    confidence: float
    title: str = ""
    author: str = ""
    publisher: str = ""
    edition: str = ""
    isbn: str = ""
    height_cm: float | None = None
    year: int | None = None          # edition year, only when an edition was chosen
    first_year: int | None = None    # first publication of the work
    source_url: str = ""
    reason: str = ""
    candidates: list[dict] = field(default_factory=list)


def score(c: Candidate, spine_text: str, publisher_read: str) -> float:
    c.title_cov = coverage(c.title.split(":")[0], spine_text)
    c.author_ok = author_hit(c.authors, spine_text)
    pub_ok = bool(publisher_read) and any(coverage(publisher_read, p) >= 0.6 for p in c.publishers)
    long_title = len([t for t in tokens(c.title.split(":")[0]) if t not in STOP]) >= 3
    s = 0.6 * c.title_cov + 0.3 * (1.0 if c.author_ok else (0.5 if long_title and c.title_cov == 1 else 0.0)) + 0.1 * pub_ok
    # Penalise candidates whose title is much longer than anything on the spine
    extra = len(tokens(c.title.split(":")[0])) - len(tokens(spine_text))
    if extra > 2:
        s -= 0.1 * extra
    c.score = round(max(0.0, min(1.0, s)), 3)
    return c.score


def parse_height(s: str | None) -> float | None:
    if not s:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:x|×)", s) or re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    v = float(m.group(1))
    if "inch" in s.lower() or " in" in s.lower():
        v *= 2.54
    elif " mm" in s.lower():
        v /= 10
    return round(v, 1) if 8 <= v <= 60 else None


async def search_openlibrary(query: str) -> list[Candidate]:
    data = await get_json(OL_SEARCH, {"q": query, "limit": 5,
                                      "fields": "key,title,author_name,publisher,first_publish_year,isbn"})
    out = []
    for d in (data or {}).get("docs", []):
        out.append(Candidate(title=d.get("title", ""), authors=d.get("author_name", []) or [], source="openlibrary",
                             url=f"https://openlibrary.org{d.get('key', '')}", work_key=d.get("key", ""),
                             publishers=(d.get("publisher") or [])[:40], isbns=(d.get("isbn") or [])[:60],
                             first_year=d.get("first_publish_year")))
    return out


async def search_googlebooks(query: str, country: str) -> list[Candidate]:
    data = await get_json(GB_SEARCH, {"q": query, "maxResults": 5, "country": country, "printType": "books"})
    out = []
    for it in (data or {}).get("items", []) or []:
        v = it.get("volumeInfo", {})
        ids = [i.get("identifier", "") for i in v.get("industryIdentifiers", []) if i.get("type", "").startswith("ISBN")]
        year = None
        if re.match(r"\d{4}", v.get("publishedDate", "")):
            year = int(v["publishedDate"][:4])
        out.append(Candidate(title=v.get("title", ""), authors=v.get("authors", []) or [], source="googlebooks",
                             url=v.get("canonicalVolumeLink") or f"https://books.google.com/books?id={it.get('id')}",
                             publishers=[v["publisher"]] if v.get("publisher") else [], isbns=ids, first_year=year,
                             height_cm=parse_height((v.get("dimensions") or {}).get("height")), gb_id=it.get("id", "")))
    return out


async def pick_edition(work_key: str, publisher_read: str, measured_h: float | None) -> dict | None:
    """Edition whose publisher matches the spine; tie-break on catalogue height."""
    if not work_key or not publisher_read:
        return None
    data = await get_json(OL_WORK_EDITIONS.format(work=work_key), {"limit": 100})
    eds = []
    for e in (data or {}).get("entries", []):
        pubs = e.get("publishers") or []
        if not any(coverage(publisher_read, p) >= 0.6 for p in pubs):
            continue
        isbn = (e.get("isbn_13") or e.get("isbn_10") or [""])[0]
        eds.append({"key": e.get("key", ""), "publisher": pubs[0], "isbn": isbn,
                    "publish_date": e.get("publish_date", ""), "format": e.get("physical_format", ""),
                    "height_cm": parse_height(e.get("physical_dimensions"))})
    if not eds:
        return None
    if measured_h:
        sized = [e for e in eds if e["height_cm"] and abs(e["height_cm"] - measured_h) <= 1.5]
        if len(sized) >= 1:
            sized.sort(key=lambda e: abs(e["height_cm"] - measured_h))
            return {**sized[0], "basis": "publisher on spine + catalogue height matches measured spine"}
    if len(eds) == 1:
        return {**eds[0], "basis": "only edition from the publisher on the spine"}
    return {"key": "", "publisher": eds[0]["publisher"], "isbn": "", "publish_date": "", "format": "",
            "height_cm": None, "basis": f"{len(eds)} editions from this publisher; format not determinable"}


async def resolve(spine_text: str, title_read: str, author_read: str, publisher_read: str,
                  country: str = "US", measured_h: float | None = None) -> Resolution:
    if len(tokens(spine_text)) == 0:
        return Resolution(False, 0.0, reason="no readable text")
    q = " ".join(x for x in (title_read or spine_text, author_read) if x)
    cands = await search_openlibrary(q)
    gb_q = (f"intitle:{title_read}" + (f"+inauthor:{author_read}" if author_read else "")) if title_read else spine_text
    cands += await search_googlebooks(gb_q, country)
    for c in cands:
        score(c, spine_text, publisher_read)
    cands.sort(key=lambda c: c.score, reverse=True)
    summary = [{"title": c.title, "authors": c.authors[:2], "source": c.source, "score": c.score, "url": c.url} for c in cands[:4]]
    if not cands or cands[0].score < ACCEPT or cands[0].title_cov < 0.8:
        why = "no catalogue match" if not cands else f"best match '{cands[0].title}' scored {cands[0].score:.2f} (< {ACCEPT})"
        return Resolution(False, cands[0].score if cands else 0.0, reason=why, candidates=summary)
    best = cands[0]
    # Two different works scoring nearly the same = ambiguous, refuse.
    rival = next((c for c in cands[1:] if c.score >= best.score - 0.05 and coverage(c.title, best.title) < 0.6), None)
    if rival:
        return Resolution(False, best.score, reason=f"ambiguous between '{best.title}' and '{rival.title}'", candidates=summary)
    res = Resolution(True, best.score, title=best.title, author=", ".join(best.authors[:2]), source_url=best.url,
                     first_year=best.first_year, height_cm=best.height_cm, candidates=summary)
    work_key = best.work_key or next((c.work_key for c in cands if c.source == "openlibrary" and coverage(c.title, best.title) >= 0.8), "")
    ed = await pick_edition(work_key, publisher_read, measured_h)
    if ed:
        res.publisher = ed["publisher"]
        res.isbn = ed["isbn"]
        res.edition = " ".join(x for x in (ed["publisher"], ed["format"], ed["publish_date"]) if x) + f" [{ed['basis']}]"
        res.height_cm = ed["height_cm"] or res.height_cm
        if ed.get("key"):
            res.source_url = f"https://openlibrary.org{ed['key']}"
        if ed.get("publish_date") and re.search(r"\d{4}", ed["publish_date"]):
            res.year = int(re.search(r"\d{4}", ed["publish_date"]).group(0))
    elif best.source == "googlebooks" and best.isbns and coverage(publisher_read, " ".join(best.publishers)) >= 0.6:
        res.publisher, res.isbn = best.publishers[0], best.isbns[0]
        res.edition = f"{best.publishers[0]} [publisher on spine matches Google Books volume]"
    return res
