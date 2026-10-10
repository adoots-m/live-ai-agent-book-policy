"""Run the full pipeline on a synthetic bookcase with mocked models.

Shows the packet/report format and exercises every stage without an API key:
    python scripts/demo_synthetic.py      -> sweeps/<id>/report.html
Model-dependent stages (vision, catalogue, pricing) are replaced by stand-ins
that return clearly fake data (retailer "SYNTHETIC"), so nothing here is a real price.
"""
import asyncio
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from synth import make_scene, mock_vision, views  # noqa: E402
import library_claim.sweep as sw  # noqa: E402
from library_claim.identify.catalogue import Resolution  # noqa: E402
from library_claim.pricing.books import BookPrice  # noqa: E402
from library_claim.schemas import ItemCost, Money, UsedMoney  # noqa: E402

scene, books, unit_box = make_scene(5, width_cm=200)
vs = views(scene, 14, seed=3)
H = {}


async def read(img, fid, ledger, tiles=None):
    return mock_vision(fid, H[fid], books, unit_box, (img.shape[1], img.shape[0]), seed=int(fid[1:]))


async def resolve(text, title, author, pub, country="US", measured_h=None):
    if hash(title) % 5 == 0:
        return Resolution(False, 0.4, reason="SYNTHETIC: no catalogue match")
    return Resolution(True, 0.9, title=title.title(), author="Synthetic Author", source_url="https://example.invalid/catalogue")


async def price(title, author, edition, isbn, loc, ledger, flags=None):
    m = Money(amount=10.0, currency=loc.currency, source="SYNTHETIC (not a real price)", url="https://example.invalid/p",
              retrieved_at="1970-01-01T00:00:00+00:00", verification="synthetic")
    return BookPrice(replacement=m, used=UsedMoney(**{**m.model_dump(), "amount": 4.0}, condition_assumed="synthetic"))


async def item(*a, **k):
    loc = a[5]
    return "range", ItemCost(low=1.0, high=2.0, currency=loc.currency, source="SYNTHETIC", url="https://example.invalid"), []


sw.read_frame, sw.resolve, sw.price_book, sw.price_item = read, resolve, price, item


async def main():
    s = sw.SweepSession(country="US", device="synthetic-demo", use_depth=False)
    s.start()
    for i, (img, Hf) in enumerate(vs):
        H[f"f{i:05d}"] = Hf
        await s.add_frame(cv2.imencode(".jpg", img)[1].tobytes(), None, t=i * 0.5)
    await s.finish()
    print("report:", s.dir / "report.html")
    print(s.summary())

asyncio.run(main())
