"""try-photo runs end to end on a synthetic shelf photo with the reader and catalogue mocked."""

import asyncio

import cv2
import numpy as np

from synth import make_scene, mock_vision


def test_try_photo(tmp_path, monkeypatch, capsys):
    import library_claim.doctor as doc
    import library_claim.identify.catalogue as cat
    import library_claim.vision.reader as rd
    scene, books, ub = make_scene(3, width_cm=60)
    p = tmp_path / "shelf.jpg"
    cv2.imwrite(str(p), scene)

    async def read(img, fid, ledger, tiles=None):
        return mock_vision(fid, np.eye(3), books, ub, (img.shape[1], img.shape[0]), jitter=0)

    async def resolve(text, title, author, pub, country="US", measured_h=None):
        return cat.Resolution(True, 0.9, title=title.title(), author="X")
    monkeypatch.setattr(rd, "read_frame", read)
    monkeypatch.setattr(cat, "resolve", resolve)
    asyncio.run(doc._try_photo(str(p), "US"))
    out = capsys.readouterr().out
    assert "marker: found" in out and "identified" in out
    assert (tmp_path / "shelf_annotated.jpg").exists()
    # heights measured from the marker match the synthetic truth
    line = next(l for l in out.splitlines() if "identified" in l and l.strip()[0].isdigit())
    assert float(line.split()[1]) > 10
