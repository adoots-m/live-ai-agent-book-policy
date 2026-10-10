"""Currency conversion from ECB reference rates (frankfurter, keyless).

Used only when no local price exists; every converted figure is labelled with
the source amount, rate and rate date."""

from __future__ import annotations

from ..http import get_json

ENDPOINTS = ("https://api.frankfurter.dev/v1/latest", "https://api.frankfurter.app/latest")


async def rate(src: str, dst: str) -> tuple[float, str, str] | None:
    """Return (rate, date, url) to convert 1 unit of src into dst."""
    if src == dst:
        return 1.0, "", ""
    for base in ENDPOINTS:
        data = await get_json(base, {"from": src, "to": dst})
        if data and dst in (data.get("rates") or {}):
            return float(data["rates"][dst]), str(data.get("date", "")), f"{base}?from={src}&to={dst}"
    return None
