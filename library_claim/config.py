"""Runtime settings. Everything that changes per deployment or per policyholder
lives here and is read from the environment (see .env.example).

Locale is a *setting*: the same sweep can be re-priced for another country with
`python -m library_claim.cli reprice --country GB`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Locale:
    country: str          # ISO 3166-1 alpha-2, e.g. "US"
    currency: str         # ISO 4217, e.g. "USD"
    symbol: str
    # Retailers/marketplaces the pricing agent is told to prefer for this market.
    new_sources: tuple[str, ...]
    used_sources: tuple[str, ...]
    appraisal_threshold: float  # per-line value above which we stop and ask a human


LOCALES: dict[str, Locale] = {
    "US": Locale(
        "US", "USD", "$",
        new_sources=("bookshop.org", "barnesandnoble.com", "amazon.com", "target.com"),
        used_sources=("abebooks.com", "thriftbooks.com", "ebay.com", "betterworldbooks.com", "biblio.com"),
        appraisal_threshold=_f("APPRAISAL_THRESHOLD_USD", 250),
    ),
    "GB": Locale(
        "GB", "GBP", "£",
        new_sources=("waterstones.com", "blackwells.co.uk", "amazon.co.uk", "uk.bookshop.org", "whsmith.co.uk"),
        used_sources=("abebooks.co.uk", "worldofbooks.com", "ebay.co.uk", "wob.com", "oxfam.org.uk"),
        appraisal_threshold=_f("APPRAISAL_THRESHOLD_GBP", 200),
    ),
    "IN": Locale(
        "IN", "INR", "₹",
        new_sources=("amazon.in", "flipkart.com", "crossword.in", "bookswagon.com"),
        used_sources=("amazon.in", "bookchor.com", "ebay.com"),
        appraisal_threshold=_f("APPRAISAL_THRESHOLD_INR", 20000),
    ),
    "DE": Locale(
        "DE", "EUR", "€",
        new_sources=("thalia.de", "hugendubel.de", "amazon.de"),
        used_sources=("zvab.com", "booklooker.de", "medimops.de", "ebay.de"),
        appraisal_threshold=_f("APPRAISAL_THRESHOLD_EUR", 230),
    ),
}


def locale_for(country: str) -> Locale:
    key = (country or "").strip().upper()
    if key == "UK":
        key = "GB"
    if key not in LOCALES:
        raise ValueError(f"Unsupported country {country!r}. Add it to LOCALES in config.py.")
    return LOCALES[key]


@dataclass(frozen=True)
class Settings:
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", ""))
    live_model: str = field(default_factory=lambda: os.getenv("LCA_LIVE_MODEL", "gemini-3.8-live"))
    vision_model: str = field(default_factory=lambda: os.getenv("LCA_VISION_MODEL", "gemini-3.8-flash"))
    pricing_model: str = field(default_factory=lambda: os.getenv("LCA_PRICING_MODEL", "gemini-3.8-flash"))
    voice: str = field(default_factory=lambda: os.getenv("LCA_VOICE", "Kore"))
    default_country: str = field(default_factory=lambda: os.getenv("LCA_COUNTRY", "US"))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("LCA_DATA_DIR", str(ROOT / "sweeps"))))
    # Printed reference marker (ArUco DICT_4X4_50, id 0..9). Side length of the
    # black square, measured with a ruler after printing.
    marker_side_cm: float = field(default_factory=lambda: _f("LCA_MARKER_SIDE_CM", 15.0))
    # Horizontal field of view of the phone's main camera, used only for the
    # room-depth stage. Refined per sweep from the marker when it is visible.
    camera_hfov_deg: float = field(default_factory=lambda: _f("LCA_CAMERA_HFOV_DEG", 67.0))
    vision_concurrency: int = field(default_factory=lambda: int(_f("LCA_VISION_CONCURRENCY", 4)))
    pricing_concurrency: int = field(default_factory=lambda: int(_f("LCA_PRICING_CONCURRENCY", 6)))
    depth_model: str = field(default_factory=lambda: os.getenv("LCA_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"))
    http_timeout_s: float = field(default_factory=lambda: _f("LCA_HTTP_TIMEOUT_S", 12))
    # Cost report rates, USD. Defaults = Gemini API paid tier as published at
    # https://ai.google.dev/gemini-api/docs/pricing on 2026-10-11 (gemini-3.8-flash
    # $0.75 in / $3.75 out per 1M tokens until 2026-12-31; Live audio in $0.005/min,
    # video in $0.002/min, audio out $0.018/min; Search grounding $14 / 1,000 after
    # 5,000 free per month). Override when prices change.
    price_in_per_m: float = field(default_factory=lambda: _f("LCA_PRICE_IN_PER_M", 0.75))
    price_out_per_m: float = field(default_factory=lambda: _f("LCA_PRICE_OUT_PER_M", 3.75))
    live_price_per_min: float = field(default_factory=lambda: _f("LCA_LIVE_PRICE_PER_MIN", 0.025))
    search_price_per_call: float = field(default_factory=lambda: _f("LCA_SEARCH_PRICE_PER_CALL", 0.014))
    # Live API per-1M-token rates by modality (used when the Live session reports usage).
    live_in_text_per_m: float = field(default_factory=lambda: _f("LCA_LIVE_IN_TEXT_PER_M", 0.75))
    live_in_audio_per_m: float = field(default_factory=lambda: _f("LCA_LIVE_IN_AUDIO_PER_M", 3.00))
    live_in_video_per_m: float = field(default_factory=lambda: _f("LCA_LIVE_IN_VIDEO_PER_M", 1.00))
    live_out_text_per_m: float = field(default_factory=lambda: _f("LCA_LIVE_OUT_TEXT_PER_M", 4.50))
    live_out_audio_per_m: float = field(default_factory=lambda: _f("LCA_LIVE_OUT_AUDIO_PER_M", 12.00))
    # Guards: pricing searches stop (lines go to review) once this sweep's
    # estimated spend passes the cap; vision reads stop after this many calls.
    max_usd_per_sweep: float = field(default_factory=lambda: _f("LCA_MAX_USD_PER_SWEEP", 8.0))
    max_vision_calls: int = field(default_factory=lambda: int(_f("LCA_MAX_VISION_CALLS", 250)))


SETTINGS = Settings()
