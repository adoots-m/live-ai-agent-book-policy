"""Data contracts.

* Observation models (FrameRecord, SpineObs, ItemObs) are what each stage emits.
* Packet models (Book, Item, Room, Totals, ClaimPacket) are the output contract
  from the brief. Field names in the brief are kept verbatim; extra fields are
  additive. Unknown values are ``None`` (serialised as ``null``), never a
  plausible-looking default: "an empty field is correct when the system does
  not know".
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels of the source frame


# --------------------------------------------------------------------------- capture
class Imu(BaseModel):
    alpha: Optional[float] = None  # DeviceOrientation, degrees
    beta: Optional[float] = None
    gamma: Optional[float] = None
    gx: Optional[float] = None     # accelerationIncludingGravity, m/s^2, device frame
    gy: Optional[float] = None
    gz: Optional[float] = None


class FrameQuality(BaseModel):
    sharpness: float               # variance of Laplacian on a 640px grey copy
    glare_frac: float              # share of near-saturated pixels
    dark_frac: float               # share of near-black pixels
    motion_px: Optional[float] = None  # median ORB displacement vs previous frame
    issues: list[str] = Field(default_factory=list)  # "blur" | "glare" | "dark" | "fast"

    @property
    def ok(self) -> bool:
        return not self.issues


class FrameRecord(BaseModel):
    id: str
    t: float                       # seconds since sweep start
    path: str                      # relative to the sweep directory
    width: int
    height: int
    imu: Imu = Field(default_factory=Imu)
    quality: Optional[FrameQuality] = None
    keyframe: bool = False


# --------------------------------------------------------------------------- vision
class SpineObs(BaseModel):
    obs_id: str
    frame_id: str
    box: Box
    orientation: Literal["vertical", "horizontal"] = "vertical"
    text: str = ""                 # verbatim transcription, may be empty
    title_read: str = ""           # only what is printed, split by the reader
    author_read: str = ""
    publisher_read: str = ""
    legible: bool = False
    flags: list[str] = Field(default_factory=list)  # signed, leather, gilt, first_edition_marked, antique, dust_jacket_worn
    row: Optional[int] = None      # shelf row index within the frame, top = 0


class ItemObs(BaseModel):
    obs_id: str
    frame_id: str
    box: Box
    category: str                  # shelving, furniture, lamp, art, rug, electronics, appliance, decor, other
    description: str
    material: str = ""
    brand_model: str = ""          # only if legible on the item
    is_art: bool = False


class ShelfUnitObs(BaseModel):
    obs_id: str
    frame_id: str
    box: Box
    rows: int = 0


class FrameVision(BaseModel):
    frame_id: str
    spines: list[SpineObs] = Field(default_factory=list)
    items: list[ItemObs] = Field(default_factory=list)
    shelf_units: list[ShelfUnitObs] = Field(default_factory=list)
    capture_feedback: str = ""     # short note from the reader, e.g. "glare on top shelf"
    unreadable_fraction: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0


# --------------------------------------------------------------------------- packet
class Money(BaseModel):
    amount: Optional[float] = None
    currency: str = ""
    source: str = ""
    url: str = ""
    retrieved_at: str = ""
    converted: bool = False
    converted_from: str = ""       # e.g. "GBP 9.99 @ 1.2710 (ECB 2026-10-09)"
    evidence: str = ""             # the text on the page / snippet that carries the number
    verification: str = ""         # "page_text" | "api" | "grounded_snippet" | ""


class UsedMoney(Money):
    condition_assumed: str = ""


class Book(BaseModel):
    id: str
    shelf: str = ""
    position: int = 0
    frame_ref: str = ""
    frame_refs: list[str] = Field(default_factory=list)
    frame_box: Optional[Box] = None
    status: Literal["identified", "unidentified", "needs_appraisal"] = "unidentified"
    title: str = ""
    author: str = ""
    edition: str = ""
    publisher: str = ""
    isbn: str = ""
    spine_text: str = ""
    orientation: str = "vertical"
    spine_height_cm: Optional[float] = None
    spine_thickness_cm: Optional[float] = None
    scale_method: str = ""
    measure_frame_ref: str = ""
    id_confidence: float = 0.0
    id_source: str = ""            # catalogue record URL used for the match
    replacement_cost: Money = Field(default_factory=Money)
    used_value: UsedMoney = Field(default_factory=UsedMoney)
    excluded_reason: str = ""
    flags: list[str] = Field(default_factory=list)
    user_notes: list[str] = Field(default_factory=list)


class ItemCost(BaseModel):
    low: Optional[float] = None
    high: Optional[float] = None
    currency: str = ""
    source: str = ""
    url: str = ""
    retrieved_at: str = ""
    sources: list[Money] = Field(default_factory=list)


class Dimensions(BaseModel):
    w: Optional[float] = None
    h: Optional[float] = None
    d: Optional[float] = None
    method: str = ""


class Item(BaseModel):
    id: str
    category: str
    description: str
    brand_model: str = ""
    material: str = ""
    frame_ref: str = ""
    frame_refs: list[str] = Field(default_factory=list)
    frame_box: Optional[Box] = None
    dimensions_cm: Dimensions = Field(default_factory=Dimensions)
    status: Literal["priced", "range", "needs_appraisal", "unpriced"] = "unpriced"
    replacement_cost: ItemCost = Field(default_factory=ItemCost)
    confidence: float = 0.0
    excluded_reason: str = ""
    user_notes: list[str] = Field(default_factory=list)


class Room(BaseModel):
    length_m: Optional[float] = None
    width_m: Optional[float] = None
    height_m: Optional[float] = None
    floor_area_m2: Optional[float] = None
    floor_area_ft2: Optional[float] = None
    wall_area_m2: Optional[float] = None
    wall_area_ft2: Optional[float] = None
    shelved_wall_area_m2: Optional[float] = None
    shelved_wall_area_ft2: Optional[float] = None
    perimeter_m: Optional[float] = None
    shape: str = ""                # "rectangle" | "orthogonal_polygon" | ""
    shape_note: str = ""
    polygon_m: list[tuple[float, float]] = Field(default_factory=list)
    scale_method: str = ""
    confidence: float = 0.0
    frame_refs: list[str] = Field(default_factory=list)
    shelving_units: list[dict] = Field(default_factory=list)


class Totals(BaseModel):
    book_count: int = 0
    books_identified: int = 0
    books_unidentified: int = 0
    books_needs_appraisal: int = 0
    shelf_run_m: Optional[float] = None
    books_replacement_cost: float = 0.0
    books_used_value: float = 0.0
    books_priced_replacement: int = 0
    books_priced_used: int = 0
    items_replacement_cost_low: float = 0.0
    items_replacement_cost_high: float = 0.0
    excluded_from_totals: int = 0
    currency: str = ""
    computed_by: str = "library_claim.packet.totals.compute_totals"


class ReviewEntry(BaseModel):
    ref_id: str
    reason: str
    kind: str = ""                 # identification | pricing | measurement | appraisal | capture | room


class Sweep(BaseModel):
    id: str
    captured_at: str = ""
    device: str = ""
    duration_s: float = 0.0
    country: str = ""
    currency: str = ""


class ClaimPacket(BaseModel):
    sweep: Sweep
    room: Room = Field(default_factory=Room)
    books: list[Book] = Field(default_factory=list)
    items: list[Item] = Field(default_factory=list)
    totals: Totals = Field(default_factory=Totals)
    review_queue: list[ReviewEntry] = Field(default_factory=list)
    meta: dict = Field(default_factory=dict)  # cost, latency per stage, model ids, versions
