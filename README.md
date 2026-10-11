# Library Contents Claim Agent

A live voice-and-vision insurance agent that inventories and values a home library in **one continuous camera sweep**. The policyholder walks the room once with a phone, talking to the agent. The agent tells them when to slow down, step closer, avoid glare, show the scale marker or re-cover a skipped shelf. The inventory fills in on screen as it goes. Within a few minutes of the sweep ending it produces a claim packet:

* every book, counted from its spine (upright, lying flat or stacked), read, matched to a catalogue record where the evidence supports it, measured in cm, and priced twice (new replacement, used market) in the policyholder's currency, with source URL, retrieval date and assumed condition;
* every other item (shelving, furniture, lamps, art, rugs, electronics, decor), classified, sized and priced (or given a sourced range);
* the room's floor area, gross wall area and shelved wall area, from the same sweep;
* a review queue listing every line the system is not confident in, and why.

Totals are computed in code from the lines (`library_claim/packet/totals.py`). Every number traces back to a saved frame (`frame_ref`) and, for prices, to a URL and date.

> **Status (be honest):** everything below is implemented and covered by 33 automated tests on synthetic data with mocked models (`pytest`). It has **not** yet been run end-to-end against the live Gemini APIs on a real room — that run, the ground truth, the demo video and the failure log come from the real sweep (see [docs/RUNBOOK.md](docs/RUNBOOK.md)).

---

## Quick start (about 10 minutes)

Needs Python 3.11+ and a Google AI Studio API key. A phone and the computer must be on the same Wi-Fi.

```bash
git clone <this repo> && cd library-claim-agent
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # room-measurement stage (CPU is fine)
pip install -r requirements-depth.txt
cp .env.example .env            # Windows: copy .env.example .env   -> put your GOOGLE_API_KEY in it
python -m library_claim.cli marker          # writes markers/marker_0.png; print it at 100 % scale
python -m pytest                            # 33 tests, ~20 s, no API key needed
python -m library_claim.cli doctor --live --depth   # checks key, model names, search grounding, Live, free APIs, depth model
python -m library_claim.cli try-photo shelf.jpg     # one photo of a shelf (with the marker): read, identify, measure
python -m library_claim.live                # prints https://<your-ip>:4177/?k=<key>
```

Open the printed URL on the phone, accept the self-signed certificate warning once, tap **Connect**, and talk. (If the certificate is a problem, run `python -m library_claim.live --http` and expose it with `cloudflared tunnel --url http://localhost:4177`.)

No API key? `python scripts/demo_synthetic.py` runs the whole pipeline on a synthetic bookcase with stand-in models and writes a sample `report.html` so you can see the packet format. Its prices are labelled SYNTHETIC.

## How a sweep works

1. The agent greets you, confirms country and currency (`confirm_locale`), and explains the sweep in two sentences.
2. Lean the printed marker against the books on each bookcase. Say "ready"; the agent calls `start_sweep`.
3. Walk the shelves slowly, top to bottom, then pan the walls, floor and ceiling. Target < 3 minutes.
4. The agent speaks when the app sees a problem (blur, fast motion, glare, darkness, spines too small, marker not seen on this bookcase, shelf rows skipped), says what it is logging, and asks what video cannot answer ("is the portrait an original or a print?").
5. Correct it any time: "that one's signed", "that's a first edition", "skip that shelf, those aren't mine", "the bookcase is 180 centimetres tall". Corrections are mapped to what was in view when you spoke.
6. Say "done" (or tap End). Background stages finish; the agent reads back a summary; the report, `claim_packet.json` and a zip with every referenced frame are linked on screen and saved in `sweeps/<id>/`.

The agent never asks you to pull books out, scan barcodes, type ISBNs or photograph items one by one.

## Pipeline

Each stage is a separate module with its own output under `sweeps/<id>/stages/` and its own tests.

| Stage | Module | What it does | Model / source |
|---|---|---|---|
| Capture QC | `capture/quality.py`, `capture/frames.py` | Saves every frame; blur (Laplacian variance), glare, darkness, motion (ORB shift); keyframe selection by new area | OpenCV |
| Live agent | `live/server.py`, `live/agent.py` | Voice in/out, camera at 1 fps to the agent, tools, capture director that turns pipeline signals into nudges | Gemini Live |
| Spine & object reader | `vision/reader.py` | Per keyframe: a box for every spine (readable or not), verbatim text, orientation, rarity cues; objects; bookcases. A title not found in the transcription is discarded | Gemini Flash (vision) |
| De-duplication | `vision/tracker.py` | Chains homographies between keyframes (ORB + RANSAC) so the same spine in overlapping views is one book; new segment when tracking is lost | OpenCV |
| Scale & measurement | `measure/scale.py`, `measure/spines.py`, `measure/refine.py` | ArUco marker -> pixel-to-cm homography on the spine plane; edge-snapped spine boxes; height, thickness, shelf rows, shelf run, bookcase faces | OpenCV |
| Identification | `identify/catalogue.py` | Spine text -> Open Library / Google Books candidates; accepted only if the candidate's title is found in the spine text and author/long title corroborate; edition only if the spine's publisher matches and catalogue height agrees with the measured height | Open Library, Google Books (keyless) |
| Book pricing | `pricing/books.py`, `pricing/verify.py`, `pricing/fx.py` | Search agent proposes offers; an offer counts only if we fetch the page and find the amount next to the currency (or the amount is attributed to a retrieved result by grounding metadata). Median of verified offers. No local price -> other market + ECB conversion, labelled. Over threshold or rarity cue -> appraisal | Gemini + Google Search grounding, Frankfurter (ECB) |
| Item pricing | `pricing/items.py` | Brand/model -> priced; otherwise interquartile range of verified comparable offers; art -> appraisal unless you say it is a print | same |
| Room | `measure/room.py` | Monocular metric depth per keyframe, rescaled by the marker; RGB-D odometry; accelerometer gravity; floor/ceiling as dense horizontal bands; walls at outermost density peaks; rectangle or traced polygon | Depth-Anything-V2 Metric Indoor (small), OpenCV |
| Packet | `packet/build.py`, `packet/totals.py`, `packet/report.py` | Lines, review queue, totals in code with a self-check, HTML report with evidence crops, zip with frames | none |

Details and the diagram: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

### Where metric scale comes from

1. **Printed ArUco marker** (primary). Its four corners give an exact homography from image pixels to centimetres on the plane of the spines, so perspective is removed as well as scale. Homography chaining carries that plane to every keyframe of the same tracking segment.
2. **Catalogue dimensions** (fallback, no marker in that segment). Identified books whose catalogue record lists a height give px/cm for other spines in the same frame (leave-one-out median). Those books' own heights are then catalogue values, and the line says so.
3. **Stated dimension** ("the bookcase is 180 cm tall").
4. Otherwise the dimension is left empty and the line goes to review. "The model estimated it" is never used.

The room stage uses the marker too: wherever it is visible, its known size gives the true distance (pinhole model), which rescales the monocular depth maps.

### Pricing integrity

* No model writes a price from memory. A proposed price without a page or retrieved search result that carries the same amount is dropped and logged.
* Every price has `source`, `url`, `retrieved_at`, `verification` (`page_text` or `grounded_snippet`) and `evidence` (the text that carried the number). Used prices state `condition_assumed`.
* Unknown stays empty (`null`) and the line is excluded from totals, with the reason in the review queue.
* Rare / signed / antiquarian cues, anything above the appraisal threshold, and all art go to `needs_appraisal` and are not auto-priced.
* Locale is a setting: `python -m library_claim.cli reprice sweeps/<id> --country GB --limit 10` prices the same 10 books for a second country and writes `locale_comparison_GB.html`.

## Other commands

```bash
python -m library_claim.cli process sweeps/<id> --reuse-vision   # re-run downstream stages on saved frames
python -m library_claim.cli process my_walk.mp4                  # whole pipeline on a recorded video (no live agent)
python -m library_claim.cli calibrate-fov marker_at_1m.jpg --distance-cm 100
python -m library_claim.cli gt-template ground_truth.xlsx        # ground-truth workbook to fill in by hand
python -m library_claim.cli score sweeps/<id> ground_truth.xlsx  # pass bars -> results sheet + score.json
```

## Reference app: kept, changed, threw away

Reference: `awesome-llm-apps/voice_ai_agents/insurance_claim_live_agent_team`. I read all of it and ran its test suite (55 Python tests and 18 client regression tests, all passing). **The live run with microphone and camera still has to be done with an API key — record what you see here:** _TODO(Adi): ran reference live on <date>; observations: …_

* **Kept (the pattern, re-implemented):** browser <-> FastAPI websocket <-> Gemini Live; 16 kHz PCM up, 24 kHz PCM down, interruption clears playback; JPEG frames as realtime video; NON_BLOCKING tools so the agent keeps talking while the team works; app-state notices sent as client content so the model can tell app state from the claimant's words.
* **Changed:** the live model no longer *is* the evidence. In the reference it describes what it sees and pins a frame. Here, high-resolution frames go to a separate pipeline that counts, reads, measures and prices; the live model only directs capture and relays results. The agent is driven by measured signals (blur, glare, motion, unreadable fraction, marker presence, missed rows) rather than its own impression. The packet is structured JSON with evidence links, not a narrative notebook.
* **Threw away:** policy directory and lookup, FNOL intake rules and ADK claim graph, incident sketches, avatar video, the claim notebook UI.

## Tools used

* Models: Gemini Live (voice agent), Gemini Flash (spine/object reader, item de-duplication, pricing search with Google Search grounding), Depth-Anything-V2 Metric Indoor Small (room depth, via Hugging Face `transformers`).
* Libraries: OpenCV (ArUco, ORB, homographies, quality), NumPy, FastAPI/uvicorn, httpx, pydantic, openpyxl.
* Data: Open Library and Google Books APIs (identification, catalogue dimensions), Frankfurter / ECB rates (conversion), retailer pages found by search (prices).
* AI coding tool: Claude (Anthropic), in Cowork, wrote most of this code and the tests.

## Cost and latency

Every packet's `meta.cost` holds the measured numbers for that sweep: calls, tokens, cost and model latency per stage, wall time per post-sweep stage, Live session tokens by modality (or a per-minute estimate if the session did not report them), and `time_to_packet_s`. Each finished sweep also appends a row to `sweeps/usage_log.csv`; `python -m library_claim.cli usage` sums them so you can reconcile with AI Studio > Dashboard > Usage.

Rates default to the Gemini API paid tier as published on 2026-10-11 ([pricing](https://ai.google.dev/gemini-api/docs/pricing)): `gemini-3.8-flash` $0.75 in / $3.75 out per 1M tokens (doubling on 2027-01-01), Search grounding $14 per 1,000 after 5,000 free a month, `gemini-3.8-live` about $0.025 per minute. Override them in `.env`.

Guards: `LCA_MAX_USD_PER_SWEEP` (default $8) stops pricing searches once a sweep's estimated spend passes it, and `LCA_MAX_VISION_CALLS` (default 250) caps vision reads; anything skipped goes to the review queue, never into an invented number.

**Estimate before the first real run** (replace with the measured figure): a 3-minute sweep of ~60 books is ~80 vision calls (~$1.00), ~150 grounded searches (~$0.80 in tokens, $0 while inside the free 5,000/month, +$2.10 beyond it), ~15 item searches (~$0.10), live session (~$0.15): about **$2 per sweep, $4 once the free searches are used up**.

## Known limitations (before real-world testing)

* Spine thickness is the hardest number: a 2 cm spine is ~15 px at shelf distance, so a 2 px box error is 13 %. Edge snapping helps on clean edges; tightly packed same-colour spines remain hard.
* Unreadable spines on either side of a tracking break can be double-counted; each break is listed in the review queue.
* Bookcases that reach the ceiling hide the wall behind them; the room is then measured to the shelf faces on that side.
* Room accuracy depends on monocular depth plus odometry drift; the marker fixes scale, not drift. Accelerometer gravity removes tilt drift, not heading drift.
* Many retailers block page fetches; those prices fall back to `grounded_snippet` verification and are flagged.
* Tested against synthetic scenes and mocked APIs only, so far.

## Tests

```bash
python -m pytest          # 33 tests: tracking, marker scale, measurement, room fit (synthetic clouds and a ray-traced room),
                          # pricing verification, identification refusal, totals, scorer, websocket flow, full synthetic sweep
```
