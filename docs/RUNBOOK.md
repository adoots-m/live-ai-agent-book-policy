# Runbook: from a working install to the six deliverables

The code is done; these steps need you, a room and a tape measure. Budget ~6-8 hours.

## 0. Run the reference app first (30 min)
The brief requires it. In a separate folder:
```bash
git clone --depth 1 https://github.com/Shubhamsaboo/awesome-llm-apps && cd awesome-llm-apps/voice_ai_agents/insurance_claim_live_agent_team
python -m venv .venv && .venv\Scripts\activate   # or source .venv/bin/activate
pip install -r requirements.txt && copy .env.example .env   # add GOOGLE_API_KEY
python -m uvicorn live_demo.server:app --host 127.0.0.1 --port 4177
```
Try the "Maya Singh" example and the camera. Write 3-5 lines of what you saw in the README's TODO.

## 1. Pick and prepare the room (30 min)
* One enclosed room, >= 2 shelving units, >= 60 books, >= 8 non-book items (include at least one framed picture, a lamp, a rug, an electronic device).
* Print `markers/marker_0.png` at 100 % scale. Measure the black square with a ruler; put it in `.env` as `LCA_MARKER_SIDE_CM`. Tape it to a piece of card so it stays flat.
* Calibrate the camera FOV once: hold the phone 1.00 m (tape) from the marker, flat-on, take a photo through the app's camera aspect (or the phone camera at the same aspect), then `python -m library_claim.cli calibrate-fov photo.jpg --distance-cm 100` and set `LCA_CAMERA_HFOV_DEG`.

## 2. Build ground truth BEFORE tuning anything (2-3 h)
`python -m library_claim.cli gt-template ground_truth.xlsx`, then fill:
* **summary.true_book_count**: count every book, including unreadable and stacked ones.
* **books**: one row per book, shelf + position (left to right, shelves top to bottom), `legible_by_human` = Y if *you* can read the title from where the camera will be, title, author.
* **spine_sample**: 20 books, tape height and thickness in cm. Pick across shelves, thin and thick, not just easy ones.
* **price_sample**: 15 books, look up a price yourself on the same kind of source the system uses (new: a national retailer; used: AbeBooks/ThriftBooks/eBay for US), record retailer, URL, date, price.
* **items**: every non-book item, category (shelving, furniture, lamp, art, rug, electronics, appliance, decor, textile, plant, other) and a short description.
* **room**: tape length, width, floor-to-ceiling height. If not a rectangle, compute floor/wall area by hand in the override rows.
Commit nothing from this sheet to code. The system must not see it.

## 3. Rehearse (30 min)
`python -m library_claim.live`, open the URL on the phone. Do one practice sweep. Check `sweeps/<id>/report.html`. If spines are mostly unreadable, the phone is too far or moving too fast; if the marker chip never turns green, the marker is too small in frame.
You may tune thresholds (`capture/quality.py`) on the practice sweep; keep a note of what you changed and why (it belongs in the failure log).

## 4. The real sweep + demo video (one take)
* Start the phone's screen recorder **with microphone on** (so the agent's voice and yours are audible), open the app, Connect, and do the whole thing without stopping: greeting -> sweep (< 3 min) -> End -> summary read-back -> open the report. Must be <= 6 minutes. Do not edit, speed up or stitch.
* Say at least one correction on camera ("skip that shelf, those aren't mine", or "that one is signed") and answer the art question.

## 5. Score and second country (30 min)
```bash
python -m library_claim.cli score sweeps/<id> ground_truth.xlsx      # writes ground_truth_scored.xlsx + score.json
python -m library_claim.cli reprice sweeps/<id> --country GB --limit 10
```
Report every bar as measured, including fails. Do not re-score on a subset.

## 6. Failure log + measured fix (1-2 h)
Open `docs/FAILURE_LOG.md`, pick the three worst errors from `score.json` / the review queue, find the root cause (look at the frame crop), try one fix, and re-run with `cli process sweeps/<id> --reuse-vision` (or without `--reuse-vision` if the fix is in the reader) and re-score. Report before/after numbers.
Candidate levers already in the code: `LCA_VISION_TILES=2` (more spines read per frame), `LCA_EDGE_REFINE=0/1` (thickness), `LCA_THINKING_LEVEL`, quality thresholds, `ACCEPT` in `identify/catalogue.py`.

## 7. Deliver
Private GitHub repo (no `.env`, no `sweeps/`), demo video, `sweeps/<id>/claim_packet_<id>.zip` (packet + report + frames), `ground_truth_scored.xlsx`, `docs/ARCHITECTURE.md`, `docs/FAILURE_LOG.md`.
