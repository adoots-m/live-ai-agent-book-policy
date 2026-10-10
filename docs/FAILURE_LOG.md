# Failure log

> Fill this in from the real sweep. Every number here must come from `score.json`, the packet's `meta.cost`, or a re-run you did. Nothing below the line is a result yet.

Sweep: `<id>` · date · room · phone · duration

## Cost per sweep and latency per stage
From `claim_packet.json -> meta.cost` and `meta.time_to_packet_s`.

| Stage | Calls | Model p50 ms | Wall ms after sweep | USD |
|---|---|---|---|---|
| vision_read | | | | |
| pricing_books_new / used | | | | |
| pricing_items | | | | |
| consolidate_items | | | | |
| measure_spines / room_fit / build_packet | – | – | | – |
| live session | – | – | – | |
| **Total** | | | **time to packet: s** | **$** |

## The three worst errors

For each: what went wrong (with the frame_ref / crop), how much it cost against a pass bar, root cause, what you tried, measured effect, what you would change.

### 1. <title>
* Symptom:
* Evidence: `frames/f.....jpg`, line ids
* Root cause:
* Fix tried: … → re-run `cli process … --reuse-vision` → before **x** / after **y** on <bar>
* Next change:

### 2. <title>

### 3. <title>

## Pre-registered risks (written before the real run, to check against)
These are the failure modes the design expects; confirm or refute each with data.
1. **Spine thickness on thin, tightly packed spines** – reader boxes a few px loose; edge snapping (`measure/refine.py`) helps only where adjacent spines differ in colour. Lever: `LCA_VISION_TILES=2`, closer capture, edge refine on/off.
2. **Missed or merged spines in dense frames** – a vision model returning 50+ thin boxes drops some. Shows up as book count below truth. Lever: tiling, slower sweep, director's unreadable nudge.
3. **Double counting across tracking breaks** – fast turns break the homography chain; unreadable spines either side cannot be merged. Shows up as count above truth; each break is in the review queue.
4. **Room drift** – odometry drift on long walks; bookcases to the ceiling hide walls. Shows up as floor/wall area error. Lever: walk in a loop, include the marker in a wide shot, pan the ceiling.
5. **Retailer bot walls** – page verification fails, prices fall back to `grounded_snippet` or are dropped; shows up as fewer priced books, not wrong prices.

## With two more weeks
* (fill in)
