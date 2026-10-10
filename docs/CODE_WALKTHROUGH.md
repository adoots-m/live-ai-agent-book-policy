# Code walkthrough (for the review call)

Read in this order. File sizes are small on purpose; each stage is one file.

## 1. Data contracts — `library_claim/schemas.py`
Observation models (`FrameRecord`, `SpineObs`, `ItemObs`, `FrameVision`) are what stages emit. Packet models mirror the brief's `claim_packet.json` field-for-field; extra fields are additive (`frame_refs`, `frame_box`, `scale_method`, `evidence`, `verification`, `converted_from`). Unknown = `None` -> `null`.

## 2. The orchestrator — `library_claim/sweep.py`
`SweepSession` owns one sweep. `add_frame` saves and QC's a frame (thread pool) and queues keyframes. `_keyframe_worker` runs **in order**: homography (tracker), marker detection, room queue, then spawns the vision read (concurrent, semaphore). `_vision` caches the reader output per frame, feeds the tracker, starts identification for any track whose best text changed, and pushes the live inventory. `_identify` -> `_schedule_price` starts pricing as soon as a book is identified, so most pricing is done when the sweep ends.
`finish()` is the post-sweep pipeline; each block is timed with `ledger.stage(...)`. Order matters: corrections are resolved before measurement (excluded shelves), measurement before final identification (measured height picks the edition), pricing gathered, room fit, items consolidated and priced, packet built.
`_resolve_corrections` maps "that shelf" to track ids using the keyframe nearest the moment the user spoke, and the spine nearest the frame centre.

## 3. Capture — `capture/quality.py`, `capture/frames.py`
`assess`: Laplacian variance (blur), share of pixels >= 250 (glare), <= 25 (dark), median ORB displacement vs previous frame (fast). `FrameStore._is_keyframe`: sharp, and either >= 35 % new area vs the last keyframe (homography overlap) or 2.5 s elapsed.

## 4. Reader — `vision/reader.py`
One prompt; Gemini box format `[ymin,xmin,ymax,xmax]` 0-1000 converted to pixels in `_to_px`. `parse_reader_output` is the guard: a `title` that is not a substring of `text` is dropped, `legible` requires a title, author/publisher must also appear in `text`. Optional tiling (`LCA_VISION_TILES`) with IoU merge.

## 5. Tracker — `vision/tracker.py`
`add_keyframe`: ORB (2500 features at 1280 px), ratio test, RANSAC homography to the previous one or two keyframes, `_sane` rejects degenerate H. `H_to_ref` maps this keyframe into the segment's first keyframe. `add_vision`: each spine box projected to mosaic coordinates; same book if it overlaps an existing track >= 50 % on both axes and the texts (if both legible) are not contradictory. `merge_across_segments`: legible tracks with text similarity > 0.88 merge across segments; unreadable ones cannot.

## 6. Measurement — `measure/scale.py`, `measure/spines.py`, `measure/refine.py`
`plane_from_marker`: 4 corners -> `getPerspectiveTransform` to a square of side `LCA_MARKER_SIDE_CM`. `segment_planes`: P_ref = M_marker · inv(H_marker_frame_to_ref). For a spine seen in keyframe k: P = P_ref · H_k. `quad_dims_cm` averages opposite edges. Up to 4 non-clipped views per book, median. `_touches_border` -> height left empty if every view is clipped. Fallback: `catalogue_dims` leave-one-out px/cm. `assign_rows` clusters spine bottoms; `shelf_run_m` = upright thicknesses + one footprint per horizontal stack. `measure_units`: each bookcase edge = median over views where that edge is not clipped. `refine_box` snaps edges to the strongest gradient within 15 % of the width.

## 7. Identification — `identify/catalogue.py`
`coverage(phrase, text)`: share of the candidate title's meaningful tokens present in the spine text (fuzzy 0.84 for OCR slips). `score` = 0.6 coverage + 0.3 author (or long exact title) + 0.1 publisher, minus a penalty for titles much longer than the spine text. Accept only if score >= 0.75 and coverage >= 0.8 and no rival work within 0.05. `pick_edition` needs the spine publisher; then catalogue height within 1.5 cm of the measured height; otherwise "format not determinable", no ISBN.

## 8. Pricing — `pricing/verify.py`, `pricing/books.py`, `pricing/items.py`, `pricing/fx.py`
`amount_patterns` builds regexes for "$12.99", "12,99 €", "£1,299.00"; a bare number does not count. `verify_offer` tries the offer URL, then the grounding chunks attributed to the sentence containing the amount; page must also mention the product (coverage >= 0.6). `price_book`: rarity flags -> stop; else NEW and USED figures, each median of verified offers, else fallback market + FX; any verified offer above the threshold -> appraisal, no amounts.

## 9. Room — `measure/room.py`
`RoomBuilder.add`: depth at 518 px, marker scale samples, ORB, odometry (`_odometry`: bilinear depth, reject depth edges, RANSAC-Kabsch). `fit`: rescale, per-frame tilt correction from the accelerometer (`_rot_between`), fuse, `fit_room`. `fit_plane_height`: lowest/highest 2 cm bin with >= 2x the density of the bins inward of it. `wall_position`: outermost density peak. Polygon only when walls were seen >= 85 % of the way round and the outline is 60-90 % of the rectangle.

## 10. Packet — `packet/build.py`, `packet/totals.py`, `packet/report.py`
`build_packet` decides status per line and writes the review queue reasons; no model call. `compute_totals` is the only place totals are summed; `check_totals` recomputes from the lines and the report prints the self-check.

## 11. Live — `live/server.py`, `live/agent.py`, `live/static/app.js`
Three tasks per connection: browser->Gemini (audio, 1 fps downscaled video, frames to the pipeline when the sweep is active, with drop-on-backlog), Gemini->browser (audio, transcripts, tool calls run as tasks), and `notices` (1 Hz: pushes status to the UI and asks the `Director` whether to speak). `Director.next_notice`: at most one notice every 5 s, per-kind cooldowns, priority fast > blur > dark > glare > too far > unreadable > marker > missed rows > progress.

## Questions you may get
* *Why not let Gemini Live do the inventory?* It sees 1 fps at low resolution and its descriptions are not measurements; separating capture direction from evidence is the point of the exercise.
* *Why homographies, not SLAM?* A bookcase face is a plane; a homography is exact for it, cheap, and testable. SLAM is used only where the scene is not planar (room), and that is the less accurate stage.
* *How do you stop invented titles?* The reader can only transcribe (substring guard); the catalogue match must be found in the transcription; unreadable -> `unidentified` with dimensions.
* *How do you stop invented prices?* The number must be on a fetched page next to a currency symbol, or attributed by grounding metadata to a retrieved result. Otherwise it is discarded and the drop is logged.
