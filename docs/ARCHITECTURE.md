# Architecture (one page)

```mermaid
flowchart LR
  subgraph Phone browser
    MIC[mic 16 kHz PCM] --> WS
    CAM[camera ~3 fps JPEG 1600px + accelerometer] --> WS
    WS((websocket)) --> SPK[speaker 24 kHz]
    WS --> UI[live inventory + capture chips]
  end
  WS <--> SRV[FastAPI server]
  SRV <-- audio + 1 fps video + APP NOTICES / tool calls --> LIVE[Gemini Live\nvoice agent]
  SRV --> FS[FrameStore\nsave every frame, QC]
  FS -->|keyframes| TR[Tracker\nORB homographies]
  FS -->|keyframes| MK[ArUco detect]
  FS -->|keyframes| RB[RoomBuilder\nDepth-Anything + RGB-D odometry]
  TR --> VR[Vision reader\nGemini Flash]
  VR --> TR
  TR --> ID[Identify\nOpen Library / Google Books]
  ID --> PB[Price books\nGemini + Search -> verify on page]
  FS -. QC, unreadable, marker, rows .-> DIR[Capture director] -. NUDGE / PROGRESS / ASK .-> LIVE
  subgraph after end_sweep
    TR --> MS[Measure spines\nmarker plane / catalogue dims]
    MK --> MS
    MS --> PK
    PB --> PK[Packet builder\ntotals in code]
    RB --> RF[Room fit] --> PK
    TR --> IC[Item consolidation\nGemini, ids only] --> PI[Price items] --> PK
    PK --> OUT[claim_packet.json\nreport.html\nzip + frames]
  end
```

## Which model does what

| Job | Model | Why this one |
|---|---|---|
| Talk, listen, see roughly (1 fps) | Gemini Live | low-latency voice with tool calls; it directs capture but its own impressions never become packet data |
| Read spines, find objects and bookcases | Gemini Flash (vision, JSON) | one call per keyframe returns boxes for 30-60 thin objects plus verbatim text; transcription only |
| Same object seen in two tracking segments? | Gemini Flash | returns groupings of ids we supply; cannot add facts |
| Find prices | Gemini Flash + Google Search grounding | finds local retailer pages; every number is then checked against the page or the grounding metadata |
| Room depth | Depth-Anything-V2 Metric Indoor Small (25M) | metric indoor depth, runs on a laptop CPU at ~0.5 s/keyframe |
| Everything else | code | geometry, matching, totals, review queue |

## Where metric scale comes from

* **Spines and bookcases:** a printed ArUco marker (15 cm) leaning on the spines. Corners -> exact homography from pixels to cm on the spine plane; homography chaining carries it to all keyframes in the same tracking segment. Fallbacks: catalogue heights of identified books in the same frame (Open Library `physical_dimensions`), a dimension the policyholder states, else empty + review.
* **Room:** monocular metric depth rescaled by the marker (true distance = fx * side / side_px), RGB-D odometry between keyframes, accelerometer gravity for "up" and per-frame tilt correction. Floor and ceiling = extreme dense horizontal bands; walls = outermost point-density peaks; rectangle unless the traced outline fills < 90 % of it (then polygon, with the note in the packet). Wall area is gross perimeter x height. Camera horizontal FOV is calibrated once with `cli calibrate-fov`.

## Where prices come from

Search agent (Gemini + Google Search) proposes offers for the policyholder's market (retailer lists per country in `config.py`). `pricing/verify.py` fetches each offer page and requires the amount next to the currency symbol (or a schema.org price) and the product's words on the page -> `page_text`. If the page blocks us but grounding metadata attributes the sentence with that amount to a retrieved result -> `grounded_snippet` (accepted, flagged for review). Otherwise discarded. Replacement = median of verified NEW offers; used = median of verified USED offers ("Good" condition assumed). No local offers -> other market converted at ECB rate (Frankfurter), labelled `converted`. Rarity cues, offers over the threshold and all art -> `needs_appraisal`, never priced.

## Separate, testable stages

Each stage writes JSON under `sweeps/<id>/stages/` (vision per keyframe, item groups, corrections), and `cli process <sweep> --reuse-vision` re-runs tracking, measurement, identification, pricing, room and packet from saved frames without re-paying for vision.
