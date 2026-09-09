# Benchmark report

Regenerate everything in this file with one command:

```bash
bash scripts/run_benchmark.sh benchmark_runs
```

It runs all 11 captures across the three tiers, times each, and scores them,
writing `timing.csv`, `results.json` and `gate_table.txt`. `SOURCE_DATE_EPOCH` is
pinned inside the script, so a rerun is byte-comparable with `diff`.

Run under test: `benchmark_runs/`, scorer 1.0.0, **26 PASS / 23 FAIL / 45 SKIP**.

> **Read the SKIPs first.** 45 of 94 rows are SKIP, and SKIP is never a pass —
> it means no ground truth covered that gate. Only the two ray-traced fixtures
> carry ground truth. No real capture has laser or tape measurements, so most
> dimensional gates cannot be scored at all. That is the honest headline of this
> report and the main thing standing between it and the brief's 15% verified-
> accuracy row.

---

## 1. What ran

| Capture | Tier | Rooms | Walls | Openings | Footprint | Interval | Ceiling | Ceiling method | Adjacencies |
|---|---|---|---|---|---|---|---|---|---|
| `synthetic_room` | lidar | 1 | 4 | 2 | 10.08 m² | ±4.0% | 2.50 m | `measured_plane` | 0 |
| `synthetic_no_ceiling` | lidar | 1 | 4 | 2 | 10.08 m² | ±4.0% | 2.49 m | `wall_extrapolation` | 0 |
| `apartment_lidar` | lidar | 1 | 14 | 0 | 17.82 m² | ±4.0% | 2.44 m | `scan_cutoff_prior` | 0 |
| `scan_with_ceiling` | lidar | 1 | 12 | 0 | 39.01 m² | ±4.0% | **3.07 m** | **`measured_plane`** | 0 |
| `scan_floor_only` | lidar | 1 | 20 | 0 | 51.33 m² | ±4.0% | 2.44 m | `scan_cutoff_prior` | 0 |
| `demo_office` | photo | 1 | 4 | 1 | 17.17 m² | ±60.1% | 2.44 m | `per_view_merge` | 0 |
| `saurabh_room` | photo | 1 | 4 | 2 | 15.38 m² | ±60.1% | 2.44 m | `per_view_merge` | 0 |
| `saurabh_room_photo` | photo | 3 | 12 | 4 | 34.20 m² | ±60.2% | 2.44 m | `multi_room_stitch` | 2 |
| `demo_fourroom` | photo | 2 | 8 | 2 | 53.93 m² | ±60.2% | 2.44 m | `multi_room_stitch` | 0 |
| `apartment_video` | video | 2 | 8 | 2 | 28.18 m² | ±60.2% | 2.44 m | `multi_room_stitch` | 1 |
| `saurabh_room_video` | video | 3 | 12 | 2 | 24.06 m² | ±60.2% | 2.44 m | `multi_room_stitch` | 2 |

The interval column is the honest core of the tier design: **±4.0% at LiDAR
against ±60% at photo and video**, a 15× widening as the sensor data thins.

Ceiling height is only ever *measured* at the LiDAR tier, and only when the
operator actually looked up. `scan_with_ceiling` is the first capture in this
set where that happened (40.8% of points above camera height → 3.07 m). Every
photo and video plan reports 2.44 m because that is the structural prior the
scale cue assumes, not a measurement — see §5.

## 2. Gates, by tier

### LiDAR

| Gate | Result | Notes |
|---|---|---|
| `wall_lengths` | **2 PASS**, 3 SKIP | On exact ray-traced truth: **4/4 within tolerance, worst error 0.1 cm** against a `max(2 cm, 1%)` gate. |
| `ceiling_height` | **2 PASS**, 3 SKIP | Worst 0.4 cm / 0.5 cm against a 1.5 cm gate. |
| `opening_widths` | **2 PASS**, 3 SKIP | 2/2 matched, **0 missed, 0 phantom** — detection scored, not just dimension. |
| `footprint` | 2 PASS, 3 FAIL* | *False failures — see §6. |
| `interval_coverage` | 2 PASS, 3 FAIL* | *Same artifact. On the fixtures: 9/9 covered, mean ±12.7 cm. |
| `room_overlap` | 5 SKIP | Every LiDAR plan is single-room, so there is nothing to overlap. |
| `adjacency_correctness` | 5 SKIP | No adjacency ground truth exists. |
| `repeatability` | 1 PASS, 1 FAIL | See §3. |
| `ceiling_spread` | 1 PASS, 1 FAIL | See §3. |

The LiDAR tier is exact where it can be checked. On the synthetic fixtures it
recovers a 3.60 × 2.80 m room to **1 mm** and both openings to within the gate.
That validates the geometry and the Stray Scanner sensor conventions; it says
nothing about real-world noise, because ray-traced depth has none.

### Photo

| Gate | Result | Notes |
|---|---|---|
| `wall_lengths` | 4 SKIP | No photo capture has ground truth. |
| `ceiling_height` | 4 SKIP | " |
| `opening_widths` | 4 SKIP | Openings *are* detected (1–4 per capture, all from the semantic detector) but cannot be scored. |
| `footprint` | 4 FAIL* | *Artifact, §6. |
| `room_overlap` | **2 PASS**, 2 SKIP | Multi-room plans have **zero overlapping rooms** — the brief's automatic-failure condition is met, on real data, needing no ground truth. |
| `adjacency_correctness` | 4 SKIP | No `element=adjacency` rows. |
| `interval_coverage` | 2 PASS, 2 FAIL* | *Artifact, §6. |

The photo tier's stitched output is real: `saurabh_room_photo` produces **3 rooms
from 3 photo folders with 2 adjacencies and no overlaps**, which is the brief's
"photo-tier whole-property stitch" row. The **±8% footprint claim on that row is
unverified** — there is no tape measurement of that flat.

### Video

| Gate | Result | Notes |
|---|---|---|
| `room_overlap` | **2 PASS** | Both video plans multi-room, no overlaps. |
| `interval_coverage` | 1 PASS, 1 FAIL* | *Artifact, §6. |
| everything dimensional | SKIP | No ground truth. |

Both video captures now segment one walkthrough into **multiple rooms**
(`saurabh_room_video`: 3 rooms, 2 adjacencies from a single 37 s clip). Frames
are sampled at a fixed stride, filtered by Laplacian variance, capped per time
bucket, and split into rooms by DINOv2 appearance similarity between consecutive
frames.

## 3. Repeatability

The gate: two captures of the same room at the same tier agree within **1 cm or
0.5% per wall**. Spread of ceiling height across repeat captures ≤ **1 cm**.

| Group | Captures | Wall pairs | Worst | Gate | Result |
|---|---|---|---|---|---|
| `lidar:test_room` | 2 (synthetic) | 4 | **0.0 cm** | ≤1 cm | **PASS** |
| `lidar:room_a` | 3 | 38 | **718.4 cm** | ≤1 cm | **FAIL** |
| `photo:living_room` | 3 | 12 | 367.8 cm | ≤1 cm | FAIL† |
| `video:room_3` | 2 | 4 | 154.7 cm | ≤1 cm | FAIL† |
| `video:room_4` | 2 | 4 | 534.1 cm | ≤1 cm | FAIL† |

| Group | Captures | Ceiling spread | Gate | Result |
|---|---|---|---|---|
| `lidar:test_room` | 2 | **0.1 cm** | ≤1 cm | **PASS** |
| `lidar:room_a` | 3 | **62.9 cm** | ≤1 cm | **FAIL** |
| `photo:living_room` | 3 | 0.0 cm | ≤1 cm | PASS‡ |
| `video:room_3` / `room_4` | 2 each | 0.0 cm | ≤1 cm | PASS‡ |

† **These groups are not real repeat captures.** `_repeat_groups()` keys on
`(tier, room_id)` and ignores `space_id`, so unrelated captures that happen to
reuse a room *name* (`living_room`, `room_3`, `room_4`) are pooled and compared.
This is a scorer defect, not a pipeline result.

‡ These pass for the wrong reason: every photo/video plan reports the same
2.44 m structural prior, so the spread is trivially zero. A gate that passes
because nothing measured anything is not evidence of repeatability.

### The genuine pair

`scan_floor_only` and `scan_with_ceiling` are two LiDAR walkthroughs of the same
flat (inferred from matching trajectory spans, both closed loops, same kitchen
and bathroom on video — **not confirmed by the operator**). Isolated to just
those two:

```
repeatability   lidar:room_a   worst 718.4 cm on room_a_w0, 12 wall pairs   FAIL
ceiling_spread  lidar:room_a   spread 62.9 cm across 2 captures             FAIL
```

Per-wall ratio, scan A ÷ scan B:

| wall | A (m) | B (m) | ratio |
|---|---|---|---|
| w0 | 0.27 | 7.46 | 0.037 |
| w11 | 0.71 | 5.36 | 0.132 |
| w2 | 0.27 | 2.28 | 0.120 |
| w1 | 1.56 | 5.36 | 0.291 |
| w10 | 1.58 | 3.11 | 0.507 |
| w6 | 0.45 | 0.55 | 0.830 |
| w5 / w7 | 0.71 | 0.67 | 1.049 |
| w8 | 1.19 | 1.13 | 1.050 |
| w3 / w9 | 0.71 | 0.29 | 2.427 |
| w4 | 2.47 | 0.38 | 6.457 |

mean 1.365 · median 0.939 · **σ 1.720** · min 0.037 · max 6.457

**This is not a scale error** (a multiplicative error clusters ratios around one
constant; these span 175×, and σ exceeds the mean). **Not noise** (half the walls
differ by 2–7 metres). **Not a pose offset** (a rigid transform preserves
lengths; these change by 175×).

Two defects stacked:

1. **The gate compares walls that are not the same wall.** Wall ids are
   positional (`{room}_w{index}`, from polygon traversal order). The two scans
   produced polygons with **12 vs 20 walls** — different topology, so index *i*
   names a different physical wall in each. The 718 cm figure compares a 0.27 m
   wall against a 7.46 m one.
2. **A real instability underneath it.** Order-independent metrics cannot be
   explained away by pairing: footprint **39.01 vs 51.33 m², 27.3% apart**;
   bounding box 7.46 × 5.36 m vs 6.96 × 9.42 m. Both plan drawings show the
   capture trajectory *leaving the room polygon*. These are multi-room apartment
   walks, and the LiDAR tier emits exactly one room per capture, so the polygon
   covers an arbitrary part of the flat and which part depends on how the
   operator walked.

The ceiling failure has a separate, cleaner cause: `scan_with_ceiling` measured
3.07 m (`measured_plane`, 40.8% of points above camera height);
`scan_floor_only` never looked up and fell back to the 2.44 m prior
(`scan_cutoff_prior`). **This is repeatable-but-biased, not unrepeatable** —
the brief asks which one, and it is the former.

## 4. Interval coverage (calibration)

Gate: ≥90% of 95% intervals contain the truth.

| Tier | Quantity | Covered | Mean half-width | Result |
|---|---|---|---|---|
| lidar | `wall_length` | **8/8 (100%)** | ±5.2 cm | PASS |
| lidar | `opening_width` | **4/4 (100%)** | ±5.0 cm | PASS |
| lidar | `ceiling_height` | **2/2 (100%)** | ±11.5 cm | PASS |
| lidar | `floor_area` | **2/2 (100%)** | ±0.40 m² | PASS |
| lidar | `footprint_area` | 2/5 (40%) | ±1.03 m² | FAIL* |
| photo | `footprint_area` | 2/4 (50%) | ±18.15 m² | FAIL* |
| video | `footprint_area` | 1/2 (50%) | ±15.73 m² | FAIL* |

\* Every `footprint_area` failure is the §6 artifact — real footprints scored
against the synthetic fixture's 10.08 m². The four non-footprint LiDAR rows are
scored against exact truth and all pass at 100%.

Photo and video half-widths of ±15–18 m² are not calibration; they are the
pipeline declining to claim precision it has not earned on thin input. That is
the intended behaviour ("confident garbage on thin input caps your score") but it
is *asserted*, not fitted — `cozmo calibrate` exists and no calibration file is
committed.

## 5. Timing

Wall-clock per capture, MacBook (Apple Silicon, MPS), from `benchmark_runs/timing.csv`:

| Capture | Tier | Frames / photos | Seconds |
|---|---|---|---|
| `synthetic_room` | lidar | 16 depth | **2.8** |
| `synthetic_no_ceiling` | lidar | 16 depth | **2.5** |
| `apartment_lidar` | lidar | 1,715 depth | **6.8** |
| `scan_floor_only` | lidar | 5,251 depth | **21.5** |
| `scan_with_ceiling` | lidar | 9,745 depth | **68.8** |
| `demo_office` | photo | 7 stills | 43.1 |
| `saurabh_room` | photo | 8 stills | 48.4 |
| `saurabh_room_photo` | photo | 3 folders (10 stills) | 90.4 |
| `demo_fourroom` | photo | 4 folders (21 stills) | 140.5 |
| `apartment_video` | video | 1,715-frame clip | 107.2 |
| `saurabh_room_video` | video | 1,105-frame clip | **233.4** |

**Whole benchmark: 12.8 minutes for 11 captures.** The LiDAR tier is
geometry-only and scales with frame count (2.5 s → 68.8 s over a 600× range).
Photo and video are dominated by VGGT: roughly 40–50 s per room regardless of
capture size, which is why multi-room captures scale linearly with room count.

All three tiers are comfortably inside the walk-in test's practical window.

## 6. Known scorer artifacts

Two defects in `cozmo/benchmark/score.py` inflate the FAIL count in this run and
must not be read as pipeline failures:

1. **One global footprint row applied to every capture.** `gate_footprint` does
   `gt.lookup("property", "footprint_area")` and finds the synthetic fixture's
   single 10.08 m² row, then scores *every* capture against it — hence
   "51.33 m² vs 10.08 m² (409.2%)". Accounts for **9 of the 23 FAILs**, plus the
   per-capture `interval_coverage` and `*_by_kind footprint_area` failures.
2. **Repeatability groups by room name, ignoring space.** Detailed in §3.

Neither is hidden by the runner; both are visible in `results.json`. Correcting
them is a scorer change and would alter the FAIL count without touching the
pipeline, which is precisely why they are called out here rather than quietly
fixed before the number was reported.

## 7. Head-to-head vs an incumbent app

**Not run.** No consumer scanning app was installed, no export obtained, no
comparison made. This is Part 3 of the brief and 10% of the score, and it is
currently zero.

| Room | Dimension | Ours (error vs laser) | poly.cam / magicplan (error vs laser) | Winner |
|---|---|---|---|---|
| — | — | not run | not run | — |

Two blockers, in order: there is no laser ground truth for any real room (§ the
SKIP note at the top), and no app export. The first blocks both columns; the
second blocks one.

## 8. Reproduction

| What | Command | Measured |
|---|---|---|
| Clean-machine setup | `scripts/setup.sh` | **58.1 s** from `git clone` to a working reconstruction, cold pip cache, no weights needed |
| Whole benchmark | `bash scripts/run_benchmark.sh benchmark_runs` | ~11 min, 11 captures |
| One capture | `cozmo run --input <dir> --out <dir>` | 2.5 s – 165 s by tier |
| Determinism | run twice, `diff plan.json` | LiDAR: `plan.json` **and** `plan.png` byte-identical. Photo (real VGGT/MPS): identical, max wall difference **0.00e+00 m** |
| Weight integrity | `scripts/fetch_weights.sh --check` | 26 files, **7/7 model weights sha256-pinned**, "all weights present and verified" |

Provenance is stamped into every `run_manifest.json`: git commit, exact command,
seed record, and a sha256 over the input directory, so any number here can be
tied to the bytes that produced it.
