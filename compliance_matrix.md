# Compliance matrix

Requirement → file path → artifact → status, one row per requirement in the
brief (Parts 1–5, Deliverables, the walk-in test, Constraints).

Statuses are **met** / **partial** / **not met**, assigned against what is in the
repository today, not against intent. Where a row is partial or not met the gap
is named rather than softened.

Generated against commit `HEAD`, benchmark run `benchmark_runs/` (11 captures,
`scorer_version` 1.0.0, 26 PASS / 23 FAIL / 45 SKIP — see
[benchmark_report.md](benchmark_report.md) for what the FAILs and SKIPs are).

---

## Part 1 — Capture route and input tiers

| # | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| 1.1 | Choose a capture route: own iOS app **or** stock capture protocol | — | — | **not met** — neither a TestFlight/dev build nor a written one-page protocol exists. The pipeline ingests Stray Scanner output, but nothing tells a non-engineer what to install or how to walk. This is the 5% "capture route quality" row and it is currently unearned. |
| 1.2 | Route 2: name the off-the-shelf tool | `cozmo/io/stray.py` | Loader targets **Stray Scanner** format (depth uint16 mm @ 256×192, camera-frame z, quaternion poses) | **partial** — the tool is implied by the loader and documented in `README.md`, but not named in a protocol page with install/walk/hand-off instructions. |
| 1.3 | Tier 1 — **Photos**, 2–8 stills per room, no depth/poses, one folder per room | `cozmo/pipeline/photo.py`, `cozmo/recon/` | `cozmo run` on `captures/saurabh_room` → `benchmark_runs/saurabh_room/plan.json` | **partial** — runs end to end and accuracy is honestly reported (±60% intervals), but the photo captures it was exercised on are an Android phone and public datasets; the one iPhone 15 capture (`room_photos`) has no ground truth. |
| 1.4 | Photo folders must produce the **same stitched whole-property plan** | `cozmo/pipeline/photo.py::build_multi_room_photo_plan`, `cozmo/stitch/` | `benchmark_runs/saurabh_room_photo/plan.json` — 3 rooms, 2 adjacencies, one plan | **met** — per-room folders stitch into one property with adjacency. |
| 1.5 | Tier 2 — **Video**, handheld walkthrough clip | `cozmo/pipeline/video.py`, `cozmo/io/video.py` | `benchmark_runs/saurabh_room_video/plan.json` — 3 rooms from one clip | **partial** — samples frames, segments the walk into rooms by DINOv2 appearance, reuses the photo path. Exercised on a Mac-recorded clip and an iPhone Pro clip, not an iPhone 15 handheld walkthrough. |
| 1.6 | Tier 3 — **LiDAR**, depth + poses + intrinsics | `cozmo/pipeline/run.py::build_lidar_plan`, `cozmo/geometry/` | `benchmark_runs/scan_with_ceiling/plan.json` — 39.01 m², 12 walls, ceiling measured 3.07 m | **met** — the strongest tier; exact on ray-traced truth (worst wall error 0.1 cm). |
| 1.7 | All three tiers mandatory, same output contract from each | `cozmo/schema.py` | All 11 plans validate against one pydantic schema | **met** — verified by `Plan.from_json` on every run in `benchmark_runs/`. |
| 1.8 | Intervals widen honestly as sensor data thins | `cozmo/pipeline/photo.py`, `cozmo/calibration.py` | LiDAR footprint ±4.0%, photo/video ±60.2% | **met** — a 15× widening from LiDAR to photo, visible in every plan. |
| 1.9 | **Device matrix**: which tier on which hardware, accuracy each delivers | `cozmo/benchmark/score.py::build_device_matrix` | `benchmark_runs/gate_table.txt` (device-matrix block), `benchmark_runs/results.json → device_matrix` | **partial** — generated from benchmark results rather than written by hand, but every row currently reads `synthetic (LiDAR)` or the capture's own `device.model` string; no real iPhone-model-by-tier accuracy table, because captures came from two devices only. |

## Part 2 — Output contract and gates

| # | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| 2.1 | Dimensioned per-room plan: walls, ceiling height, floor area, openings | `cozmo/schema.py` (`Room`, `Wall`, `Opening`) | every `plan.json` | **met** |
| 2.2 | Stitched multi-room plan with correct adjacency | `cozmo/stitch/graph.py` | `saurabh_room_photo` (2 adjacencies), `saurabh_room_video` (2) | **partial** — adjacency is produced and overlap-resolved, but never scored against adjacency ground truth (no `element=adjacency` rows exist), so `adjacency_correctness` SKIPs on all 11 captures. |
| 2.3 | Per-surface damage regions with class and metric extent | `cozmo/semantics/detect.py`, `project.py` | LiDAR run of `apartment_lidar` → 7 regions (crack, water) with m² extents | **partial** — works at the LiDAR tier; benchmark runs used `--no-semantics` for timing. No damage ground truth exists, so precision is unvalidated. |
| 2.4 | Concealed-damage flags **with the rule that fired** | `cozmo/semantics/rules.py` | `ConcealedFlag.rule_id`, `.rule_text`, `.triggering_values` — e.g. `CD-CRACK-STRUCTURAL-01`, p=0.55, `{max_extent_m: 2.02, min_height_above_floor_m: 0.0}` | **met** — explicit YAML rule engine, not a model; the firing rule and its input values are in the output. |
| 2.5 | Scope line items keyed to surfaces | `cozmo/semantics/scope.py` | `ScopeItem.surface_id` + `.basis` showing the arithmetic | **met** |
| 2.6 | A confidence interval on **every** measurement | `cozmo/schema.py::Measurement` | Validators enforce `ci_95` brackets `value`; no bare float for any physical quantity | **met** — structurally enforced, not by convention. |
| 2.7 | One command per capture | `cozmo/cli.py` | `cozmo run --input DIR --out DIR` | **met** — tier is read from `capture.json`, never a flag. |
| 2.8 | JSON to the published schema | `cozmo/schema.py` | `SCHEMA_VERSION` stamped in every plan | **met** |
| 2.9 | Rendered plan | `cozmo/geometry/render.py` | `plan.png` + `plan.svg` per run | **met** |
| 2.10 | Benchmark set: **multi-room capture, 3+ rooms plus a connector** | `captures/saurabh_room_photo` (3 rooms), `captures/demo_fourroom` (4 folders) | — | **partial** — 3 room folders exist and stitch, but the corridor folder holds 1 photo and cannot reconstruct; no capture is a deliberate 3-rooms-plus-connector set with a connector measured. |
| 2.11 | Benchmark set: **furnished room, staged damage, two damage classes** | — | `apartment_lidar` yields crack + water opportunistically | **not met** — nothing was staged, and there is no damage ground truth to score against. |
| 2.12 | Benchmark set: **same rooms at all three tiers**, multi-room set included | `captures/apartment_lidar`, `apartment_video`, `room_photos` (all `space_id: apartment_room_a`) | — | **partial** — one space has all three tiers. The multi-room set exists at photo and video only, never at LiDAR. |
| 2.13 | Benchmark set: **one room captured twice at the same tier** | `captures/scan_floor_only`, `captures/scan_with_ceiling` (`space_id: apartment_full`) | `repeatability` and `ceiling_spread` gates fire on this pair | **met** — supplied late; both are LiDAR scans of the same flat. ⚠️ that they are the same space is inferred from matching trajectory spans and video content, not asserted by the capture operator. |
| 2.14 | Benchmark set: **laser or tape ground truth on everything** | `tests/fixtures/benchmark/ground_truth_synthetic.csv` | 9 exact rows, ray-traced | **not met** — the only ground truth is synthetic (exact by construction). No real capture has laser or tape measurements, which is why 45 gate rows SKIP. This is the single largest gap in the submission. |
| 2.15 | Gate: **opening widths** ≤2 cm on ≥85%, detection scored (missed and phantom each a miss) | `cozmo/benchmark/score.py::gate_opening_widths` | 2 PASS / 9 SKIP | **partial** — gate implemented exactly as specified including phantom/missed accounting; only scoreable on the synthetic fixtures. |
| 2.16 | Gate: **ceiling height** ≤1.5 cm per room; spread ≤1 cm across repeat captures | `gate_ceiling_height`, `gate_ceiling_spread` | ceiling_height 2 PASS / 9 SKIP; ceiling_spread 4 PASS / **1 FAIL (62.9 cm)** | **partial** — implemented; the real repeat pair fails at 62.9 cm and the report says which kind of failure it is (biased, not unrepeatable — one scan measured the ceiling, the other fell back to the 2.44 m prior). |
| 2.17 | Gate: **repeatability** ≤1 cm or 0.5% per wall | `gate_repeatability` | 1 PASS / **4 FAIL (worst 718.4 cm)** | **partial** — implemented and firing on real data; fails badly. Root cause analysed in `benchmark_report.md` (wall ids are positional, so the gate compares walls that are not the same wall, over genuinely different polygons). |
| 2.18 | Gate: **drift accountability** — state the method, ablation with it on and off; "poses used as-is" is an automatic fail | `cozmo/stitch/drift.py`, `cozmo/pipeline/run.py` | Every plan carries `drift_correction.method` + `ablation_footprint_area`. `scan_with_ceiling`: 39.01 m² on / 42.71 m² off, 12 loop closures | **met** — loop closure + scipy pose graph, switchable via `--drift-correction on\|off`, and the flag changes the geometry (re-fuses with corrected poses), not just the axes. |
| 2.19 | Gate: **photo-tier whole-property stitch** — one plan, correct adjacency, no overlaps, footprint ±8% with calibrated intervals | `cozmo/pipeline/photo.py`, `gate_room_overlap` | room_overlap 4 PASS / 7 SKIP (SKIP = single-room plans) | **partial** — stitching and zero-overlap are implemented and pass where multi-room; the ±8% footprint claim is **unverified** because no photo-tier capture has ground truth. |
| 2.20 | Photo gate ±8% walls, video ±3% | `cozmo/benchmark/score.py::TOLERANCES` | photo `wall_rel=0.08`, video `wall_rel=0.03` | **met** (thresholds encoded) / **not verified** (no GT at those tiers). |
| 2.21 | Calibration scored at every tier | `cozmo/calibrate.py`, `cozmo/calibration.py`, `gate_interval_coverage_by_kind` | `interval_coverage_by_kind`: 4 PASS / 3 FAIL, per quantity per tier | **partial** — the machinery exists (`cozmo calibrate` fits half-widths, persists a versioned file, pipeline reloads it); no calibration file is committed and fitting would be on synthetic data only. |
| 2.22 | Confident garbage on thin input caps the score | `cozmo/pipeline/photo.py`, `QualityReport.degradations` | photo/video intervals ±60%; 9 degradation strings on `saurabh_room_video` | **met** — thin input widens intervals and emits named degradations rather than a confident number. |

## Part 3 — Head-to-head vs an incumbent app

| # | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| 3.1 | LiDAR-tier output vs one consumer scanning app on 2 benchmark rooms | — | — | **not met** — no app installed, no export, no table. |
| 3.2 | Name the app and version, submit its export | — | — | **not met** |
| 3.3 | One table, your error and theirs, dimension by dimension | `benchmark_report.md` § Head-to-head | Table present with the structure and an explicit "not run" marker | **not met** — placeholder only. This is 10% of the score and is currently zero. |
| 3.4 | Beat or tie on ≥70% of shared dimensions | — | — | **not met** |

## Part 4 — The fix loop (25%)

| # | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| 4.1 | One-page fix declaration: worst gate + failing number | `fixloop/declaration.md` | **`ceiling_spread`, 62.9 cm vs a ≤1 cm gate**, committed (`62aff1d`) before the fix commit (`4be839e`) so the order is auditable | **met** |
| 4.2 | Root-cause hypothesis with evidence | `benchmark_report.md` § Repeatability | Ratio distribution across the repeat pair: 0.037–6.457, σ 1.72 > mean 1.37 → rules out scale, noise and pose | **partial** — root cause established from the data; awaiting confirmation before the fix is implemented. |
| 4.3 | Predicted post-fix number | `fixloop/declaration.md` § 3 | Predicted ≈35 cm and an explicit prediction that the gate would still fail; actual **34.5 cm**, still failing. All 7 predicted quantities landed | **met** |
| 4.4 | Ship the fix | `cozmo/geometry/planes.py::estimate_ceiling` | Commit `4be839e` — one branch, two lines of behaviour; `ceiling_spread` 62.9 → 34.5 cm on the genuine pair | **met** |
| 4.5 | Before run, after run, both regenerable by us | `fixloop/before/`, `fixloop/after/`, `scripts/run_benchmark.sh` | Both runs, same 11 captures, one command each; tags `fixloop-before` / `fixloop-after` | **met** |
| 4.6 | Readable diff | `fixloop/diff.md` | Code diff, prediction-vs-outcome table, and the whole-benchmark diff: **exactly 1 gate row changed** of 94 | **met** |

## Part 5 — Process evidence

| # | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| 5.1 | Commit as you work; history must be auditable | `.git/` | 15 commits, 2026-09-08 → 2026-09-10 | **partial** — not a single-commit dump, so it does not score zero; but the history is two days wide with large commits, which reads as compressed. |
| 5.2 | Not a repo that materialises in one or two commits | `.git/` | 15 commits across the build order (schema → io → pipeline → LiDAR → photo → stitch → video) | **met** — commit order follows the actual build order. |
| 5.3 | AI tooling allowed; every decision defended live | — | — | **owner: Saurabh** — see the defence list at the end of the technical-report outline. |

## Deliverables

| # | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| D1 | **Compliance matrix** | `compliance_matrix.md` | this file | **met** |
| D2 | **Capture route** (build or protocol) + device matrix | `cozmo/benchmark/score.py::build_device_matrix` | device matrix generated; no capture route | **partial** — device matrix ✅ (generated, not hand-written), capture route ❌. |
| D3 | **Repo**, README to a fresh capture in <15 min, one command per capture | `README.md`, `scripts/setup.sh` | **Measured: 120 s** from `git clone` to a working reconstruction on a clean machine, cold pip cache | **met** — see § Reproduction below. |
| D4 | **Reproduction bundle**: regenerate every reported number from raw inputs | `scripts/run_benchmark.sh` | `bash scripts/run_benchmark.sh` → all 11 captures + `timing.csv` + `results.json` | **met** for the pipeline's own numbers; **partial** overall, since raw captures (998 MB) are distributed outside git. |
| D5 | **Benchmark report**: gates at 3 tiers, repeatability, head-to-head, timing | `benchmark_report.md` | this run | **partial** — gates/repeatability/timing/coverage ✅, head-to-head ❌. |
| D6 | **Fix loop bundle** | `fixloop/` | `declaration.md`, `before/`, `after/`, `diff.md`, both tagged | **met** |
| D7 | **Technical report**, max 6 pages | — | — | **not met** — owner: Saurabh; outline supplied. |
| D8 | **Raw benchmark data**: sensor logs, ground truth, app exports | `captures/` (998 MB, 11 captures), `tests/fixtures/benchmark/*.csv` | sensor logs ✅, synthetic GT ✅ | **partial** — no laser/tape ground truth, no app exports. |

## The walk-in test

| # | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| W1 | Pipeline runs cold on an unseen capture, all three tiers ready | `cozmo/cli.py` | All 3 tiers ran unattended across 11 captures in this benchmark | **met** — with the caveat that photo/video accuracy on an unseen room is unverified. |
| W2 | Tier chosen on the day | `cozmo/io/capture.py` | Tier read from `capture.json`; no code path selects it | **met** |
| W3 | Scored against laser measurements on the spot | `cozmo/benchmark/score.py` | `cozmo benchmark` takes a CSV of measurements and prints the gate table | **met** — a laser CSV written on the day drops straight in. |

## Constraints

| # | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| C1 | Handheld consumer capture only | `captures/`, README § Capture provenance | 5 of 11 captures are handheld phone captures (3 LiDAR iPhone Pro, 1 iPhone 15 photo, 1 phone video). 3 are public datasets (MSR 7-Scenes Kinect, VGGT sample); 3 more are handheld but on an Android phone / a Mac, not the iPhone 15+ the brief specifies | **partial** — every capture declares its device, and the README names which are non-compliant and why. No accuracy claim rests on the non-compliant ones. |
| C2 | Any pretrained model/dataset/API **with disclosure** | `README.md`, `scripts/fetch_weights.sh`, `captures/*/capture.json` | 7 models named with HF repo + pinned commit revision; the 3 public-dataset captures carry an explicit DISCLOSURE note in their `capture.json` and in the README's provenance table | **met** |
| C3 | **Everything runs without calling your infrastructure** | `cozmo/` | `grep` for `requests\|urllib\|http://\|https://\|boto3\|api_key` across the package → **no runtime network calls**; all models load `local_files_only=True` | **met** — verifiable by grep; the only network access is `curl` inside the weight-fetch script. |
| C4 | Weights and large binaries fetched by script or volume | `scripts/fetch_weights.sh` | 7 models, `--group` selective fetch, `--check` verification | **met** |
| C5 | Weights hash-verified | `scripts/fetch_weights.sh` | **7/7 `model.safetensors` pinned by sha256** + HF commit revision in every URL; `--check` → "all weights present and verified" (26 files) | **met** — config/tokenizer JSONs are pinned by revision only, not sha256. |
| C6 | No large binaries committed | `.gitignore` | Largest object in the **entire git history** is a 332 KB JPEG (the failure-mode evidence figure); largest source file 56 KB. `/captures/` (997 MB), `/data/`, `weights/` (6.9 GB), `out/` all ignored | **met** — verified with `git rev-list --objects --all`. |
| C7 | Seeds set | `cozmo/seed.py` | `{"seed": 20260908, "seeded": ["random","numpy","open3d","torch"]}` recorded in every manifest | **met** — includes open3d's own global RNG, which otherwise makes RANSAC non-reproducible. |
| C8 | Deterministic replay | `cozmo/pipeline/run.py` (`SOURCE_DATE_EPOCH`) | LiDAR: `plan.json` **and** `plan.png` byte-identical across two runs. Photo tier (real VGGT on MPS): `plan.json` identical, max wall diff **0.00e+00 m** | **met** — verified by diff, both tiers. |
| C9 | Mirrors, glass, wet-look surfaces and low light covered | `known_failure_modes.md`, `docs/evidence/confidence_failure_modes.jpg` | Confidence-channel evidence from the real captures | **met** — with a measured, non-obvious finding: low light barely affects LiDAR depth (r = −0.08…−0.12) because the sensor is active; it degrades the passive tiers instead. |

---

## Summary

64 requirement rows, plus row 5.3 which is an owner action rather than a status.

| Status | Rows |
|---|---|
| **met** | 36 |
| **partial** | 19 |
| **not met** | 8 |

The **fix loop (25%) is now complete**: declared, shipped, before/after both
regenerable and tagged, one gate row changed of 94. `ceiling_spread` improved
62.9 cm → 34.5 cm and still fails the 1 cm gate, with that shortfall predicted
in the declaration and its cause evidenced (one of the two repeat captures
contains no ceiling information at all).

The 8 remaining **not met** rows concentrate in three places, in descending
score weight: the **head-to-head** (10%, nothing started), the **capture route**
(5%, nothing written), and the **benchmark set's ground truth** (no laser/tape
measurements), which is what turns 45 gate rows into SKIP and blocks the 15%
verified-accuracy row.
