# Cozmo AI — floor-plan reconstruction pipeline (skeleton)

A working scoreboard before any algorithm. The output contract, the CLI, the
capture readers, the provenance trail and the gate scorer are real; the
reconstruction is a stub that emits a hardcoded property and says so in its own
output. The point is that when a real algorithm lands, there is already
something that will tell it, honestly, that it is wrong.

**Status:** no vision code. Every plan produced by `cozmo run` carries
`STUB PIPELINE` in `quality.warnings`. Do not report any number it prints.

## Install and run in under 5 minutes

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
./scripts/fetch_weights.sh            # no-op today; the hook exists

cozmo run --input tests/fixtures/captures/demo_photo --out out/demo_photo
cozmo benchmark \
    --results tests/fixtures/benchmark/results \
    --ground-truth tests/fixtures/benchmark/ground_truth.csv \
    --out out/bench
```

The second command prints a gate table with both PASS and FAIL rows against the
committed fixtures, and writes `out/bench/results.json`.

With Docker instead:

```bash
docker build -t cozmo .
docker run --rm -v "$PWD:/data" cozmo \
    benchmark -r /data/tests/fixtures/benchmark/results \
              -g /data/tests/fixtures/benchmark/ground_truth.csv \
              -o /data/out/bench
```

## The two commands

```
cozmo run       --input DIR --out DIR [--drift-correction on|off] [--seed N]
cozmo benchmark --results DIR --ground-truth CSV --out DIR [--strict]
```

One command per capture. **Tier is never a flag** — it is read from
`capture.json` inside the input directory, because the tier is a fact about the
capture and a flag would let a photo capture be scored against LiDAR gates by
typo.

`--drift-correction off` is the ablation arm: it stitches with poses as-is and
reports the corrected footprint in `drift_correction.ablation_footprint_area`,
so the two runs diff on exactly the number the drift gate asks about.

### `cozmo run` writes two files

- **`plan.json`** — the output contract (below).
- **`run_manifest.json`** — git commit, branch and dirty flag; pipeline and
  schema version; the exact command; the seed and what it seeded; SHA-256 of the
  input directory (paths *and* bytes) and of the plan; a summary of what the
  capture actually contained. A reported number that cannot be tied back to the
  bytes that produced it is not reproducible, so the tie is written every run.

Set `SOURCE_DATE_EPOCH` to make `plan.json` byte-identical across reruns; the
test suite uses this to assert determinism.

## Capture directories

Every capture carries a `capture.json`:

```json
{
  "capture_id": "demo_photo",
  "tier": "photo",
  "space_id": "demo_property",
  "device": {"model": "iPhone 15", "os_version": "18.5", "has_lidar": false},
  "declared_rooms": ["living_room", "bedroom"]
}
```

`space_id` is shared by repeat captures of the same physical space.

| Tier | Layout | Reader |
|---|---|---|
| `photo` | `rooms/<room>/*.jpg` (or bare `<room>/` folders, or loose images) | [photo.py](cozmo/io/photo.py) |
| `video` | one `.mov`/`.mp4`, or `video_path` in `capture.json` | [capture.py](cozmo/io/capture.py) |
| `lidar` | Stray Scanner layout: `rgb.mp4`, `camera_matrix.csv`, `odometry.csv`, `depth/`, `confidence/` | [stray.py](cozmo/io/stray.py) |

[stray.py](cozmo/io/stray.py) is a placeholder to be replaced by the existing
loader; keep `StrayCapture`, `load_stray_capture` and `summarize` and nothing
downstream changes.

## Output contract

[schema.py](cozmo/schema.py), pydantic v2, `extra="forbid"`. One rule runs
through it: **every physical quantity is a `Measurement`** —
`{value, ci_95: [lo, hi], unit}` — never a bare float. A number without an
interval is a claim that cannot be defended, and the scorer treats a missing or
dishonest interval as a failed calibration row rather than a free pass.

`Plan` holds `schema_version`, `capture_id`, `tier`, `pipeline_version`,
`generated_at`, `scale` (how metric scale was recovered), `drift_correction`
(method, loop closures, residual, ablation footprint), `property_totals`,
`rooms` (walls, openings, surfaces, ceiling height, floor area, perimeter, pose
in the property frame), `adjacencies`, `damage`, `concealed_flags` (each with
the rule id and rule text that fired), `scope` (line items keyed to surface ids)
and `quality`. Ids are validated across the whole document: an opening on an
unknown wall, damage on an unknown surface or a scope item citing damage that
does not exist are all rejected at parse time.

## Benchmark scoring

`ground_truth.csv`:

```
room,element,element_id,dimension,value_m,method,notes
living_room,wall,living_room_w0,length,4.000,laser,south wall
living_room,room,living_room,ceiling_height,2.450,laser,
living_room,opening,op_lr_door,width,0.810,tape,door to bedroom
,property,property,footprint_area,22.500,derived,sum of room areas
```

`element` ∈ wall | opening | room | property. `element_id` matches the id in the
plan. `dimension` ∈ length | width | height | ceiling_height | floor_area |
footprint_area.

| Gate | Threshold |
|---|---|
| `wall_lengths` | LiDAR ≤ max(2 cm, 1%), video ±3%, photo ±8%, on ≥ 85% of walls |
| `ceiling_height` | ≤ 1.5 cm every room (LiDAR); widened to 1.5% / 3% at video / photo |
| `ceiling_spread` | ≤ 1 cm across repeat captures of one room |
| `opening_widths` | ≤ 2 cm on ≥ 85% — **detection scored**: a missed opening and a phantom opening each count as a miss |
| `footprint` | LiDAR ±2%, video ±3%, photo ±8% |
| `repeatability` | ≤ 1 cm or 0.5% per wall, between two captures of one room at one tier |
| `interval_coverage` | ≥ 90% of 95% intervals contain the truth, mean half-width reported alongside |

Two rules the scorer will not bend:

1. **A gate with no ground truth behind it is SKIP, never PASS.** Silence must
   never read as success.
2. **Detection is part of the opening gate.** The denominator is
   matched + missed + phantom, so a pipeline cannot buy accuracy by reporting
   only the openings it is confident about.

Repeats are found structurally — plans are grouped by `(tier, room_id)`, which
is exactly "two captures of the same room at the same tier" — so no bookkeeping
is needed to enable the repeatability and spread gates.

### Assumptions, stated rather than buried

- The brief gives ceiling height (1.5 cm) and opening widths (2 cm) without a
  tier. They are applied literally at LiDAR and widened at video and photo.
  Ceiling height widens *less* than the tier's plan-scale looseness (1.5% / 3%,
  against ±3% / ±8% on walls): a ceiling is one vertical extent measured in one
  place and does not accumulate the pose drift that stretches a wall run, and
  inheriting the wall tolerance would have handed the photo tier a 19.6 cm
  ceiling gate, which is not a gate.
- LiDAR wall lengths and footprint have no stated Round 2 gate; max(2 cm, 1%)
  and ±2% are inherited from Round 1 and are ours to defend.
- Interval coverage is gated at 90% for nominal 95% intervals, allowing
  finite-sample slack on a benchmark of this size.

## Fixtures

`tests/fixtures/benchmark/` holds a two-room ground truth and three synthetic
plans, each built to land on a specific side of a specific gate:

| Capture | Tier | What it demonstrates |
|---|---|---|
| `cap_lidar_a` | lidar | clean capture, every gate passes |
| `cap_lidar_b` | lidar | repeat of the same property: living-room walls wander (repeatability FAIL, ceiling spread FAIL) and one window is out by 2.1 cm; the bedroom repeats cleanly |
| `cap_photo_a` | photo | walls hold inside ±8%, but a missed window, a phantom closet, an inflated footprint and intervals far too tight — confident garbage on thin input |

Regenerate with `python tests/fixtures/generate.py`; the values live there as
readable intent rather than as magic JSON.

## Tests

```bash
pytest
```

Covering the schema round-trip and its referential integrity, gate arithmetic at
its boundaries (including that a perfect-but-incomplete opening detector still
fails), determinism of the run path, the provenance the manifest must carry, and
both CLI commands end to end.

## Layout

```
cozmo/
  schema.py            output contract (pydantic v2)
  cli.py               typer CLI: run, benchmark, version
  seed.py              random / numpy / torch seeding, recorded per run
  io/  stray.py        Stray Scanner loader (placeholder)
       photo.py        per-room photo folders
       capture.py      tier dispatch on capture.json
  pipeline/run.py      single-capture orchestrator (stubbed reconstruction)
  benchmark/score.py   ground truth in, gate table + results.json out
tests/                 schema, scorer, pipeline, CLI + fixtures
scripts/fetch_weights.sh
```

## What is deliberately not here yet

Reconstruction of any kind — no depth fusion, plane fitting, pose graph, opening
detection or damage segmentation. `build_stub_plan` in
[run.py](cozmo/pipeline/run.py) is the seam: replace its body, keep its
signature, and the contract, CLI, manifest and scoreboard carry over unchanged.
The stitched-plan renderer and the head-to-head comparison against a consumer
scanning app are also still to come.
