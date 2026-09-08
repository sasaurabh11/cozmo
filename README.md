# Cozmo AI — floor-plan reconstruction pipeline

**LiDAR tier is real.** A Stray Scanner capture goes in; a dimensioned room
polygon, a ceiling height that admits when it was not measured, detected doors
and windows, and a rendered plan come out. Photo and video tiers are still
stubs, and say so in their own output.

| Tier | Status |
|---|---|
| `lidar` | Real reconstruction: fuse → floor plane → wall layout → ceiling → openings → render |
| `video` | Stub. Emits a hardcoded property with `STUB PIPELINE` in `quality.warnings` |
| `photo` | Stub, as above |

On a 1715-frame capture of a real room the whole pipeline runs in **2.9 s**.
On the synthetic fixture, where the answer is known exactly, it recovers
3.60 m and 2.80 m walls to **1 mm**, a 2.50 m ceiling to **9 mm**, and both
openings to within one occupancy cell.

## Install and run in under 5 minutes

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
./scripts/fetch_weights.sh            # no-op today; the hook exists
python tests/fixtures/synthesize.py   # build the capture fixtures (0.25 s)

# Reconstruct a room from a LiDAR capture, with a drawing
cozmo run --input tests/fixtures/captures/synthetic_room --out out/room
open out/room/plan.png

# Score that reconstruction against ground truth
cozmo benchmark --results out --ground-truth \
    tests/fixtures/benchmark/ground_truth_synthetic.csv --out out/bench
```

The fixture's room is ray-traced from known constants, so the second command
scores a real reconstruction against an exact answer: every LiDAR gate passes,
worst wall error 0.1 cm. It validates the geometry and the sensor conventions —
not real-world accuracy, since synthetic depth has no noise.

For the stub tiers and a table with deliberate failures in it:

```bash
cozmo run --input tests/fixtures/captures/demo_photo --out out/demo_photo
cozmo benchmark \
    --results tests/fixtures/benchmark/results \
    --ground-truth tests/fixtures/benchmark/ground_truth.csv \
    --out out/bench_stub
```

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
                [--stride N] [--voxel-size M] [--verbose]
cozmo benchmark --results DIR --ground-truth CSV --out DIR [--strict]
```

One command per capture. **Tier is never a flag** — it is read from
`capture.json` inside the input directory, because the tier is a fact about the
capture and a flag would let a photo capture be scored against LiDAR gates by
typo.

`--drift-correction off` is the ablation arm, and at the LiDAR tier it does
something real. With it **on**, wall normals are histogrammed to find the room's
own axes and the layout is built on those. With it **off**, ARKit's world axes
are used exactly as they came:

| Capture | Correction on | Correction off |
|---|---|---|
| Synthetic room (truth 10.08 m²) | **10.08 m²** | 21.04 m² |
| Real capture, room A | **8.05 m²** | 16.59 m² |

Rooms are not aligned to the capture's world frame — the real one sits 23° off,
and the fixture is built 23° off for the same reason — so taking the pose frame
as given roughly doubles the footprint. Each arm records the other's number in
`drift_correction.ablation_footprint_area`, so one run gives you both.

No loop closure: the sample walk ends 3.18 m from where it started, so there is
no loop to close, and the plan says that rather than claiming a pose graph.

### `cozmo run` writes

- **`plan.json`** — the output contract (below).
- **`plan.png` / `plan.svg`** — the drawing, at the LiDAR tier.
- **`run_manifest.json`** — git commit, branch and dirty flag; pipeline and
  schema version; the exact command; the seed and what it seeded; SHA-256 of the
  input directory (paths *and* bytes) and of every output; per-stage timings;
  the fusion, floor, layout, ceiling and opening statistics. A reported number
  that cannot be tied back to the bytes that produced it is not reproducible, so
  the tie is written every run.

Three runs over the same capture produce byte-identical plans. Getting there
required seeding **Open3D's** RNG as well as numpy's and Python's: `segment_plane`
draws from its own global generator, and without seeding it the floor plane
moved between runs and every dimension downstream moved with it — which is
exactly the failure the repeatability gate exists to catch.

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

[stray.py](cozmo/io/stray.py) is the verified loader, used as-is and not modified
here. [lidar.py](cozmo/io/lidar.py) adapts it to the pipeline and is where
ingest-time checks live. Its conventions — depth is uint16 millimetres at
256×192, RGB 1920×1440, depth intrinsics are the RGB intrinsics scaled by
`depth_width / rgb_width`, the odometry quaternion is camera-to-world, world y is
up — are load-bearing everywhere downstream. **Depth is the camera-frame z of the
hit, not the distance along the ray**; treating it as ray length inflates every
dimension by 1/cos(angle from the optical axis), about 8% at the frame edge.

## The LiDAR reconstruction

Five stages in [cozmo/geometry/](cozmo/geometry/), each usable on its own:

| Stage | What it does |
|---|---|
| [fuse.py](cozmo/geometry/fuse.py) | Unprojects depth frames to a world cloud. Confidence ≥ 2 only; voxel downsample; `--stride` exposed |
| [planes.py](cozmo/geometry/planes.py) | RANSAC floor, then a ceiling *if there is one* |
| [layout.py](cozmo/geometry/layout.py) | Room axes from normals, RANSAC wall lines, cell arrangement → polygon |
| [openings.py](cozmo/geometry/openings.py) | Doors and windows as holes in wall planes |
| [render.py](cozmo/geometry/render.py) | `plan.png` and `plan.svg` |

Four decisions in there are worth defending:

**The floor is fitted, but not by RANSAC alone.** Plain RANSAC on the raw cloud
returns whichever plane is largest, which in a furnished room is often a wall —
on the real capture it settled 6–9 cm above the floor. The vertical histogram
picks the seed first: the *lowest* band with substantial support, not the
densest, since a bed can out-populate the floor it stands on. RANSAC then fits
within that band, which is what recovers the tilt. The sample room's floor is
0.7° off level, and 0.7° across a 9 m room is 11 cm — so heights everywhere else
are signed distances from that plane, never raw `y`.

**A capture that never looks up cannot measure a ceiling.** In the sample
capture 3.7% of points sit above camera height. Fitting a plane to that returns
a confident number derived from a light fitting and the top of a door frame. The
plan records which path ran, in `quality.ceiling_method`:

- `measured_plane` — a real ceiling with real support.
- `wall_extrapolation` — no ceiling, but the walls stop together high up, so
  that agreement is read as the wall/ceiling junction.
- `scan_cutoff_prior` — the walls stop at inconsistent heights, or too low. This
  is where the scan ended, not where the room does. The only defensible claim is
  a lower bound, so the reported value is a structural prior and the interval
  runs from the highest observed wall point to 3.0 m. The real capture gets
  2.44 m [1.89, 2.99]: reporting the 1.7 m the walls stopped at, because nobody
  looked up, would be a confident answer to a question the capture did not ask.

Telling a ceiling from the top edge of the walls needs care, because a ring of
wall-tops fits a horizontal plane beautifully. Coverage does not separate them —
a partly-seen ceiling covers little of its footprint. Erosion does: a one-cell
ring has no cell whose neighbours are all occupied, and any real patch of
ceiling does.

**Rooms are rectilinear, but not axis-aligned.** Wall normals are histogrammed
after multiplying the angle by four, so directions 90° apart count as one wall
family; the resultant length doubles as a Manhattan-ness score, reported in the
plan. The room polygon is then the union of arrangement cells supported by
interior evidence — floor points, plus the camera path, which is interior by
definition. That is why an L-shaped room is just a different set of kept cells
rather than a special case, and why the boundary always lands on a fitted wall:
letting a cell edge fall on the data extent stretched a 3.60 m wall to 3.89 m.

**"No points here" has two causes.** An opening, and a piece of wall nobody
pointed the phone at. Detection works one column at a time — a door is a run of
columns empty from the floor up with wall on both sides; a window is a run of
columns whose gap is closed above and below at a consistent height. Treating the
empty cells as 2D blobs fails: on the fixture the doorway merged with the
unobserved strip of wall above it and a plainly visible door went undetected.
Every wall reports the fraction of its length that returned anything, because
"no openings found" in a wall observed across 12% of its length is not evidence
that the wall is solid. On the real capture five of ten walls fall below 60%,
and the plan says so in `quality.degradations`.

### Intervals at this tier are placeholders

Wall ±(2 cm + 1% of length), openings ±5 cm (one occupancy cell), areas ±4%.
These are asserted, not calibrated, and `quality.calibration_note` says so. The
ceiling interval is the exception — it comes from the estimator and means
something.

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

### Captures

`tests/fixtures/captures/` holds two ray-traced Stray Scanner captures of a
3.60 × 2.80 m room with a 2.50 m ceiling, a 0.85 m door and a 1.10 m window,
yawed 23° off the world frame. `synthetic_room` observes the ceiling;
`synthetic_no_ceiling` is identical with the ceiling returns dropped, as when an
operator never points the phone up. Ground truth is written from the same
constants the renderer used, so `ground_truth_synthetic.csv` is exact.

Capture fixtures are **generated, not committed**: 472 KB of depth PNGs and MP4s
that this 0.25 s script reproduces exactly, and that would otherwise drift from
the code defining them. `pytest` rebuilds them automatically when they are
missing or when either generator changes; run
`python tests/fixtures/synthesize.py` to build them by hand for CLI use.

Text fixtures — ground truth CSVs and the stub-tier `plan.json` files — stay
committed: they are small, they diff, and they are meant to be read in review.

### Stub-tier plans

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

- **Photo and video reconstruction.** Both still emit `build_stub_plan` in
  [run.py](cozmo/pipeline/run.py), which is the seam: replace its body, keep its
  signature, and the contract, CLI, manifest and scoreboard carry over.
- **Multi-room segmentation and stitching.** The LiDAR tier emits one room. On
  the real capture it returns the region the operator actually walked, which is
  one room of a larger apartment; adjacency and whole-property stitching are the
  next stage.
- **Loop closure and a pose graph.** The only drift handling today is the
  Manhattan/plane-anchored correction described above, and the plan claims
  exactly that and nothing more.
- **Calibrated intervals.** See above — LiDAR intervals are asserted.
- **Damage, concealed-damage rules and scope** at the LiDAR tier. The contract
  carries them and the stub populates them; the reconstruction does not.
- **Spatial matching of openings in the scorer.** Ground truth is matched to
  plans by id, so opening ground truth has to be keyed to the reconstruction's
  ids. Openings should be matched by position along the wall instead.
- **Head-to-head against a consumer scanning app.**
