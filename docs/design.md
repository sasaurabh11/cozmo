# Design notes

How the pipeline works and why. [README](../README.md) covers install and running.

Companion docs: [technical_report.md](../technical_report.md) ·
[benchmark_report.md](../benchmark_report.md) ·
[compliance_matrix.md](../compliance_matrix.md) ·
[known_failure_modes.md](../known_failure_modes.md)

---

## 1. Captures

Every capture has a `capture.json`. `tier` decides everything; it is never a CLI flag.

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
| `photo` | `rooms/<room>/*.jpg` | [photo.py](../cozmo/io/photo.py) |
| `video` | one `.mov`/`.mp4` | [capture.py](../cozmo/io/capture.py) |
| `lidar` | Stray Scanner: `rgb.mp4`, `camera_matrix.csv`, `odometry.csv`, `depth/`, `confidence/` | [stray.py](../cozmo/io/stray.py) |

[stray.py](../cozmo/io/stray.py) is a verified loader, used unmodified.
[lidar.py](../cozmo/io/lidar.py) adapts it and holds ingest checks.

Its conventions are load-bearing downstream: depth is uint16 mm at 256×192, RGB
1920×1440, depth intrinsics are RGB intrinsics scaled by `depth_width/rgb_width`,
the odometry quaternion is camera-to-world, world y is up.

> **Depth is the camera-frame z of the hit, not distance along the ray.**
> Treating it as ray length inflates every dimension by 1/cos(angle from the
> optical axis) — about 8% at the frame edge.

## 2. Output contract

[schema.py](../cozmo/schema.py), pydantic v2, `extra="forbid"`. One rule runs
through it: **every physical quantity is a `Measurement`** —
`{value, ci_95: [lo, hi], unit}`, never a bare float. A number without an
interval is a claim that cannot be defended.

`Plan` holds `schema_version`, `capture_id`, `tier`, `pipeline_version`,
`generated_at`, `scale`, `drift_correction`, `property_totals`, `rooms`,
`adjacencies`, `damage`, `concealed_flags`, `scope`, `quality`.

Ids are validated across the whole document: an opening on an unknown wall,
damage on an unknown surface, or a scope item citing damage that does not exist
are all rejected at parse time.

## 3. LiDAR tier

Five stages in [cozmo/geometry/](../cozmo/geometry/), each usable alone:

| Stage | What it does |
|---|---|
| [fuse.py](../cozmo/geometry/fuse.py) | Depth frames → world cloud. Confidence ≥ 2 only, voxel downsample |
| [planes.py](../cozmo/geometry/planes.py) | RANSAC floor, then a ceiling *if there is one* |
| [layout.py](../cozmo/geometry/layout.py) | Room axes from normals, RANSAC wall lines, cells → polygon |
| [openings.py](../cozmo/geometry/openings.py) | Doors and windows as holes in wall planes |
| [render.py](../cozmo/geometry/render.py) | `plan.png`, `plan.svg` |

Four decisions worth defending:

**The floor is not fitted by RANSAC alone.** Plain RANSAC returns the largest
plane, which in a furnished room is often a wall — on the real capture it settled
6–9 cm above the floor. A vertical histogram picks the seed first: the *lowest*
band with substantial support, not the densest, since a bed can out-populate the
floor it stands on. RANSAC then fits within that band, recovering tilt. The
sample floor is 0.7° off level; across 9 m that is 11 cm, so every height is a
signed distance from that plane, never raw `y`.

**A capture that never looks up cannot measure a ceiling.** Only 3.7% of points
sit above camera height in the sample capture; fitting a plane to those returns a
confident number derived from a light fitting. `quality.ceiling_method` records
which path ran:

| Method | Meaning |
|---|---|
| `measured_plane` | A real ceiling with real support |
| `wall_extrapolation` | No ceiling, but walls stop together high up — read as the wall/ceiling junction |
| `scan_cutoff_prior` | Walls stop at inconsistent or low heights. This is where the *scan* ended, not the room. Only a lower bound is defensible, so the value is a structural prior with a wide interval |

The real capture gets 2.44 m [1.89, 2.99]. Reporting the 1.7 m the walls stopped
at — because nobody looked up — would be a confident answer to a question the
capture never asked.

Distinguishing a ceiling from wall-tops needs care: a ring of wall-tops fits a
horizontal plane beautifully. Coverage does not separate them (a partly-seen
ceiling covers little of its footprint). Erosion does: a one-cell ring has no
cell whose neighbours are all occupied; any real ceiling patch does.

**Rooms are rectilinear, but not axis-aligned.** Wall normals are histogrammed
after multiplying the angle by four, so directions 90° apart count as one family;
the resultant length doubles as a Manhattan-ness score. The polygon is the union
of arrangement cells supported by interior evidence — floor points plus the
camera path, which is interior by definition. An L-shaped room is a different set
of kept cells, not a special case. Letting a cell edge fall on the raw data
extent stretched a 3.60 m wall to 3.89 m, which is why boundaries always land on
a fitted wall.

**"No points here" has two causes:** an opening, or wall nobody pointed the phone
at. Detection works one column at a time — a door is a run of columns empty from
the floor up with wall on both sides; a window is a run whose gap closes above
and below at consistent heights. Treating empty cells as 2D blobs failed: the
doorway merged with the unobserved strip above it and a visible door went
undetected. Every wall reports the fraction of its length that returned anything,
because "no openings found" in a wall observed across 12% of its length is not
evidence the wall is solid. Five of ten walls on the real capture fall below 60%,
and `quality.degradations` says so.

### Drift correction

`--drift-correction on|off` runs real loop-closure detection (spatial revisit
search) and pose-graph correction (translation + yaw per keyframe) *before*
fusion, so the flag changes geometry, not a label. On the real apartment capture
it finds 5 genuine loop closures and the footprint moves 16.68 m² (off) →
17.82 m² (on) — a real 6.7% difference.

The optimiser is ours (scipy least-squares over x/y/z/yaw, shared by `drift.py`
and `stitch/graph.py`), not open3d's `PoseGraph`: that binding segfaults
unconditionally on open3d 0.18.0 on this platform the moment a `PoseGraphNode` is
constructed. 0.19 fixes it upstream but is not published for this
Python/platform pair. Four unknowns per node and one residual per edge was small
enough to own outright rather than pin the whole numpy stack around one binding.

## 4. Photo tier — 2–8 unposed stills, no depth, no poses

The hardest tier, and the point of the exercise.

```
frames (EXIF or FOV default) → VGGT (scale-free cloud + poses)
  → per-view planes → cluster across views → polygon (scale-free)
  → scale recovery (3 cues, weighted median) → metric layout → openings
```

| Module | What it does |
|---|---|
| [frames.py](../cozmo/recon/frames.py) | Loads a photo folder; reads EXIF 35mm-equivalent focal length, else assumes 65° horizontal FOV — and records which |
| [backbone.py](../cozmo/recon/backbone.py) | `Reconstructor` protocol (one method) + `VGGTReconstructor` |
| [layout.py](../cozmo/recon/layout.py) | Per-view plane fitting, cross-view clustering, polygon assembly |
| [scale.py](../cozmo/recon/scale.py) | Three cues → weighted median → one scale factor |
| [worker.py](../cozmo/recon/worker.py) | Runs VGGT in its own interpreter |

**Per-view planes, not one global RANSAC.** Five photos fuse into a few thousand
points across a whole room — too sparse for a global fit, which needs a dense
wall band. So the order inverts: fit floor/ceiling/wall planes *per image*, where
that frame's own points are dense enough, then carry each hypothesis into the
world frame by that frame's pose and cluster those that agree. Per-view plane
count and cross-view agreement are what a sparse reconstruction can honestly say
about its own confidence, and both feed each wall's interval.

A single monocular frame cannot reveal metric "down", so per-view fitting assumes
the phone was roughly upright (camera-local +Y ≈ gravity). A rolled or tilted
photo breaks this per-frame — which is exactly why cross-view agreement, not any
one frame, decides how tight an interval gets.

*Fallback:* with 12+ views and 20k+ points, per-view fitting is unnecessary and
the LiDAR global RANSAC runs directly. Both paths call the **same**
[assemble_polygon](../cozmo/geometry/layout.py) — refactored out for reuse, not
forked. Which ran is recorded as `layout_method`.

### Scale: three cues, weighted median

A photo reconstruction is correct only up to an unknown global factor.

| Cue | Source | Strength |
|---|---|---|
| Metric depth alignment | ZoeDepth (NYU-indoor, metric) vs the backbone's own depth | Strongest, when it fires |
| Door height | Grounding DINO × prior N(2.032 m, 0.05 m) | Medium |
| Ceiling height | Floor-to-ceiling span × prior N(2.44 m, 0.30 m) | Weakest, widest prior |

**Weighted median, not mean** — one bad cue (a door prior firing on a closet)
should not drag two correct ones toward it. **Cue disagreement widens the
interval directly**, on top of each cue's own claimed uncertainty: a single cue
firing alone is floored at ±30%, because one cue cannot be cross-checked. Which
cues fired, their estimates and their spread are recorded under `scale`.

### VGGT runs in its own interpreter

Upstream `vggt` pins `numpy<2` and needs Python ≥3.10; the rest of the project
runs numpy 2.x. Forcing a downgrade into the shared venv would silently change
behaviour everywhere else, so VGGT gets a second virtualenv:

```bash
scripts/setup_recon_env.sh
```

`VGGTReconstructor.reconstruct()` dispatches to `python -m cozmo.recon.worker`
under `.venv-recon` and reads results back over JSON + npz — the same
process-boundary pattern used to keep open3d and torch apart, here keeping two
numpy majors apart. The protocol does not change; the subprocess is one
backbone's implementation detail.

### Swapping backbones

```python
from cozmo.recon.backbone import BACKBONES
BACKBONES["dust3r"] = MyDust3rReconstructor   # or --backbone dust3r
```

`scale.py` and `layout.py` depend only on `ReconstructionResult`, so replacing
the backbone is one registry line.

`--backbone stub` (fixed cloud, no weights, no subprocess) runs the CLI/schema/
manifest plumbing in under a second and is what the fast test suite uses.

### Honest status

`captures/room_photos` is 7 real stills pulled from the apartment video, chosen
for sharpness. That criterion picked frames clustered around one close range of a
kitchen corner — almost no baseline, one wall direction visible. The pipeline
runs end to end and produces a well-formed plan whose own confidence signals say
exactly what is wrong: one scale cue floored wide, `layout_method` falling back to
the data extent on the missing axis, a footprint interval spanning
[13.5, 54.3] m². **That is the machinery working.** A confident number here would
be the bug.

What it does *not* give is a verified sub-10% wall-length number against tape,
because this capture has no laser measurement and these seven photos are poor
input for room-scale photogrammetry. A deliberate capture session — one photo per
wall, spread around the room — is the next step, not more pipeline work.

## 5. Video tier — more views, same code

**There is no separate video reconstruction pipeline**, and that is disclosed
rather than hidden. [video.py](../cozmo/io/video.py) decodes at a fixed stride
(`--video-stride`, default 15 ≈ one frame per half second at 30 fps), drops
frames below a Laplacian-variance blur floor (`--video-blur-threshold`), and
keeps survivors up to `--video-max-frames` (default 8 — the backbone's own 2–8
view contract). Those frames become an ordinary photo folder.

Frames are kept **temporally stratified**, not globally sharpest: taking the top
8 by sharpness returned the first 11 s of a 37 s walk and collapsed the whole
capture to one room.

A walkthrough crosses doorways, so one clip is not one room. Sampled frames are
split into per-room runs by appearance — DINOv2 descriptors of consecutive
frames, cut where agreement drops, since crossing a doorway changes almost
everything in view at once. Each run becomes its own folder, at which point the
walkthrough *is* a photo capture and
[`build_multi_room_photo_plan`](../cozmo/pipeline/photo.py) does the rest
unchanged. A walk that never leaves one room segments into one run. On a real
37 s clip this recovers 3 rooms with 2 adjacencies.

`quality.video_sampling` records the stride, threshold and cap alongside what
actually happened — frames decoded, surviving the stride, surviving the blur
filter, finally used — so a thin or blurry walkthrough is visible in the output
rather than silently degrading.

## 6. Multi-room stitching

```
per-room reconstruction (one call per room)
  → stitch.match: DINOv2 retrieval, SuperPoint+LightGlue keypoints,
    doorway-width matching (own process — torch)
  → stitch.graph: lift 2D matches to 3D, weighted 2D Procrustes per pair,
    pose graph (x, y, yaw), Manhattan-snap, push apart shapely overlaps
  → one Plan: rooms placed, Adjacency naming the connecting opening and wall
```

**Two evidence sources, neither preferred.** A keypoint transform is precise with
enough inliers (≥4, RANSAC-lite consensus over the Procrustes fit). A matched
doorway — two rooms each reporting an opening of matching width — still works
when a door is shut and the rooms share no visible scene, the case the brief
calls the strongest signal. Both are recorded per pair; a pair with neither is
**rejected and reported, not guessed at**.

Verified two ways:

- *Graph assembly* against synthetic rooms with hand-placed doorways — 13 tests
  in `tests/test_stitch.py`, agreement within polygon-simplification tolerance.
- *Real matching* with live weights: two disjoint real rooms score 0.086 DINOv2
  similarity and zero keypoint matches; two overlapping views of one real room
  score 0.81 and 133 matches. A real 4-folder run placed all four, connected the
  three related views, and correctly left the unrelated room unconnected —
  `room_overlap` PASS, `adjacency_correctness` 3/3.

What that does **not** give: a verified sub-8%-of-tape footprint for four rooms
of one real house, because no such capture existed to run it on.

## 7. Damage, concealed flags and scope

[cozmo/semantics/](../cozmo/semantics/) is tier-agnostic by construction: it
takes a metric cloud, posed RGB frames and fitted surfaces, and knows nothing
about how they were obtained.

| Module | What it does |
|---|---|
| [detect.py](../cozmo/semantics/detect.py) | Grounding DINO (open-vocabulary boxes) + SAM 2 (masks), opening cross-check |
| [project.py](../cozmo/semantics/project.py) | Projects a mask onto a fitted plane, measures m², merges across frames |
| [rules.py](../cozmo/semantics/rules.py) | YAML rule engine for concealed damage |
| [scope.py](../cozmo/semantics/scope.py) | CSV lookup → line items with cut-back margins and a `basis` string |

Prompts: `door`, `window`, `doorway` for openings; `water stain`, `mould`,
`cracked drywall`, `burn mark`, `missing drywall` for damage. Weights come from
`scripts/fetch_weights.sh`, pinned to a Hugging Face commit revision, checked by
sha256, loaded `local_files_only`, never committed.

**The stage runs in a separate process.** open3d and torch each bundle their own
OpenMP runtime. A process holding both either aborts (`OMP: Error #179`) or
**deadlocks inside the first inference call at 0% CPU with no error at all** —
reproducibly, in both import orders. `KMP_DUPLICATE_LIB_OK` papers over it and is
documented as unsafe, so the stage gets its own interpreter and everything
crossing the boundary is JSON. It costs one interpreter start, and a detector
that dies on a bad frame loses the damage findings rather than the whole run.
Nothing under `cozmo/semantics/` may import `cozmo.geometry`; a test asserts it.

For the same reason detector tests are opt-in:
`COZMO_TEST_DETECTOR=1 pytest tests/test_semantics.py -k Detector`

**Openings are cross-checked, not replaced.** Geometry finds a doorway as a hole;
the detector finds one as a door. They fail differently — a mirror or dark recess
fools geometry, an unobserved wall defeats it entirely, a poster of a door fools
the detector — so an opening is reported when **either** fires, and
`detection_sources` records which agreed. Detector-only openings carry 3× the
interval (extent comes from a projected mask, not depth) and are held to the same
plausible width bands, so a partial mask implying a 0.42 m "door" is dropped.

**Concealed damage is a rule engine, not a model.** The contract requires the rule
that fired, so [rules.yaml](../cozmo/semantics/rules.yaml) holds seven rules with
structured predicates — `all_of`/`any_of`/`none_of` over `{field, op, value}`
leaves. There is no `eval`: a data file that can execute arbitrary Python is not a
data file. A rule naming an unknown field is a **load error**, because the failure
mode of a rule engine is a rule that quietly never fires. Every flag carries the
numbers that satisfied it:

```json
{"rule_id": "CD-WATER-SUBFLOOR-01",
 "triggering_values": {"damage_class": "water", "surface_kind": "wall",
                       "min_height_above_floor_m": 0.0},
 "probability": 0.72, "inspection_priority": 1}
```

**Scope quantities show their arithmetic.**
[scope_items.csv](../cozmo/semantics/scope_items.csv) maps (damage class, surface
kind) to line items with a cut-back margin, waste factor and minimum charge. Four
bases: `area` (the patch, grown by the cut-back), `surface` (the whole surface,
for work that cannot stop at the damage — a repaint flashes), `extent` (linear),
`count`.

```
DRY-RMV-2   1.80 m2  basis: damaged patch 0.62 x 0.74 m, cut back 0.30 m each
                            side -> 1.22 x 1.34 m = 1.63 m2, +10% waste = 1.80 m2
```

A pair with no catalogue row produces **no line item** rather than a guess.

## 8. Intervals and calibration

Asserted half-widths: wall ±(2 cm + 1% of length), openings ±5 cm (one occupancy
cell), areas ±4%. Each is reasoned from something structural — occupancy-grid
quantisation at LiDAR, view count and scale-cue disagreement at photo/video — and
that reasoning is never discarded. Each is then scaled by a per-tier,
per-quantity factor fitted against ground truth. The factor is 1.0 until
`cozmo calibrate` has actually run.

```bash
cozmo calibrate --captures tests/fixtures/captures \
    --ground-truth tests/fixtures/benchmark/ground_truth_synthetic.csv \
    --out calibration/calibration.json
```

It runs the *real* pipeline (all factors forced to 1.0 for the pass — fitting
against an already-calibrated prediction would just re-derive the previous
factor), pairs measurements against truth using exactly the pairing the
`interval_coverage` gate uses, and per (tier, quantity kind) fits the smallest
half-width factor covering ≥95% of truth values.

**The factor floors at 1.0** — calibration only ever widens here. Narrowing a
generous placeholder on the handful of captures a benchmark this size supplies is
the exact "confident garbage on thin evidence" failure this project exists to
catch.

```
TIER   QUANTITY        N  SCALE  COVERAGE  TRUSTED
lidar  wall_length     8  1.230  95.0%     yes
photo  wall_length     3  4.100  100.0%    no
```

`N` and `TRUSTED` (≥3 samples) sit next to the factor so a fit backed by three
captures is never mistaken for one backed by three hundred. The file is versioned
and loaded at runtime (`COZMO_CALIBRATION_FILE`), so a new fit changes reported
uncertainty without a code change. It is **not committed**: a fit from a thin
synthetic set is not a number worth shipping. Uncalibrated runs say so in
`quality.calibration_note`.

## 9. Benchmark scoring

`ground_truth.csv`:

```
room,element,element_id,dimension,value_m,method,notes
living_room,wall,living_room_w0,length,4.000,laser,south wall
living_room,room,living_room,ceiling_height,2.450,laser,
living_room,opening,op_lr_door,width,0.810,tape,door to bedroom
,property,property,footprint_area,22.500,derived,sum of room areas
```

| Gate | Threshold |
|---|---|
| `wall_lengths` | LiDAR ≤ max(2 cm, 1%), video ±3%, photo ±8%, on ≥85% of walls |
| `ceiling_height` | ≤1.5 cm (LiDAR); 1.5% / 3% at video / photo |
| `ceiling_spread` | ≤1 cm across repeat captures of one room |
| `opening_widths` | ≤2 cm on ≥85% — **detection scored** |
| `footprint` | LiDAR ±2%, video ±3%, photo ±8% |
| `repeatability` | ≤1 cm or 0.5% per wall, between two captures of one room at one tier |
| `interval_coverage` | ≥90% of 95% intervals contain truth |
| `interval_coverage_by_kind` | Same, broken out by tier and quantity kind |
| `room_overlap` | Fails if two rooms share >0.02 m² of ground |
| `adjacency_correctness` | Truth rows `element=adjacency`; a missed and a phantom connection both count as misses |

Two rules the scorer will not bend:

1. **A gate with no ground truth is SKIP, never PASS.** Silence must never read
   as success.
2. **Detection is part of the opening gate.** The denominator is
   matched + missed + phantom, so a pipeline cannot buy accuracy by reporting
   only the openings it is confident about.

Repeats are found structurally — plans grouped by `(tier, room_id)` — so no
bookkeeping enables the repeatability and spread gates.

**Device matrix.** Every run prints tier × device class × accuracy per gate,
generated from that run's gate results and each capture's `run_manifest.json`
(where device info actually lives), not written by hand. A plan scored without
its manifest reports device class `unknown` rather than a guess.

**Assumptions, stated rather than buried:**

- The brief gives ceiling height (1.5 cm) and opening widths (2 cm) without a
  tier. Applied literally at LiDAR, widened at video/photo. Ceiling height widens
  *less* than plan-scale looseness (1.5%/3% against ±3%/±8% on walls): a ceiling
  is one vertical extent measured in one place and does not accumulate the pose
  drift that stretches a wall run. Inheriting the wall tolerance would have given
  the photo tier a 19.6 cm ceiling gate, which is not a gate.
- LiDAR wall lengths and footprint have no stated Round 2 gate; max(2 cm, 1%) and
  ±2% are inherited from Round 1 and are ours to defend.
- Interval coverage is gated at 90% for nominal 95% intervals, allowing
  finite-sample slack.

## 10. Fixtures

**Captures.** `tests/fixtures/captures/` holds two ray-traced Stray Scanner
captures of a 3.60 × 2.80 m room, 2.50 m ceiling, 0.85 m door, 1.10 m window,
yawed 23° off the world frame. `synthetic_room` observes the ceiling;
`synthetic_no_ceiling` drops the ceiling returns, as when an operator never
points the phone up. Ground truth is written from the same constants the renderer
used, so it is exact.

Capture fixtures are **generated, not committed** — 472 KB of depth PNGs and MP4s
that a 0.25 s script reproduces exactly and that would otherwise drift from the
code defining them. `pytest` rebuilds them when missing or when a generator
changes.

**Hand-authored gate-table plans.** `tests/fixtures/benchmark/` holds a two-room
ground truth and three synthetic plans, built directly rather than run through
the pipeline, so the scorer can be exercised without a capture:

| Capture | Tier | Demonstrates |
|---|---|---|
| `cap_lidar_a` | lidar | Clean capture, every gate passes |
| `cap_lidar_b` | lidar | Repeat of the same property: living-room walls wander (repeatability + ceiling-spread FAIL), one window out by 2.1 cm; bedroom repeats cleanly |
| `cap_photo_a` | photo | Walls inside ±8%, but a missed window, a phantom closet, an inflated footprint and intervals far too tight — confident garbage on thin input |

Regenerate with `python tests/fixtures/generate.py`.

## 11. Layout

```
cozmo/
  schema.py            output contract (pydantic v2)
  calibration.py       loads/applies fitted interval factors at runtime
  calibrate.py         `cozmo calibrate`: fits factors from (prediction, truth) pairs
  cli.py               typer CLI: run, benchmark, calibrate, version
  seed.py              random / numpy / torch seeding, recorded per run
  io/                  stray.py (verified, as-is), photo.py, video.py, capture.py
  geometry/            LiDAR: fuse, planes, layout, openings, render
  recon/               photo/video backbone (VGGT), scale recovery, per-view layout
  stitch/              multi-room matching, pose graph, drift/loop-closure correction
  semantics/           damage, concealed-damage rules, scope (tier-agnostic)
  pipeline/            run.py (tier dispatch), photo.py (shared assembly), video.py
  benchmark/score.py   truth in → gate table + device matrix + results.json out
tests/                 schema, scorer, pipeline, geometry, recon, stitch, semantics, CLI
scripts/fetch_weights.sh
```

## 12. Deliberately not here yet

- **Damage/scope at photo and video.** Openings *are* detected there (Grounding
  DINO, via [photo_worker.py](../cozmo/semantics/photo_worker.py)), but the rest
  of the semantic stage is not wired in: those Plans carry empty
  `damage`/`concealed_flags`/`scope`.
- **Room segmentation at LiDAR.** The LiDAR path emits exactly one room per
  capture, so a whole-flat walkthrough becomes a single polygon covering an
  arbitrary part of the flat. This is why "stitched plan from every tier" is met
  at photo and video but not LiDAR, why two scans of one flat disagree by 27% on
  footprint, and the open root cause behind the worst gate in the benchmark.
- **Exact ground truth for photo/video real captures.** The synthetic LiDAR
  fixtures are exact because they are ray-traced; VGGT needs real texture, so the
  acceptance criterion at those tiers is a well-formed reconstruction with
  intervals that behave correctly, not a verified sub-10% error against tape.
- **A held-out calibration split.** `cozmo calibrate` fits and evaluates on the
  same set — see §8 for why, and why every factor prints its sample count.
- **Detector precision.** Open-vocabulary prompts are noisy: "cracked drywall" at
  a low threshold returns the wall. Damage is held to a higher threshold and
  masks over 35% of a frame are dropped, but nothing is validated against damage
  ground truth — the benchmark has no damage rows.
- **Spatial matching of openings in the scorer.** Truth is matched by id, so
  opening truth must be keyed to reconstruction ids. It should match by position
  along the wall instead.
- **Head-to-head against a consumer scanning app.**
