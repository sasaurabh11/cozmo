# Cozmo AI — floor-plan reconstruction pipeline

**LiDAR tier is real.** A Stray Scanner capture goes in; a dimensioned room
polygon, a ceiling height that admits when it was not measured, detected doors
and windows, and a rendered plan come out. Photo and video tiers are still
stubs, and say so in their own output.

| Tier | Status |
|---|---|
| `lidar` | Real reconstruction: fuse → floor plane → wall layout → ceiling → openings → render, then damage → concealed-damage rules → scope |
| `photo` | Real reconstruction: VGGT → per-view planes → scale recovery → openings → render. Single room only |
| `video` | Stub. Emits a hardcoded property with `STUB PIPELINE` in `quality.warnings` |

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

## Multi-room stitching

`cozmo/stitch/` turns several separately-captured rooms into one connected
property plan, wired in for the photo tier (`cozmo run` on a capture with
multiple `rooms/<name>/` folders automatically stitches them; a single folder
still takes the single-room path, unchanged).

```
per-room reconstruction (unchanged, one call per room)
  -> cozmo.stitch.match: DINOv2 coarse retrieval, SuperPoint+LightGlue
     keypoint matching, doorway-width matching -- run in their own process
     (torch; this one already has open3d loaded for the LiDAR path)
  -> cozmo.stitch.graph: lift 2D matches to 3D via each room's own
     reconstruction, weighted 2D Procrustes per room pair, a small pose graph
     (x, y, yaw; scipy least-squares, not open3d's PoseGraph -- see below),
     Manhattan-snap the whole property, push apart anything shapely still
     says overlaps
  -> one Plan: every room placed, Adjacency objects naming the connecting
     opening and wall, confidence from match evidence
```

**Two evidence sources, not one preferred over the other.** A keypoint-based
transform is precise when there are enough inliers (`>= 4`, RANSAC-lite
consensus over the Procrustes fit); a matched doorway (two rooms each
reporting an opening of matching width) is what still works when a door is
shut and the two rooms share no visible scene at all -- the case the brief
calls the strongest signal. Both are recorded per room pair; a pair with
neither is rejected and reported, not guessed at.

**Verified two ways**, for the same reason the photo tier's single-room work
was: geometry against exact truth, real models against real photos.

- *Graph assembly* (pose graph, Manhattan snap, overlap resolution, Adjacency
  emission) is checked against synthetic rooms with known, hand-placed
  doorways -- 13 tests in `tests/test_stitch.py`, agreement to within the
  polygon-simplification tolerance.
- *Real matching* (DINOv2 + SuperPoint + LightGlue, live weights) was run on
  real photos: two disjoint real rooms correctly score a 0.086 DINOv2
  similarity and zero keypoint matches; two overlapping views of the same real
  room score 0.81 and 133 matches. A real 4-folder `cozmo run` (three
  overlapping views of one real office plus one unrelated real room) placed
  all four, correctly connected the three related views to each other,
  correctly left the unrelated room unconnected, and the new benchmark gates
  (below) confirm it: `room_overlap` PASS, `adjacency_correctness` 3/3.

**What that real run does not give**: a verified sub-8%-of-tape footprint
number for four rooms of one real house, because no such capture existed to
run it on in the time available (the honest-status pattern this project has
followed throughout — see the photo tier's own section above). Building one is
the natural next step, not further pipeline work.

**Drift correction now does something.** `--drift-correction on|off` used to
only change which axes a single room's layout was drawn in. It now runs real
loop-closure detection (spatial revisit search over the trajectory) and pose
graph correction (translation + yaw per keyframe) *before* the cloud is fused,
so the flag changes the actual geometry, not just a label. On the real
apartment capture -- a single continuous walk that happens to cross the same
hallway twice -- correction finds 5 genuine loop closures and the footprint
moves from 16.68 m² (off) to 17.82 m² (on): a real ~6.7% difference from a
real correction, not a synthetic toggle. The optimiser is our own (scipy
least-squares over x/y/z/yaw, reused by both drift.py and stitch/graph.py),
not open3d's `PoseGraph` -- that binding segfaults unconditionally on this
open3d build (0.18.0) the moment a `PoseGraphNode` is constructed, on this
platform; open3d 0.19 fixes it upstream but is not published for this
Python/platform combination. Rather than pin the project's whole numpy stack
around one binding, the graph itself -- four unknowns per node, a residual per
edge -- was small enough to own directly.

### Benchmark additions

`room_overlap` (fails if any two rooms in one plan share more than 0.02 m² of
ground -- the brief's own "must be zero") and `adjacency_correctness` (ground
truth rows `element=adjacency`, `element_id=room_a:room_b`, `value_m` 1 or 0
for "should" / "should not" be connected; a missed real connection and a
phantom one both count as a miss, the same rule the opening-detection gate
already uses). The existing `footprint` gate needed no changes -- it already
reads `property_totals.footprint_area`, which a stitched plan populates the
same way a single-room one does.

## The photo tier: 2-8 unposed stills, no depth, no poses

The hardest tier, and the point of the exercise: recover a dimensioned room
from photos with no metric sensor behind them at all.

```
frames (EXIF or FOV default) -> VGGT (scale-free cloud + poses)
  -> per-view planes -> cluster across views -> room polygon (scale-free)
  -> scale recovery (3 cues, weighted median) -> metric layout -> openings
```

### `cozmo/recon/`

| Module | What it does |
|---|---|
| [frames.py](cozmo/recon/frames.py) | Loads a photo folder; reads EXIF 35mm-equivalent focal length when present, else assumes a 65° horizontal FOV -- and records which happened |
| [backbone.py](cozmo/recon/backbone.py) | `Reconstructor` protocol (one method: `reconstruct`) + `VGGTReconstructor`. A second backbone is a registry entry, not a rewrite |
| [layout.py](cozmo/recon/layout.py) | Per-view plane fitting, cross-view clustering, polygon assembly (Plane-DUSt3R ordering) |
| [scale.py](cozmo/recon/scale.py) | Three cues -> weighted median -> one scale factor, interval widened by cue disagreement |
| [worker.py](cozmo/recon/worker.py) | Runs VGGT in its own interpreter (see below) |

### Why per-view planes, not one global RANSAC

Five photos fuse into a few thousand points scattered across a whole room --
far too sparse for the LiDAR tier's global plane fit, which needs a dense wall
band to find lines in. So the order inverts: fit floor/ceiling/wall planes
**per image first**, where that one frame's own points are still dense enough
to see clearly, then carry each hypothesis into the world frame by that
frame's pose and cluster the ones that agree. Per-view plane count and
cross-view agreement are what a sparse reconstruction can actually say about
its own confidence, and both go straight into each wall's interval.

A single monocular frame can't reveal metric "down" on its own, so per-view
fitting assumes the phone was held roughly upright (camera-local +Y ≈ gravity)
-- the ordinary case for a deliberately-taken room photo. A photo shot rolled
or tilted breaks this per-frame, which is exactly why cross-view agreement,
not any one frame's split, decides how tight a wall's interval gets.

**Fallback:** with enough frames that the fused cloud is genuinely dense (12+
views, 20k+ points -- what the video tier will hit), per-view fitting is
unnecessary and the LiDAR tier's own global RANSAC runs directly. Both paths
call the *same* polygon-assembly code
([assemble_polygon](cozmo/geometry/layout.py)) -- refactored out of
`extract_layout` for exactly this reuse, not forked. Which path ran is
recorded as `layout_method`.

### Recovering scale: three cues, weighted median, not averaged

A photo reconstruction is correct only up to an unknown global factor. Three
independent cues estimate it:

| Cue | Source | Strength |
|---|---|---|
| Metric depth alignment | ZoeDepth (NYU-indoor, metric) vs. the backbone's own depth | Strongest, when it fires |
| Door height | Grounding DINO detection × prior N(2.032 m, 0.05 m) | Medium |
| Ceiling height | Floor-to-ceiling span (scale-free) × prior N(2.44 m, 0.30 m) | Weakest, widest prior |

Combined by **weighted median**, not mean -- one bad cue (a door prior firing
on a closet) should not drag two correct ones toward it. **Cue disagreement
widens the interval directly**, on top of each cue's own claimed uncertainty:
a single cue firing alone is floored at ±30% regardless of how confident it
claims to be, because one cue cannot be cross-checked against anything. Which
cues fired, their individual estimates and their spread are all recorded under
`scale` in the plan.

### VGGT runs in its own interpreter

The upstream `vggt` package pins `numpy<2` and needs Python ≥ 3.10; the rest of
this project runs numpy 2.x on Python 3.9 (open3d and several geometry fixes
need the numpy-2 API). Forcing a downgrade into the shared venv to satisfy one
backbone would silently change behaviour everywhere else, so VGGT gets a
second virtualenv, built once:

```bash
scripts/setup_recon_env.sh          # needs python3.11 (or newer) on PATH
```

`VGGTReconstructor.reconstruct()` dispatches to `python -m cozmo.recon.worker`
under `.venv-recon` and reads the result back over JSON + npz -- the same
process-boundary pattern already used to keep open3d and torch apart (see
"The stage runs in a separate process" above), applied here to keep two numpy
majors apart instead. The `Reconstructor` protocol does not change; the
subprocess is this one backbone's own implementation detail.

### Swapping backbones

```python
from cozmo.recon.backbone import BACKBONES
BACKBONES["dust3r"] = MyDust3rReconstructor   # or --backbone dust3r on the CLI
```

`scale.py` and `layout.py` depend only on `ReconstructionResult` (points, poses,
confidence, optional per-frame local depth), so a fix-loop entry that reads
"VGGT is the worst-performing gate, try X instead" is one registry line.

### Acceptance

```bash
cozmo run --input captures/room_photos --out out/room_photos
```

Single room only -- multi-room stitching is the next phase. Openings reuse
[geometry/openings.py](cozmo/geometry/openings.py) unchanged, called on the
merged wall planes once they are in metres; nothing there was forked for this
tier.

**Honest status on the real capture.** `captures/room_photos` is 7 real stills
pulled from the apartment video, chosen for sharpness (low odometry speed).
That selection criterion turned out to pick frames clustered around one close
range of a kitchen corner -- almost no baseline between them, one wall
direction visible. The pipeline runs end to end on them (VGGT reconstructs,
per-view planes fit, scale recovers from the ceiling prior alone) and produces
a well-formed plan whose own confidence signals say exactly what's wrong: a
single fired scale cue floored at wide, `layout_method` falling back to the
data extent on the missing axis, a footprint interval spanning [13.5, 54.3] m².
That is the machinery working, not failing -- a system that returned a
confident number here would be the bug. What it does **not** give is a
verified sub-10% wall-length number against tape, because (a) this real
capture has no independent laser measurement and (b) these particular seven
photos are a poor input for room-scale photogrammetry. Getting a genuinely
good real-photo set (spread around the room, one photo per wall) needs a
deliberate capture session, which there wasn't time for in this pass -- picking
better frames from the same walkthrough, or a fresh set of real photos, is the
immediate next step, not further pipeline work.

`--backbone stub` (a trivial fixed point cloud, no weights, no subprocess) runs
the CLI/schema/manifest plumbing in under a second and is what the fast test
suite uses -- `cozmo.recon.backbone.VGGTReconstructor` is exercised for real,
separately, in `tests/test_recon.py`'s opt-in `TestVGGTBackbone` (needs
weights + `.venv-recon` + `COZMO_TEST_RECON=1`).

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

## Damage, concealed flags and scope

[cozmo/semantics/](cozmo/semantics/) is tier-agnostic by construction: it takes a
metric point cloud, RGB frames with poses, and fitted surfaces, and knows nothing
about how they were obtained. Any tier that can supply those three gets damage
regions, concealed-damage flags and scope out of it unchanged.

| Module | What it does |
|---|---|
| [detect.py](cozmo/semantics/detect.py) | Grounding DINO (open-vocabulary boxes) + SAM 2 (mask refinement), and the opening cross-check |
| [project.py](cozmo/semantics/project.py) | Projects a mask onto a fitted plane; measures extent in m²; merges across frames |
| [rules.py](cozmo/semantics/rules.py) | YAML rule engine for concealed damage |
| [scope.py](cozmo/semantics/scope.py) | CSV lookup → line items with cut-back margins and a `basis` string |
| [worker.py](cozmo/semantics/worker.py) | Runs the stage in its own process (see below) |

Prompts are `door`, `window`, `doorway` for openings, and `water stain`,
`mould`, `cracked drywall`, `burn mark`, `missing drywall` for damage. Weights
come from `scripts/fetch_weights.sh`, pinned to a Hugging Face commit revision
and checked by sha256, loaded `local_files_only` and never committed.

### The stage runs in a separate process

open3d and torch each bundle their own OpenMP runtime. A process holding both
either aborts (`OMP: Error #179: pthread_mutex_init failed`) or **deadlocks
inside the first inference call at 0% CPU with no error at all** — reproducibly,
on this machine, in both import orders. `KMP_DUPLICATE_LIB_OK` papers over it and
is documented as unsafe, so the semantic stage gets its own interpreter and
everything crossing the boundary is JSON. It costs one interpreter start, and it
also means a detector that dies on a bad frame loses the damage findings rather
than the whole run. Nothing under `cozmo/semantics/` may import
`cozmo.geometry`; a test asserts it.

For the same reason the detector tests are opt-in — pytest would otherwise load
open3d and torch together:

```bash
COZMO_TEST_DETECTOR=1 pytest tests/test_semantics.py -k Detector
```

### Openings are cross-checked, not replaced

Geometry finds a doorway as a hole in a wall plane; the detector finds one as a
door. They fail differently — a mirror or a dark recess fools the geometry, an
unobserved wall defeats it entirely, and a poster of a door fools the detector —
so an opening is reported when **either** source fires, and `detection_sources`
records which agreed. Detector-only openings carry three times the interval,
because their extent comes from a projected mask rather than from depth, and
they are held to the same plausible width bands, so a partial mask implying a
0.42 m "door" is dropped rather than becoming a phantom.

### Concealed damage is a rule engine, not a model

The contract requires the rule that fired, so [rules.yaml](cozmo/semantics/rules.yaml)
holds seven rules with structured predicates — `all_of` / `any_of` / `none_of`
over `{field, op, value}` leaves. There is no `eval`: a rule file is a data file,
and a data file that can execute arbitrary Python is not one. A rule naming an
unknown field is a **load error**, because the failure mode of a rule engine is a
rule that quietly never fires. Every flag carries `triggering_values` — the
actual numbers that satisfied it:

```json
{
  "rule_id": "CD-WATER-SUBFLOOR-01",
  "triggering_values": {"damage_class": "water", "surface_kind": "wall",
                        "min_height_above_floor_m": 0.0},
  "probability": 0.72, "inspection_priority": 1
}
```

### Scope quantities show their arithmetic

[scope_items.csv](cozmo/semantics/scope_items.csv) maps (damage class, surface
kind) to line items with a cut-back margin, a waste factor and a minimum charge.
Four bases: `area` (the patch, grown by the cut-back), `surface` (the whole
surface, for work that cannot stop at the damage — a repaint flashes, soot is not
confined to the scorch mark), `extent` (linear work), `count`. Every line states
how it got its number:

```
DRY-RMV-2   1.80 m2  basis: damaged patch 0.62 x 0.74 m, cut back 0.30 m each
                            side -> 1.22 x 1.34 m = 1.63 m2, +10% waste = 1.80 m2
```

A (damage class, surface kind) pair with no catalogue row produces **no line
item** rather than a guess.

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

- **Video reconstruction.** Still emits `build_stub_plan`; the seam is the same
  one the photo tier just came out of.
- **Multi-room photo stitching.** One room per photo folder today.
- **Damage/scope at the photo tier.** `cozmo.semantics` is tier-agnostic and could
  attach here, but wiring it in is not done yet -- the photo tier's Plan carries
  empty `damage`/`concealed_flags`/`scope`.
- **Exact ground truth for the photo tier's own real-capture test.** The
  synthetic LiDAR fixtures have exact wall lengths because they're ray-traced;
  VGGT needs real texture to reconstruct anything, so the photo-tier check runs
  against real photos of the apartment capture with no independent laser
  measurement -- the acceptance criterion here is a well-formed, plausible
  reconstruction with intervals that behave correctly (widen on disagreement,
  bracket the estimate), not a verified sub-10% error against tape.
- **Multi-room segmentation and stitching.** The LiDAR tier emits one room. On
  the real capture it returns the region the operator actually walked, which is
  one room of a larger apartment; adjacency and whole-property stitching are the
  next stage.
- **Loop closure and a pose graph.** The only drift handling today is the
  Manhattan/plane-anchored correction described above, and the plan claims
  exactly that and nothing more.
- **Calibrated intervals.** See above — LiDAR intervals are asserted.
- **Detector precision.** Open-vocabulary prompts are noisy: "cracked drywall"
  at a low threshold returns the wall. Damage is held to a higher score
  threshold than openings and masks covering more than 35% of a frame are
  dropped, but nothing here is validated against damage ground truth yet — the
  benchmark has no damage rows.
- **Spatial matching of openings in the scorer.** Ground truth is matched to
  plans by id, so opening ground truth has to be keyed to the reconstruction's
  ids. Openings should be matched by position along the wall instead.
- **Head-to-head against a consumer scanning app.**
