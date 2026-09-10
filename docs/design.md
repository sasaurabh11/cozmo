# Design notes

How the pipeline works and why it is built this way — the reasoning behind each
tier, the error budget, and the decisions worth defending. The [README](../README.md)
covers install and running; this file covers everything else.

Companion documents: [technical_report.md](../technical_report.md) ·
[benchmark_report.md](../benchmark_report.md) ·
[compliance_matrix.md](../compliance_matrix.md) ·
[known_failure_modes.md](../known_failure_modes.md)

---

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
| `photo` | `rooms/<room>/*.jpg` (or bare `<room>/` folders, or loose images) | [photo.py](../cozmo/io/photo.py) |
| `video` | one `.mov`/`.mp4`, or `video_path` in `capture.json` | [capture.py](../cozmo/io/capture.py) |
| `lidar` | Stray Scanner layout: `rgb.mp4`, `camera_matrix.csv`, `odometry.csv`, `depth/`, `confidence/` | [stray.py](../cozmo/io/stray.py) |

[stray.py](../cozmo/io/stray.py) is the verified loader, used as-is and not modified
here. [lidar.py](../cozmo/io/lidar.py) adapts it to the pipeline and is where
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
| [frames.py](../cozmo/recon/frames.py) | Loads a photo folder; reads EXIF 35mm-equivalent focal length when present, else assumes a 65° horizontal FOV -- and records which happened |
| [backbone.py](../cozmo/recon/backbone.py) | `Reconstructor` protocol (one method: `reconstruct`) + `VGGTReconstructor`. A second backbone is a registry entry, not a rewrite |
| [layout.py](../cozmo/recon/layout.py) | Per-view plane fitting, cross-view clustering, polygon assembly (Plane-DUSt3R ordering) |
| [scale.py](../cozmo/recon/scale.py) | Three cues -> weighted median -> one scale factor, interval widened by cue disagreement |
| [worker.py](../cozmo/recon/worker.py) | Runs VGGT in its own interpreter (see below) |

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
([assemble_polygon](../cozmo/geometry/layout.py)) -- refactored out of
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

One folder is one room; several folders stitch into one property, see
[Multi-room stitching](#multi-room-stitching). Openings reuse
[geometry/openings.py](../cozmo/geometry/openings.py) unchanged, called on the
merged wall planes once they are in metres; nothing there was forked for this
tier. Where the reconstructed cloud is too sparse for that geometric detector
to see a gap -- which is most real photo captures -- the semantic detector
supplies openings instead, and each one records which source found it in
`detection_sources`.

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

## The video tier: more views, same code

A deliberate simplification, disclosed here and in the technical report
rather than hidden: **there is no separate video reconstruction pipeline.**
[`cozmo/io/video.py`](../cozmo/io/video.py) decodes the walkthrough clip at a
fixed stride (`--video-stride`, default 15 -- roughly one sampled frame every
half second of a 30 fps walk), drops any sampled frame whose Laplacian
variance falls below a blur floor (`--video-blur-threshold`, default 80 --
motion blur and out-of-focus pans read low, sharp texture reads high), then
keeps the sharpest survivors up to `--video-max-frames` (default 8, the
reconstruction backbone's own 2-8 view contract). Those frames are written as
an ordinary photo folder and handed straight to
[`cozmo/pipeline/photo.py`](../cozmo/pipeline/photo.py)'s existing
`_reconstruct_room` — unmodified. More views, same code.

```bash
cozmo run --input captures/walkthrough --out out/walkthrough \
    --video-stride 15 --video-blur-threshold 80 --video-max-frames 8
```

Every sampling decision is recorded, not just applied: `quality.video_sampling`
in the plan carries the stride, threshold and cap alongside what actually
happened -- frames decoded, how many survived the stride, how many survived
the blur filter, how many were finally used -- so a thin or blurry walkthrough
is visible in the output rather than silently degrading.

A walkthrough usually crosses doorways, so one clip is not one room. The
sampled frames are split into per-room runs by appearance -- DINOv2
descriptors of consecutive frames, cut where agreement drops, since crossing a
doorway changes almost everything in view at once -- and each run is written
into its own folder. At that point the walkthrough *is* a photo capture, and
[`build_multi_room_photo_plan`](../cozmo/pipeline/photo.py) does the rest
unchanged: reconstruction, stitching, adjacency, overlap resolution. A walk
that never leaves one room segments into one run and takes the single-room
path. On a real 37 s clip this recovers 3 rooms with 2 adjacencies.

## The LiDAR reconstruction

Five stages in [cozmo/geometry/](../cozmo/geometry/), each usable on its own:

| Stage | What it does |
|---|---|
| [fuse.py](../cozmo/geometry/fuse.py) | Unprojects depth frames to a world cloud. Confidence ≥ 2 only; voxel downsample; `--stride` exposed |
| [planes.py](../cozmo/geometry/planes.py) | RANSAC floor, then a ceiling *if there is one* |
| [layout.py](../cozmo/geometry/layout.py) | Room axes from normals, RANSAC wall lines, cell arrangement → polygon |
| [openings.py](../cozmo/geometry/openings.py) | Doors and windows as holes in wall planes |
| [render.py](../cozmo/geometry/render.py) | `plan.png` and `plan.svg` |

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

### Intervals: asserted first, then calibrated

Wall ±(2 cm + 1% of length), openings ±5 cm (one occupancy cell), areas ±4%.
These half-widths are reasoned from something structural -- occupancy-grid
quantisation at this tier, view count and scale-cue disagreement at photo and
video -- and that reasoning is never thrown away. What changes is that every
one of them is now scaled by a per-tier, per-quantity factor fitted against
ground truth (see [Calibration](#calibration)); the factor is 1.0, i.e. no
change, until `cozmo calibrate` has actually run and produced a file. The
ceiling interval is the one exception worth calling out on its own -- it comes
from the ceiling estimator, not an asserted constant, and is calibrated the
same way as everything else.

## Calibration

`cozmo calibrate` closes the loop between an asserted interval and a checked
one:

```bash
cozmo calibrate --captures tests/fixtures/captures \
    --ground-truth tests/fixtures/benchmark/ground_truth_synthetic.csv \
    --out calibration/calibration.json
```

It runs the *real* pipeline (every calibration factor forced to 1.0 for this
pass -- fitting against an already-calibrated prediction would just re-derive
the previous factor) over every `capture.json`-rooted capture under
`--captures`, pairs every measurement against ground truth using exactly the
pairing `cozmo benchmark`'s own `interval_coverage` gate uses, and for each
(tier, quantity kind) group fits the smallest half-width scale factor whose
scaled interval covers ≥ 95% of that group's truth values. The factor floors
at 1.0 -- calibration only ever widens an interval here; narrowing an
already-generous placeholder on the handful of captures a benchmark this size
can supply is exactly the "confident garbage on thin evidence" failure mode
the project exists to catch, not commit.

```
TIER   QUANTITY        N  SCALE  COVERAGE  TRUSTED
-----  --------------  -  -----  --------  -------
lidar  wall_length     8  1.230  95.0%     yes
photo  wall_length     3  4.100  100.0%    no
```

`N` and `TRUSTED` (≥ 3 calibration samples) sit right next to the factor so a
fit backed by three captures is never mistaken for one backed by three
hundred. The fitted file is versioned (`calibration_version`, `fitted_at`,
which captures and ground truth produced it) and is loaded by the pipeline at
runtime — from `calibration/calibration.json` by default, or wherever
`COZMO_CALIBRATION_FILE` points — so a newly fitted file changes reported
uncertainty on the next run without a code change. It is not committed (see
`.gitignore`): a fit from a thin or synthetic benchmark set is not a number
worth shipping, and every tier runs uncalibrated (factor 1.0, stated plainly
in `quality.calibration_note`) when no file is present.

`cozmo benchmark` reports the same achieved-coverage numbers the fit is
checked against, broken out by tier and quantity kind, as
`interval_coverage_by_kind` gate rows — so a benchmark run shows where
calibration is and isn't earning its keep without needing to re-run
`cozmo calibrate` first.

## Damage, concealed flags and scope

[cozmo/semantics/](../cozmo/semantics/) is tier-agnostic by construction: it takes a
metric point cloud, RGB frames with poses, and fitted surfaces, and knows nothing
about how they were obtained. Any tier that can supply those three gets damage
regions, concealed-damage flags and scope out of it unchanged.

| Module | What it does |
|---|---|
| [detect.py](../cozmo/semantics/detect.py) | Grounding DINO (open-vocabulary boxes) + SAM 2 (mask refinement), and the opening cross-check |
| [project.py](../cozmo/semantics/project.py) | Projects a mask onto a fitted plane; measures extent in m²; merges across frames |
| [rules.py](../cozmo/semantics/rules.py) | YAML rule engine for concealed damage |
| [scope.py](../cozmo/semantics/scope.py) | CSV lookup → line items with cut-back margins and a `basis` string |
| [worker.py](../cozmo/semantics/worker.py) | Runs the stage in its own process (see below) |

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

The contract requires the rule that fired, so [rules.yaml](../cozmo/semantics/rules.yaml)
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

[scope_items.csv](../cozmo/semantics/scope_items.csv) maps (damage class, surface
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

[schema.py](../cozmo/schema.py), pydantic v2, `extra="forbid"`. One rule runs
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
| `interval_coverage_by_kind` | Same threshold, broken out by tier and quantity kind (wall_length, opening_width, ceiling_height, floor_area, footprint_area) across every capture in the run — see [Calibration](#calibration) |

Two rules the scorer will not bend:

1. **A gate with no ground truth behind it is SKIP, never PASS.** Silence must
   never read as success.
2. **Detection is part of the opening gate.** The denominator is
   matched + missed + phantom, so a pipeline cannot buy accuracy by reporting
   only the openings it is confident about.

Repeats are found structurally — plans are grouped by `(tier, room_id)`, which
is exactly "two captures of the same room at the same tier" — so no bookkeeping
is needed to enable the repeatability and spread gates.

### Device matrix

Every `cozmo benchmark` run also prints a device matrix — tier × device class
× measured accuracy per gate — generated from that run's own gate results and
each capture's `run_manifest.json` (which is where device info actually lives;
`plan.json` itself doesn't carry it), not written by hand:

```
TIER   DEVICE             GATE               N  PASS RATE  WORST VALUE
-----  -----------------  -----------------  -  ---------  -----------
lidar  synthetic (LiDAR)  wall_lengths       1  100%       1.0000
```

A `plan.json` scored without its sibling manifest (e.g. the hand-authored
gate-table fixtures below) reports device class `unknown` rather than a
guess. The table is also written into `results.json` under `device_matrix`.

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

Text fixtures — ground truth CSVs and the hand-authored `plan.json` files below
— stay committed: they are small, they diff, and they are meant to be read in
review.

### Hand-authored gate-table plans

`tests/fixtures/benchmark/` holds a two-room ground truth and three synthetic
plans -- built directly, not run through the pipeline, so the scorer can be
exercised against known-good and known-bad numbers without needing a capture
at all. Each lands on a specific side of a specific gate:

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
  calibration.py        loads/applies fitted interval factors at runtime
  calibrate.py           `cozmo calibrate`: fits factors from (prediction, truth) pairs
  cli.py                typer CLI: run, benchmark, calibrate, version
  seed.py               random / numpy / torch seeding, recorded per run
  io/  stray.py         Stray Scanner loader (verified, used as-is)
       photo.py         per-room photo folders
       video.py         video frame sampling (stride + blur filter + cap)
       capture.py       tier dispatch on capture.json
  geometry/             LiDAR: fuse, floor/ceiling planes, layout, openings, render
  recon/                photo/video backbone (VGGT), scale recovery, per-view layout
  stitch/               multi-room matching, pose graph, drift/loop-closure correction
  semantics/            damage, concealed-damage rules, scope (tier-agnostic)
  pipeline/run.py       single-capture orchestrator; dispatches on tier
       photo.py         photo tier + shared single-room assembly (video reuses it)
       video.py         video tier: sample frames, hand to the photo path
  benchmark/score.py    ground truth in, gate table + device matrix + results.json out
tests/                  schema, scorer, pipeline, geometry, recon, stitch, semantics, CLI + fixtures
scripts/fetch_weights.sh
```

## What is deliberately not here yet

- **Damage/scope at the photo and video tiers.** Openings *are* detected at
  those tiers (Grounding DINO, via
  [semantics/photo_worker.py](../cozmo/semantics/photo_worker.py)), but the rest
  of the semantic stage is not wired in: their Plans carry empty
  `damage`/`concealed_flags`/`scope`. Only the LiDAR tier produces damage
  regions, concealed flags and scope today.
- **Room segmentation at the LiDAR tier.** The LiDAR path emits exactly one
  room per capture. A whole-flat walkthrough therefore becomes a single room
  polygon covering an arbitrary part of the flat, which is why the brief's
  "stitched plan from every tier" is met at photo and video but *not* at
  LiDAR, and why two scans of one flat disagree by 27% on footprint. This is
  the open root cause behind the worst gate in the benchmark.
- **Exact ground truth for the photo/video tiers' own real-capture tests.** The
  synthetic LiDAR fixtures have exact wall lengths because they're ray-traced;
  VGGT needs real texture to reconstruct anything, so a real-capture check at
  those tiers runs against real photos/video with no independent laser
  measurement -- the acceptance criterion there is a well-formed, plausible
  reconstruction with intervals that behave correctly (widen on disagreement,
  bracket the estimate), not a verified sub-10% error against tape.
- **A held-out calibration split.** `cozmo calibrate` fits and evaluates on the
  same benchmark set -- see [Calibration](#calibration) for why, and why every
  fitted factor is printed next to its sample count.
- **Detector precision.** Open-vocabulary prompts are noisy: "cracked drywall"
  at a low threshold returns the wall. Damage is held to a higher score
  threshold than openings and masks covering more than 35% of a frame are
  dropped, but nothing here is validated against damage ground truth yet — the
  benchmark has no damage rows.
- **Spatial matching of openings in the scorer.** Ground truth is matched to
  plans by id, so opening ground truth has to be keyed to the reconstruction's
  ids. Openings should be matched by position along the wall instead.
- **Head-to-head against a consumer scanning app.**
