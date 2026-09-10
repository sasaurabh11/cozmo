# Cozmo AI — Technical Report

Floor-plan reconstruction and damage-scope pipeline, three input tiers.
12,244 lines across 46 modules · 206 tests (202 run by default, 4 opt-in against
real model weights) · one command per capture.

Companion documents: [compliance_matrix.md](compliance_matrix.md) ·
[benchmark_report.md](benchmark_report.md) ·
[known_failure_modes.md](known_failure_modes.md) ·
[fixloop/declaration.md](fixloop/declaration.md)

---

## 1. Architecture

One contract, three tiers, one command.

```
capture.json ──► load_capture() ──► tier dispatch
                                      ├── lidar  → build_lidar_plan()
                                      ├── photo  → build_photo_plan()
                                      └── video  → build_video_plan()
                                            ↓
                            plan.json · plan.png/svg · run_manifest.json
```

Three decisions shape everything downstream.

**Every physical quantity is a `Measurement`, never a bare float.**
`schema.py` defines `Measurement(value, ci_95, unit)` with validators that the
interval is ordered and brackets the value. There is no code path that emits a
dimension without an interval, because the type system forbids it. Counts,
ids and unitless confidences stay plain.

**Tier is a property of the capture, not of the invocation.** It is read from
`capture.json`; there is no `--tier` flag. A flag would let a photo capture be
scored against LiDAR gates by typo, and the tier is a fact about how the data
was collected, not a runtime choice.

**Every torch model runs in a separate interpreter.** open3d and torch each
bundle their own OpenMP runtime; a process holding both aborts with
`OMP: Error #179` or deadlocks inside the first inference call, reproducibly,
at 0% CPU. VGGT adds a second, independent reason: upstream pins `numpy<2` and
requires Python ≥3.10, while open3d here needs numpy 2.x on 3.9. So there are
four subprocess workers (`semantics/worker.py`, `semantics/photo_worker.py`,
`stitch/worker.py`, `recon/worker.py`), everything crosses as JSON, and VGGT
runs under its own `.venv-recon`. The `Reconstructor` interface — one call, one
result — does not change; the subprocess is an implementation detail of one
backbone.

**Reproducibility is structural.** Every run stamps `run_manifest.json` with the
git commit, the exact command, the seed record, and a SHA-256 over the input
directory (paths *and* bytes). Seeding covers Python, numpy, torch **and
open3d's own global RNG** — `segment_plane` draws from it, and without seeding
it the floor plane moved between runs and every dimension moved with it. With
`SOURCE_DATE_EPOCH` pinned, two runs of the LiDAR tier produce byte-identical
`plan.json` *and* `plan.png`; the photo tier through real VGGT on MPS produces
identical plans with a maximum wall difference of 0.00e+00 m.

**Degrade, never fake.** A missing detector, an unobserved ceiling, an
unreconstructable room — each degrades the plan and says so in
`quality.warnings` / `quality.degradations`, rather than failing the run or
inventing a number.

## 2. Tier design and device matrix

### LiDAR — the strongest tier, and it uses no models at all

```
correct_trajectory → fuse_capture → fit_floor → extract_layout
                   → estimate_ceiling → detect_openings → semantics → Plan
```

Pure geometry (open3d RANSAC), so it needs **0 GB of weights** and runs in
2.5 s on a 16-frame fixture, 6.8 s on 1,715 frames, 68.8 s on 9,745.

On the ray-traced fixture, where the answer is known to the millimetre:

| Quantity | Truth | Recovered | Gate | Result |
|---|---|---|---|---|
| Walls | 3.60 / 2.80 m | worst error **0.1 cm** | ≤ max(2 cm, 1%) on ≥85% | PASS |
| Ceiling | 2.50 m | worst error **0.4 cm** | ≤1.5 cm | PASS |
| Openings | 0.85 / 1.10 m | 2/2, **0 missed, 0 phantom** | ≤2 cm on ≥85% | PASS |
| Footprint | 10.08 m² | **10.08 m²** (0.0%) | ±2% | PASS |

This validates the geometry and the Stray Scanner sensor conventions. It says
nothing about real-world noise: ray-traced depth has none.

### Photo — VGGT, and a deliberate separation

```
frames → VGGT (scale-free) → per-view planes → cluster → polygon
       → scale recovery → metres → openings
```

The design decision worth defending: **per-view plane fitting, not one global
RANSAC.** A cloud from 5–8 photos is far too sparse for the LiDAR path's global
fitter — fitting one plane to that is fitting noise. So planes are fitted per
image, where each frame's own cloud is locally dense, then carried into a
common frame by that frame's pose and clustered. The merged wall lines then
enter the *same* `assemble_polygon` the LiDAR tier uses; nothing is forked.

The second: **VGGT returns geometry, not size** (`scale_is_metric = False`).
Recovering metres is `recon/scale.py`'s separate job. Conflating "what the
network saw" with "how big it thinks the room is" would make a backbone swap
expensive; as built, swapping to DUSt3R/MASt3R is a class plus a registry entry.

Multi-room: per-room reconstruction → DINOv2 shortlist → SuperPoint/LightGlue
matching → pose graph → Manhattan snap → overlap resolution → one property.
`saurabh_room_photo` yields 3 rooms, 2 adjacencies, **zero room overlap**.

### Video — more views, same code

No separate reconstruction pipeline. The clip is decoded at a fixed stride,
frames below a Laplacian-variance floor are dropped, and the sharpest survivor
of each equal time-slice is kept. What a video needs *beyond* a photo folder is
one decision a photo capture never poses: **which room is this frame of?**
Consecutive frames are embedded with DINOv2 and cut where agreement drops —
crossing a doorway changes nearly everything in view at once — and each run
becomes a room folder. At that point the walkthrough *is* a photo capture and
the multi-room photo path runs unchanged. A 37 s clip yields 3 rooms, 2
adjacencies.

Two measured findings shaped this. First, the blur threshold: real H.264 phone
video has Laplacian variance an order of magnitude below JPEG stills (measured
1.9–34, median 6.7), so a still-photo threshold silently rejects every video
frame. Second, and more damaging: capping by *global* sharpness collapsed the
selection onto the steadiest stretch — on a 1,105-frame walk it took all 8
frames from the first 11 seconds and discarded two rooms entirely. Sampling
per time-slice fixed it. There is a regression test that fails against the old
sampler.

### Device matrix

Generated from benchmark results, not written by hand
(`benchmark/score.py::build_device_matrix`, emitted into `results.json`):

| Tier | Device | Gates with ground truth | Pass rate | Honest accuracy |
|---|---|---|---|---|
| lidar | synthetic (ray-traced) | wall, ceiling, opening, footprint | **100%** | wall 0.1 cm, ceiling 0.4 cm |
| lidar | iPhone Pro-class | none | — | **unverified** — no laser truth |
| photo | iPhone 15 | none | — | **unverified** |
| photo | OnePlus Nord 2T | none | — | **unverified** |
| video | iPhone Pro-class / Mac | none | — | **unverified** |

The empty rows are the honest content of this table. **Only the synthetic
fixtures carry ground truth**, so accuracy on real hardware is not established
at any tier. Interval width is the one thing that *is* device-differentiated and
real: **±4.0% at LiDAR against ±60.2% at photo and video.**

Device compliance is also imperfect and disclosed in the README: the brief
specifies iPhone 15+ for photo/video, and three captures use an Android phone
or a Mac while three more are public datasets (MSR 7-Scenes, VGGT samples).
Those are development fixtures; no accuracy claim rests on them.

## 3. Drift handling

`--drift-correction on|off` is a real ablation, not a label: with it on, the
trajectory is corrected *before* fusion, so the flag changes the geometry rather
than just the axes the room is drawn on.

Loop closures are detected on keyframes within a radius, then a pose graph over
(x, y, z, yaw) is solved with scipy least-squares, node 0 fixed. **open3d's own
`PoseGraph` was not usable** — `PoseGraphNode` segfaults unconditionally on
0.18.0 with numpy 2 on this platform, with a minimal reproducer, and 0.19 has no
build for this Python/platform. The scipy optimiser is ~90 lines and is shared
by the room-stitching graph.

| Capture | On | Off | Δ | Loop closures | Method |
|---|---|---|---|---|---|
| Synthetic (truth 10.08 m²) | **10.08 m²** | 13.41 m² | +33% | 0 | `pose_graph` |
| `apartment_lidar` | **17.82 m²** | 16.70 m² | −6% | 5 | `loop_closure` |
| `scan_with_ceiling` | **39.01 m²** | 42.71 m² | +9% | 12 | `loop_closure` |
| `scan_floor_only` | **51.33 m²** | 62.24 m² | +21% | 12 | `loop_closure` |

Each arm records the other's number in
`drift_correction.ablation_footprint_area`, so one run gives both. Whether
closure has anything to bind is a property of the walk and is reported as such:
the two `apartment_full` scans return within 0.4 m of their start and fire 12
closures each; `apartment_lidar` ends 3.18 m away and finds 5; the synthetic
fixture is a single turn with no revisit and reports `pose_graph` — ran, found
nothing to close — rather than claiming a closure it did not make.

Levenberg–Marquardt requires residuals ≥ unknowns (4 per edge against 4 per free
node), which a partly-connected property violates; the solver falls back to
trust-region there rather than raising.

## 4. Error budget

Independent relative uncertainties combine **in quadrature**, not by addition
and not by taking whichever is larger.

**LiDAR.** Sensor-metric depth, so there is no scale term. Wall intervals are
±(2 cm + 1% of length); areas ±4%. The ceiling interval is not a placeholder —
it comes from the estimator and differs by method: ±3 cm for a fitted plane,
wider for extrapolation, wider still when the ceiling was never seen.

**Photo/video.** Two independent sources:

1. *Per-wall confidence* — from view count and cross-view plane agreement. A
   wall seen once cannot claim better than the tier's gate; a wall seen from
   every photo with tight agreement can claim better.
2. *Scale factor* — the dominant term, and the honest weakness of these tiers.

Worked example, the `bathroom` room of `saurabh_room_photo`:

```
scale cues:  metric_depth  fired=False   (disabled — see below)
             door_height   fired=False   (disabled)
             ceiling_height fired=True   floor-to-ceiling span 0.709 scale-free
                                         against a 2.44 m ± 0.3 m prior
scale_factor 3.439  ci [2.407, 4.471]  →  scale_rel = ±30.0%
area_rel = √( max(0.08, 2×0.300)² + 0.03² ) = 0.601   →  ±60.1%
```

Area doubles the scale error because area goes as scale². The result:
`floor_area 8.83 m² [3.53, 14.14]`. That interval is close to useless — and
saying so is the point. The brief penalises confident garbage on thin input;
this is the machinery declining to be confident.

**The known weakness.** Only one scale cue fires. ZoeDepth metric depth and
door-height priors are implemented but gated behind `--unsafe-scale-cues`,
off by default, because they load torch into a process that already has open3d.
So scale rests on an assumed 2.44 m ceiling, which is also why every photo/video
plan reports exactly 2.44 m — it is the prior fed back out, not a measurement.
Moving those cues into a subprocess is the highest-value open work on these
tiers; the pattern already exists four times over.

## 5. Calibration analysis

Interval coverage is scored per tier per quantity — gate: ≥90% of 95% intervals
contain the truth.

| Tier | Quantity | Covered | Mean half-width | Result |
|---|---|---|---|---|
| lidar | wall_length | **8/8 (100%)** | ±5.2 cm | PASS |
| lidar | opening_width | **4/4 (100%)** | ±5.0 cm | PASS |
| lidar | ceiling_height | **2/2 (100%)** | ±11.5 cm | PASS |
| lidar | floor_area | **2/2 (100%)** | ±0.40 m² | PASS |
| lidar/photo/video | footprint_area | 40–50% | ±1.0–18.2 m² | FAIL* |

\* Every `footprint_area` failure is a **scorer artifact**: `gate_footprint`
looks up a single global `property/footprint_area` row and scores every capture
against it, so a real 51 m² flat is compared to the fixture's 10.08 m². It
accounts for 9 of the 23 FAILs in the run. It is reported here rather than
quietly fixed, because correcting the scorer before publishing the number would
change the count without changing the pipeline.

Where intervals can be checked against exact truth, **they are correctly
calibrated at 100%**. `cozmo calibrate` fits half-widths per tier per quantity
and persists a versioned file the pipeline reloads at runtime; no calibration
file is committed, because a fit over synthetic data only is not a number worth
shipping, and the pipeline says `Uncalibrated` when the file is absent.

## 6. The fix loop

Declaration committed (`62aff1d`) **before** the code change (`4be839e`), so the
prediction is auditable in `git log`.

**Worst trustworthy gate:** `ceiling_spread`, **62.9 cm against a ≤1 cm gate**,
on the one genuine repeat pair. (`repeatability` fails harder at 718.4 cm, but
wall ids are positional and the two scans produced 12- and 20-wall polygons, so
that gate is comparing walls that are not the same wall — an untrustworthy
number to target.)

**Root cause.** The two captures do not disagree about a measurement; one never
made it. `scan_with_ceiling` measured 3.069 m (`measured_plane`, 40.8% of points
above camera height). `scan_floor_only` fell through to a hardcoded 2.44 m
prior. Its height histogram decays monotonically — 13,484 points in the
1.5–1.6 m band down to 2,669 at 2.0–2.1 m and **nothing above**; the highest
point on any wall is 2.05 m, a full metre below the true ceiling. The
information is absent, not mis-estimated.

**The actual defect** was not the disagreement but the interval:
`2.440 [2.049, 2.990]` — a 95% interval that **did not contain** the 3.069 m
truth, and structurally could not, since its top was `prior + 0.55`.

**The fix.** One branch of one function. The interval now spans from the
observed lower bound to a plausible maximum ceiling, and the estimate is that
band's midpoint — with nothing seen above the bound, every height in the band
fits the data equally, and the midpoint minimises worst-case error.

| | Predicted | Actual |
|---|---|---|
| `scan_floor_only` ceiling | ≈2.72 m | **2.724 m** |
| its interval | contains 3.069 | **[2.049, 3.400]** ✓ |
| `ceiling_spread` | ≈35 cm | **34.5 cm** |
| gate result | **still FAIL** | **still FAIL** |
| Synthetic fixtures | unchanged | unchanged |
| Tests | all pass | all pass |

All seven predictions landed, including the prediction that the gate would not
pass. Across the whole benchmark, **exactly 1 of 94 gate rows changed.**

**Why it fell short, declared in advance:** closing 34.5 cm to 1 cm would mean
inventing a number or copying it from the paired scan. The second would make the
gate pass while making the pipeline worse. The real remedy is a capture protocol
that tells the operator to sweep upward — a Part 1 deliverable, not a code
change.

## 7. Known failure modes

Measured from the Stray Scanner **confidence channel** — the sensor's own record
of where it failed. Evidence figure:
`docs/evidence/confidence_failure_modes.jpg`.

| Mode | Worst frame | Effect | Our behaviour |
|---|---|---|---|
| **Mirror** | 37% of frame zero-confidence | Depth measures the reflected path | Gated out at ingest; wall marked under-observed. Never becomes phantom geometry |
| **Glass** | 36% | Little or no return | Same; openings not claimed in under-observed walls |
| **Dark gloss** | **44%** — the worst in the set | Dark + specular is the worst case for IR ToF | Same |
| **Wet-look / polished** | 28% | Sparse *floor* returns | Weakens the evidence the room polygon is built from |

The bias is deliberately toward false negatives: **we would rather miss a glass
door than invent one.** That is why all three real LiDAR captures report zero
geometric openings despite plainly having doors, and why the plan says
*"absence of openings in them is not evidence they are solid"* rather than
staying silent.

**Low light contradicts the expectation.** Correlation between frame luminance
and zero-confidence depth is **−0.08 to −0.12**; the darkest quartile is only
1.2–2.2× worse than the brightest. The LiDAR is an *active* sensor, so ambient
light is nearly irrelevant to it. Low light instead degrades the **passive**
paths — VGGT and the RGB detector — where it is real but unquantified: on the
real walkthrough the blur filter discarded **47 of 74 sampled frames**.

### What is not there

- **No room segmentation at the LiDAR tier.** One room per capture, so a
  whole-flat walk becomes one arbitrary polygon. This is a contract gap: the
  brief's stitched plan is produced at photo and video but **not at LiDAR**, and
  it is the root cause behind the 27% footprint disagreement between the two
  scans of one flat.
- **No mirror or glass *detection*.** We degrade gracefully because the
  confidence channel happens to mark these surfaces, not because anything
  recognises them. The photo/video tiers have no equivalent signal at all.
- **No laser or tape ground truth on any real capture.** 45 of 94 gate rows
  SKIP, and SKIP is never a pass. This is the single largest gap in the
  submission and the reason the device matrix is mostly empty.
- **No head-to-head** against a consumer scanning app.
- **Damage/scope at photo and video.** Openings are detected there; damage,
  concealed flags and scope run only at the LiDAR tier. Detector precision is
  unvalidated — the benchmark has no damage ground truth.
