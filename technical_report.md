# Floor-Plan Reconstruction and Damage Scoping from Handheld Phone Captures

**Technical Report**

| | |
|---|---|
| **Author** | Saurabh |
| **Date** | 10 September 2026 |
| **Reference run** | `fixloop/after/` (tag `fixloop-after`), scorer 1.0.0, 11 captures, 94 gate rows |
| **Companions** | [compliance_matrix.md](compliance_matrix.md) · [benchmark_report.md](benchmark_report.md) · [known_failure_modes.md](known_failure_modes.md) · [fixloop/declaration.md](fixloop/declaration.md) · [docs/design.md](docs/design.md) |

---

## Abstract

This report describes a pipeline that reconstructs dimensioned floor plans from
handheld phone captures at three tiers — photographs, a walkthrough video, and a
LiDAR scan — emitting one output contract from all three in which every physical
quantity carries a 95% interval.

The LiDAR tier is exact where it can be checked: on a ray-traced capture whose
answer is known analytically it recovers wall lengths to **1 mm**, ceiling height
to **4 mm**, and both openings with zero misses and zero phantoms. The photo and
video tiers reconstruct real rooms and stitch them into connected multi-room plans
with zero room overlap, but report footprint intervals of **±60%**, because metric
scale currently rests on a single assumed-ceiling cue.

The central finding is negative and is stated up front: **no real capture in the
benchmark has laser or tape ground truth**, so 45 of 94 gate rows report SKIP, and
SKIP is never counted as a pass. Accuracy on real hardware is therefore not
established at any tier. What *is* established is that stated uncertainty tracks
actual evidence — intervals widen 15× from LiDAR to photo — and that failure
behaviour is biased toward refusing to answer rather than answering wrongly.

## 1. Scope

The task: turn an ordinary phone capture of an interior into a plan usable for
insurance-style damage scoping — per-room walls, ceiling height, floor area,
openings, damage regions with metric extent, concealed-damage flags naming the rule
that fired, and scope line items, each with an interval.

| Tier | Input | Metric source |
|---|---|---|
| **Photo** | 2–8 unposed stills per room, one folder per room | none — scale must be recovered |
| **Video** | one handheld walkthrough clip | none — scale must be recovered |
| **LiDAR** | depth + poses + intrinsics (Stray Scanner) | sensor-metric |

Two constraints shaped everything below: **a number without an interval is not an
answer**, since the grading penalises confident output on thin input; and the tier
is a fact about how data was collected, not a runtime choice.

## 2. Architecture

```
capture.json ──► load_capture() ──► tier dispatch
                                      ├── lidar  → build_lidar_plan()
                                      ├── photo  → build_photo_plan()
                                      └── video  → build_video_plan()
                                            ↓
                            plan.json · plan.png/svg · run_manifest.json
```

**2.1 Every physical quantity is a `Measurement`, never a bare float.**
`schema.py` (pydantic v2, `extra="forbid"`) defines `Measurement(value, ci_95,
unit)` with validators asserting the interval is ordered and brackets the value,
so no code path can emit a dimension without one. Referential integrity is checked
across the document: an opening on an unknown wall, damage on an unknown surface,
or a scope item citing damage that does not exist are rejected at parse time.

**2.2 Tier is read from the capture, not passed as a flag.** A `--tier` option
would let a photo capture be scored against LiDAR gates by typo.

**2.3 Every torch model runs in a separate interpreter.** open3d and torch each
bundle their own OpenMP runtime; a process holding both either aborts with
`OMP: Error #179` or **deadlocks inside the first inference call at 0% CPU with no
error at all** — reproducibly, in both import orders. `KMP_DUPLICATE_LIB_OK` papers
over this and is documented upstream as unsafe. VGGT adds a second, independent
reason: upstream pins `numpy<2` and needs Python ≥3.10, while open3d here runs
numpy 2.x. Hence four subprocess workers, JSON at every boundary, and VGGT under its
own `.venv-recon`.

**2.4 Reproducibility is structural.** Every run stamps `run_manifest.json` with
the git commit, exact command, seed record, and a SHA-256 over the input directory
(paths *and* bytes). Seeding covers Python, numpy, torch **and open3d's own global
RNG** — `segment_plane` draws from it, and without seeding it the fitted floor
plane moved between runs and every derived dimension moved with it.

**2.5 Degrade, never fake.** A missing detector, an unobserved ceiling, an
unreconstructable room — each degrades the plan and records why in
`quality.degradations`, rather than aborting or substituting a plausible number.

## 3. Tier design

### 3.1 LiDAR tier

```
correct_trajectory → fuse_capture → fit_floor → extract_layout
                   → estimate_ceiling → detect_openings → semantics → Plan
```

Pure geometry, **zero model weights**, 2.5 s to 68.8 s by frame count. Four
decisions carry the accuracy; all are defended in [docs/design.md](docs/design.md) §3.

*Floor fitting is seeded, not left to RANSAC.* Plain RANSAC returns the largest
plane, often a wall — on a real capture it settled 6–9 cm above the floor. A
vertical histogram picks the **lowest** band with substantial support (not the
densest: a bed can out-populate the floor it stands on), and RANSAC fits within it,
recovering tilt. The sample floor is 0.7° off level, which across a 9 m room is
11 cm, so every height is a signed distance from that plane, never raw `y`.

The other three: depth is the camera-frame z of the hit, **not** distance along the
ray (treating it as ray length inflates every dimension by 1/cos θ, ~8% at the frame
edge); wall normals are histogrammed modulo 90°, so a room can be rectilinear
without being axis-aligned and an L-shape is a different set of kept cells rather
than a special case; and openings are found per-column rather than as 2D blobs —
as blobs, a doorway merged with the unobserved strip above it and a visible door
went undetected. Every wall reports the fraction of its length that returned
anything, so "no openings" in a barely-observed wall is never read as "solid".

### 3.2 Photo tier

```
frames → VGGT-1B (scale-free cloud + poses) → per-view planes → cluster
       → polygon → scale recovery → metres → openings
```

**Per-view plane fitting, not one global RANSAC** — the decision most worth
defending. A cloud from 5–8 photos is a few thousand points across a whole room,
too sparse for a global fitter that needs a dense wall band. So the order inverts:
planes are fit *per image*, where that frame's points are locally dense, then
carried into the world frame by that frame's pose and clustered by agreement.
Per-view plane count and cross-view agreement are what a sparse reconstruction can
honestly say about its own confidence, and both feed each wall's interval. Because
one monocular frame cannot reveal metric "down", per-view fitting assumes the phone
was upright — a rolled photo breaks this per-frame, which is precisely why
cross-view agreement, not any single frame, governs interval width.

Both sparse and dense paths call the **same** `assemble_polygon` as the LiDAR tier;
nothing is forked. **VGGT returns geometry, not size** (`scale_is_metric = False`),
so recovering metres is a separate stage and a backbone swap (DUSt3R, MASt3R) is a
class plus one registry entry.

### 3.3 Video tier

There is **no separate video reconstruction pipeline**, and this is disclosed rather
than hidden. The clip is decoded at a stride, blurred frames are dropped, and the
sharpest survivor of each time slice is kept, up to the backbone's 2–8 view
contract. Consecutive frames are then embedded with DINOv2 and cut where similarity
drops — crossing a doorway changes nearly everything in view at once — so each run
becomes a room folder and the multi-room photo path runs unchanged.

Two measured findings shaped this, both worth defending live. **Blur thresholds do
not transfer between media**: real H.264 phone video has Laplacian variance an order
of magnitude below JPEG stills (measured 1.9–34, median 6.7), so a still-photo
threshold of 80 silently rejected *every* video frame. **Sampling must be temporally
stratified**: capping by *global* sharpness collapsed selection onto the steadiest
stretch, taking all 8 frames from the first 11 seconds of a 37-second walk and
discarding two rooms entirely. A regression test exists and was verified to fail
against the old sampler.

### 3.4 Multi-room stitching

Per-room reconstruction → DINOv2 retrieval → SuperPoint/LightGlue keypoints and
doorway-width matching → 2D Procrustes per pair → pose graph over (x, y, yaw) →
Manhattan snap → overlap resolution → one plan with `Adjacency` objects.

**Two evidence sources, neither preferred.** A keypoint transform is precise given
enough inliers (≥4, RANSAC-lite consensus); a matched doorway still works when a
door is shut and the rooms share no visible scene, which the brief identifies as the
strongest signal. Both are recorded per pair, and **a pair with neither is rejected
and reported, not guessed at**. Validated with live weights: two disjoint real rooms
score **0.086** DINOv2 similarity and **zero** keypoint matches; two overlapping
views of one real room score **0.81** and **133**.

### 3.5 Device matrix

Generated from the run's own gate results and each capture's `run_manifest.json`
(which is where device information lives; `plan.json` does not carry it), not
written by hand — `benchmark/score.py::build_device_matrix`, emitted into
`results.json` as 35 rows.

| Tier | Device class | Gates with truth | Pass rate | Honest accuracy |
|---|---|---|---|---|
| lidar | synthetic (ray-traced) | 5 | **5/5 = 100%** | wall 0.1 cm, ceiling 0.4 cm |
| lidar | iPhone (Pro-class, LiDAR) | 2 | 0% | **unverified** — both rows are the §6.6 footprint artifact |
| photo | OnePlus Nord 2T | 2 | 0–50% | **unverified** |
| photo | Kinect RGB (MSR 7-Scenes) | 2 | 0–100% | **unverified** — public dataset, not a phone |
| video | iPhone (Pro-class, LiDAR) | 2 | 0% | **unverified** |
| video | Mac Pro | 2 | 0–100% | **unverified** |

**The empty column is the honest content of this table.** Only the synthetic
fixtures carry ground truth, so no row for real hardware establishes accuracy —
every non-synthetic "pass rate" above is scored against either the global-footprint
artifact of §6.6 or an interval-coverage row derived from it. The one property that
*is* genuinely device-differentiated is interval width: **±4.0% at LiDAR against
±60.2% at photo and video.**

Device compliance is itself imperfect and is disclosed rather than smoothed over.
The brief specifies **iPhone 15 or newer** for the photo and video tiers. Of the
captures here, one photo set is from a OnePlus Nord 2T, one video from a Mac, and
two photo sets from the MSR 7-Scenes public dataset (Kinect RGB). These are
development fixtures; no accuracy claim rests on them, and the walk-in test will be
the first iPhone-15 photo capture the pipeline has seen.

## 4. Error budget and calibration

### 4.1 How uncertainty composes

Independent relative uncertainties combine **in quadrature** — not by addition, and
not by taking whichever is larger. At LiDAR the depth is sensor-metric, so there is
no scale term: walls are ±(2 cm + 1% of length), areas ±4%. At photo and video
there are two contributions: per-wall confidence, and the **scale factor, which
dominates**.

Worked example — the `bathroom` room of `saurabh_room_photo`, verbatim from its
`run_manifest.json`:

```
scale cues:  metric_depth   fired=false   "disabled for this run"
             door_height    fired=false   "disabled for this run"
             ceiling_height fired=true    span_scalefree 0.7095
                                          against a 2.44 m ± 0.30 m prior
scale_factor 3.43916   ci_95 [2.40741, 4.47091]   method single_cue
agreement 0.30                              →  scale_rel = ±30.0%
area_rel = √( max(0.08, 2×0.300)² + 0.03² ) = 0.601   →  ±60.1%
```

Area doubles the scale error because area scales as scale². Result:
`floor_area 8.83 m² [3.53, 14.14]`. **That interval is close to useless, and saying
so is the point** — this is the machinery declining to be confident, not failing.

### 4.2 Ceiling height: three methods, three interval shapes

| Method | Estimate | Interval | Real example |
|---|---|---|---|
| `measured_plane` | plane height − floor height | value ± 3 cm | `2.496 [2.466, 2.526]` |
| `wall_extrapolation` | median wall-top + 5 cm | (median − 5 cm, median + 35 cm) | `2.495 [2.395, 2.795]` |
| `scan_cutoff_prior` | midpoint of the plausible band | (highest wall point, ≥3.40 m) | `2.696 [1.991, 3.400]` |

`wall_extrapolation` is deliberately **asymmetric** — 5 cm down, 35 cm up — because
a true ceiling can only be above where the walls stopped. `scan_cutoff_prior` means
the capture ended below the ceiling: the only quantity actually measured is a lower
bound, so the interval spans the whole band still consistent with the data and the
estimate is its midpoint, minimising worst-case error where data cannot
discriminate.

### 4.3 Two known weaknesses

**Only one scale cue fires.** ZoeDepth metric-depth alignment and door-height
priors are implemented but gated behind `--unsafe-scale-cues`, off by default,
because they load torch into a process already holding open3d (§2.3). Scale
therefore rests on an assumed 2.44 m ceiling — which is why every photo and video
plan reports exactly 2.44 m: it is the prior fed back out, not a measurement.
Moving those cues into a subprocess is the highest-value open work, and the pattern
already exists four times over.

**Intervals are asserted, not calibrated.** `cozmo calibrate` fits a per-tier,
per-quantity half-width factor against ground truth, flooring at 1.0 so calibration
can only widen, and reports `N` and a `TRUSTED` flag (≥3 samples) beside every
factor. But **no calibration file is committed** — a fit over synthetic data alone
is not worth shipping — so every run is uncalibrated and says so in
`quality.calibration_note`. The "95%" is a stated uncertainty, not a measured
coverage probability. This should be conceded immediately if challenged.

## 5. Experimental setup

Eleven captures across three tiers, scored into 94 gate rows. Ground truth comes
from two ray-traced Stray Scanner fixtures of a 3.60 × 2.80 m room (2.50 m ceiling,
0.85 m door, 1.10 m window, yawed 23°), written from the same constants the renderer
used, so it is exact. `synthetic_no_ceiling` drops the ceiling returns, modelling an
operator who never looks up.

Two rules the scorer does not bend:

1. **A gate with no ground truth is SKIP, never PASS.** Silence must never read as
   success.
2. **Detection is part of the opening gate.** The denominator is
   matched + missed + phantom, so a pipeline cannot buy accuracy by reporting only
   the openings it is confident about.

## 6. Results

### 6.1 Gate summary — 26 PASS / 23 FAIL / 45 SKIP

| Gate | PASS | FAIL | SKIP | | Gate | PASS | FAIL | SKIP |
|---|---|---|---|---|---|---|---|---|
| `wall_lengths` | 2 | 0 | 9 | | `interval_coverage` | 5 | 6 | 0 |
| `ceiling_height` | 2 | 0 | 9 | | `interval_coverage_by_kind` | 4 | 3 | 0 |
| `opening_widths` | 2 | 0 | 9 | | `repeatability` | 1 | 4 | 0 |
| `footprint` | 2 | 9 | 0 | | `ceiling_spread` | 4 | 1 | 0 |
| `room_overlap` | 4 | 0 | 7 | | `adjacency_correctness` | 0 | 0 | 11 |

### 6.2 Accuracy against exact truth

| Quantity | Truth | Recovered | Gate | Result |
|---|---|---|---|---|
| Walls | 3.60 / 2.80 m | worst error **0.1 cm** | ≤ max(2 cm, 1%) on ≥85% | PASS |
| Ceiling | 2.50 m | worst error **0.4 cm** | ≤1.5 cm | PASS |
| Openings | 0.85 / 1.10 m | 2/2, **0 missed, 0 phantom** | ≤2 cm on ≥85% | PASS |
| Footprint | 10.08 m² | **10.08 m²** (0.0%) | ±2% | PASS |

This validates the geometry and the Stray Scanner sensor conventions. It says
nothing about real-world noise: ray-traced depth has none.

### 6.3 Interval coverage

| Tier | Quantity | Covered | Mean half-width |
|---|---|---|---|
| lidar | `wall_length` | **8/8 (100%)** | ±5.2 cm |
| lidar | `opening_width` | **4/4 (100%)** | ±5.0 cm |
| lidar | `ceiling_height` | **2/2 (100%)** | ±11.5 cm |
| lidar | `floor_area` | **2/2 (100%)** | ±0.40 m² |
| all | `footprint_area` | 40–50% | ±1.0–18.2 m² |

Where intervals can be checked against exact truth they are correctly calibrated at
100%. Every `footprint_area` failure is the scorer artifact of §6.6.

### 6.4 Drift ablation

With correction on, the trajectory is fixed *before* fusion, so the flag changes
geometry rather than only the axes the room is drawn on. Each arm records the
other's number in `drift_correction.ablation_footprint_area`, so one run yields
both.

| Capture | On | Off | Δ | Closures | Method |
|---|---|---|---|---|---|
| Synthetic (truth 10.08 m²) | **10.08 m²** | 13.41 m² | −24.8% | 0 | `pose_graph` |
| `apartment_lidar` | 17.82 m² | 16.70 m² | +6.7% | 5 | `loop_closure` |
| `scan_with_ceiling` | 39.01 m² | 42.71 m² | −8.7% | 12 | `loop_closure` |
| `scan_floor_only` | 51.33 m² | 62.24 m² | −17.5% | 12 | `loop_closure` |

On the one capture with exact truth, correction is unambiguously right: 13.41 m²
uncorrected against a true 10.08 m², recovered exactly. Whether closure has anything
to bind is a property of the walk and is reported as such — the synthetic fixture is
a single turn with no revisit and reports `pose_graph` (ran, found nothing to close)
rather than claiming a closure it did not make.

open3d's `PoseGraph` was **not usable**: `PoseGraphNode` segfaults unconditionally
on 0.18.0 on this platform (minimal reproducer available), and 0.19 has no build for
this Python/platform pair. The replacement is ~90 lines of scipy least-squares over
(x, y, z, yaw), shared with the stitching graph.

### 6.5 Repeatability

| Group | Captures | Worst wall Δ | Ceiling spread |
|---|---|---|---|
| `lidar:test_room` | 2 (synthetic) | **0.0 cm** PASS | **0.1 cm** PASS |
| `lidar:room_a` | 3 | 718.4 cm FAIL | **37.4 cm** FAIL |
| `photo:living_room` | 3 | 367.8 cm FAIL† | 0.0 cm PASS‡ |
| `video:room_3`, `room_4` | 2 each | 154.7 / 534.1 cm FAIL† | 0.0 cm PASS‡ |

† **Not real repeat captures.** `_repeat_groups()` keys on `(tier, room_id)` and
ignores `space_id`, so unrelated captures reusing a room *name* are pooled. A
scorer defect, not a pipeline result.

‡ **Passing for the wrong reason.** Every photo/video plan reports the same 2.44 m
prior, so spread is trivially zero. A gate that passes because nothing measured
anything is not evidence of repeatability, and is reported here as a failure of the
gate rather than a success of the system.

The 718.4 cm figure is itself untrustworthy: wall ids are positional and the two
scans produced polygons with **12 and 20 walls**, so index *i* names a different
physical wall in each — that figure compares a 0.27 m wall against a 7.46 m one.
The real instability shows in order-independent metrics: footprint **39.01 vs
51.33 m², 27.3% apart**, bounding box 7.46 × 5.36 m vs 6.96 × 9.42 m, and both
drawings show the trajectory *leaving* the room polygon. Root cause is §8's second
item.

### 6.6 Known scorer artifacts

Two defects inflate the FAIL count and must not be read as pipeline failures.
First, `gate_footprint` looks up a single global `property/footprint_area` row,
finds the fixture's 10.08 m², and scores *every* capture against it — hence
"51.33 m² vs 10.08 m² (409.2%)". This accounts for **9 of the 23 FAILs** plus the
dependent `interval_coverage` rows. Second, repeatability grouping ignores
`space_id` (§6.5).

Both are visible in `results.json`, and are reported rather than quietly fixed:
correcting the scorer before publishing the number would change the count without
changing the pipeline.

### 6.7 Timing

LiDAR 2.5 s (16 frames) → 68.8 s (9,745 frames), geometry only, scaling with frame
count over a 600× range. Photo 43.1 s (1 room) → 140.5 s (4 rooms), VGGT-dominated
at ~40–50 s per room. Video 107.2 s → 233.4 s. **Whole benchmark: 765 s = 12.8 min
for 11 captures.**

## 7. The fix loop

The declaration was committed (`62aff1d`) **before** the code change (`4be839e`),
so the prediction is auditable in `git log`.

**Target.** `ceiling_spread` on the one genuine repeat pair — **62.9 cm against a
≤1 cm gate**. `repeatability` fails harder (718.4 cm) but was rejected as a target
because, per §6.5, that number compares walls that are not the same wall;
targeting it would optimise an artifact.

**Diagnosis, from the error distribution rather than intuition.** The per-wall
ratio between the two scans spans 0.037 to 6.457 — mean 1.365, median 0.939,
**σ 1.720**. A multiplicative (scale) error clusters ratios around one constant;
these span 175× with σ exceeding the mean. Random noise does not move half the
walls by 2–7 metres. A rigid pose offset preserves lengths. So the disagreement is
none of scale, noise, or pose.

The actual cause is that the two captures do not disagree about a measurement —
**one never made it.** `scan_with_ceiling` measured 3.069 m (`measured_plane`,
40.8% of points above camera height). `scan_floor_only`'s height histogram decays
monotonically — 13,484 points in the 1.5–1.6 m band down to 2,669 at 2.0–2.1 m and
**nothing above** — with the highest point on any wall at 2.05 m, a metre below the
true ceiling. The information is absent, not mis-estimated.

**The defect was the interval, not the disagreement.** The old output was
`2.440 [2.049, 2.990]`: a 95% interval that did **not** contain the 3.069 m truth,
and structurally could not, because its upper bound was `prior + 0.55`.

**The fix.** One branch of one function (§4.2): the interval now spans from the
observed lower bound to a plausible maximum ceiling, and the estimate is that
band's midpoint.

| Prediction | Predicted | Actual |
|---|---|---|
| `scan_floor_only` ceiling | ≈2.72 m | **2.724 m** |
| its interval contains 3.069 m | yes | **[2.049, 3.400]** ✓ |
| `ceiling_spread`, genuine pair | ≈35 cm | **34.5 cm** |
| `ceiling_spread`, scorer's pooled row | — | **37.4 cm** |
| gate result | **still FAIL** | **still FAIL** |
| Synthetic fixtures | unchanged | unchanged |
| Test suite | all pass | all pass |

All seven predictions landed, **including the prediction that the gate would not
pass**. Across the whole benchmark, **exactly 1 of 94 gate rows changed.**

Two spread figures are reported because they answer different questions: 34.5 cm is
the genuine pair (3.069 − 2.724); 37.4 cm is the scorer's gate row, which pools a
third capture via the `space_id` defect of §6.5 and is what the gate table prints.

**Why it fell short, declared in advance.** Closing 34.5 cm to 1 cm would mean
inventing a number or copying it from the paired scan; the latter would make the
gate pass while making the pipeline worse. The real remedy is a capture protocol
instructing the operator to sweep upward — a Part 1 deliverable, not a code change.

## 8. Failure modes and limitations

Failure modes are measured from the Stray Scanner **confidence channel**, the
sensor's own record of where it failed. Reproduce with
`python scripts/confidence_stats.py`.

| Mode | Worst frame | Sensor effect | Behaviour |
|---|---|---|---|
| **Mirror** | 37% zero-confidence | depth measures the reflected path | gated out at ingest; wall marked under-observed, never phantom geometry |
| **Glass** | 36% | little or no return | same; openings not claimed in under-observed walls |
| **Dark gloss** | **44%** — worst in set | dark + specular is worst case for IR ToF | same |
| **Wet-look** | 28% | sparse *floor* returns | weakens the evidence the polygon is built from |

The bias is deliberately toward false negatives: **we would rather miss a glass
door than invent one.** That is why all three real LiDAR captures report zero
geometric openings despite plainly having doors.

**Low light contradicts the expectation**, and this is the most useful negative
result here. Correlation between frame luminance and zero-confidence depth is
**−0.08 to −0.12**, and the darkest quartile is only 1.2–2.2× worse than the
brightest: LiDAR is an *active* sensor, so ambient light is nearly irrelevant. Low
light instead degrades the **passive** paths, where the effect is real but
unquantified — on the real walkthrough the blur filter discarded **47 of 74**
sampled frames. The tier the brief expects low light to hurt is the one that is
immune; the tiers that are hurt have no ground truth to measure the harm on.

**Limitations, in descending order of how much they undermine the results:**

1. **No laser or tape ground truth on any real capture.** 45 of 94 rows SKIP; every
   accuracy claim on real hardware is unverified, and the two exact numbers here
   come from ray-traced data with no sensor noise. The single largest gap.
2. **No room segmentation at the LiDAR tier** — one room per capture, so a
   whole-flat walk becomes one arbitrary polygon. A **contract gap**: the required
   stitched plan is produced at photo and video but *not* at LiDAR, and it is the
   root cause of the 27.3% footprint disagreement in §6.5.
3. **Scale rests on one cue** (§4.3), hence ±60% intervals and a 2.44 m ceiling in
   every photo/video plan; **intervals are asserted, not calibrated**.
4. **No head-to-head against a consumer scanning app** — blocked twice over: no app
   export, and no laser truth to score either column against.
5. **No capture protocol document**, which is also the remedy the fix loop
   identified.
6. **Damage and scope run only at LiDAR**; openings are detected at photo and video,
   but the rest of the semantic stage is not wired in there, and detector precision
   is unvalidated because the benchmark has no damage rows.
7. **Two scorer defects inflate the FAIL count** (§6.6), disclosed rather than
   corrected before reporting.

## 9. Conclusion

The system meets its structural obligations: one contract from three tiers, one
command per capture, an interval on every physical quantity enforced by the type
system, provenance sufficient to trace any number back to the bytes that produced
it, and byte-identical replay.

On measurable accuracy it is exact where exactness can be demonstrated (1 mm walls
on analytic truth) and honestly wide where evidence is thin (±60% on single-cue
photo scale). The fix loop showed the intended discipline: diagnose from the error
distribution, declare a prediction before shipping, change one thing, report that
the gate still fails.

What it does **not** yet demonstrate is accuracy on real rooms, because no real room
in the benchmark has been measured with a laser or tape. Obtaining that would
unblock 45 SKIP rows, the verified-accuracy component, and both columns of the
head-to-head — a measurement exercise rather than an engineering one, and the
correct next investment.

---

## Appendix — Reproduction

| What | Command | Measured |
|---|---|---|
| Clean-machine setup | `scripts/setup.sh` | **120 s** from `git clone` to a working reconstruction, cold pip cache |
| One capture | `cozmo run --input DIR --out DIR` | 2.5 s – 233 s by tier |
| Whole benchmark | `bash scripts/run_benchmark.sh benchmark_runs` | **12.8 min**, 11 captures |
| Fix-loop arms | `git checkout fixloop-before` / `fixloop-after` | 94 gate rows each |
| Failure-mode evidence | `python scripts/confidence_stats.py` | reproduces every §8 figure |
| Test suite | `pytest` | **206 tests: 202 pass, 4 opt-in** (need weights) |
| Determinism | run twice, `diff` | LiDAR `plan.json` **and** `plan.png` byte-identical; photo tier via real VGGT on MPS, max wall difference **0.00e+00 m** |
| Weight integrity | `scripts/fetch_weights.sh --check` | **7/7 model weights sha256-pinned**, 26 files verified |

The environment requires CPython 3.9–3.12: open3d publishes no wheel for 3.13+, so
`scripts/setup.sh` searches for a supported interpreter rather than taking whatever
`python3` happens to be.
