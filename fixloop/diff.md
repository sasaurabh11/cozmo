# Fix loop: before → after

Declaration (written first, committed first): [declaration.md](declaration.md)

This fix loop shipped in two stages against the same declared gate
(`ceiling_spread`). Stage 1 is the originally declared fix. Stage 2 is a
follow-on correctness pass on the scorer itself, found while verifying stage
1's own numbers, which changes what the rest of the benchmark reports too.
Both are real, both are shown below with their own before/after.

| Stage | Command | Tag / dir |
|---|---|---|
| 1 (declared fix) | `bash scripts/run_benchmark.sh fixloop/before` / `fixloop/after` | `fixloop-before` / `fixloop-after` |
| 2 (scorer correctness) | `bash scripts/run_benchmark.sh fixloop/round1_before` / `fixloop/round1_after` | `fixloop/round1_before` / `fixloop/round1_after` |

All four regenerate byte-comparably — `SOURCE_DATE_EPOCH` is pinned inside the
script, so `diff` is a valid check, not just inspection.

---

## Stage 1 — the declared fix

**One function, one branch.** `cozmo/geometry/planes.py::estimate_ceiling`, the
`scan_cutoff_prior` path — the fallback taken when no ceiling plane fits and the
wall tops disagree too much to extrapolate.

```diff
+CEILING_PLAUSIBLE_MAX_M = 3.40

     lower = max(highest, CEILING_MIN_ABOVE_FLOOR_M * 0.9)
-    upper = max(prior_height_m + 0.55, lower + 0.35)
-    height = min(max(prior_height_m, lower + 0.05), upper)
+    upper = max(CEILING_PLAUSIBLE_MAX_M, lower + 0.35)
+    height = 0.5 * (lower + upper)
```

Two lines of behaviour:

1. **The interval's top no longer hangs off the 2.44 m prior.** It was
   `prior + 0.55 = 2.99 m`, which cannot contain an ordinary 3.0 m ceiling. It
   is now a plausible tall-ceiling limit.
2. **The estimate is no longer the prior.** With nothing observed above `lower`,
   every height in the band is equally consistent with the data, so the
   midpoint is reported — the choice that minimises worst-case error. The prior
   never had any claim on this particular building.

The `measured_plane` and `wall_extrapolation` paths are untouched, which is why
the synthetic fixtures do not move.

### The declared gate

| | Before | After | Gate |
|---|---|---|---|
| `ceiling_spread`, genuine pair (`apartment_full`) | **62.9 cm** | **34.5 cm** | ≤1 cm |
| Result | FAIL | **FAIL** (predicted) | |

**45% of the error removed.** Still failing, exactly as the declaration
predicted and for the reason it gave.

### Prediction vs outcome

| Metric | Predicted | Actual | |
|---|---|---|---|
| `scan_floor_only` ceiling | ≈2.72 m | **2.724 m** | ✅ |
| `scan_floor_only` interval | ≈[2.05, 3.40], contains 3.069 | **[2.049, 3.400]**, contains 3.069 | ✅ |
| `ceiling_spread` (pair) | ≈35 cm | **34.5 cm** | ✅ |
| `ceiling_spread` gate | still FAIL | **still FAIL** | ✅ |
| `apartment_lidar` ceiling | ≈2.69 m | **2.696 m** | ✅ |
| Synthetic fixtures | unchanged | **unchanged** (2.496 / 2.495) | ✅ |
| Test suite | 202 pass | **202 pass** | ✅ |

Every prediction landed, including the prediction that the gate would not pass.

### The repair that matters more than the gate

The gate measures agreement between two captures. The actual defect was an
interval that excluded reality:

| | Before | After |
|---|---|---|
| `scan_floor_only` ceiling | 2.440 m, ci `[2.049, 2.990]` | 2.724 m, ci `[2.049, 3.400]` |
| Truth (paired scan, `measured_plane`) | 3.069 m | 3.069 m |
| Truth inside the 95% interval? | **No** — upper bound 2.99 < 3.069 | **Yes** |

Before the fix the plan asserted 95% confidence in a range that did not contain
the answer. That is a calibration failure regardless of what any gate says, and
it is now fixed. The same applies to `apartment_lidar`, whose interval also now
reaches 3.40 m.

### Stage 1's own whole-benchmark diff was one row

At the time this specific fix was shipped, the rest of the benchmark did not
move:

```
before   26 PASS / 23 FAIL / 45 SKIP
after    26 PASS / 23 FAIL / 45 SKIP

gate rows that changed: 1
  ceiling_spread / lidar:room_a   FAIL both sides, number improved
```

The change was confined to a single branch of a single function, and this
specific diff was attributable to it and nothing else. (At the time, this row
was reported as "62.9 cm / 37.4 cm across 3 captures" — that 3rd capture was a
scorer grouping bug, described next, not a real second repeat of `room_a`.)

---

## Stage 2 — the scorer correctness pass

Found while verifying stage 1's own repeatability/ceiling_spread numbers: the
scorer itself had three bugs unrelated to reconstruction accuracy, all in
`cozmo/benchmark/score.py`.

**1. Footprint ground truth applied globally.** `gate_footprint` and
`covered_measurements` both matched a ground-truth row by the literal
`element_id="property"`, with no per-capture scoping — so the *one* synthetic
fixture's 10.08 m² footprint was being compared against every real capture in
the benchmark, real properties included. Fixed by checking for a
capture-specific truth row first, and only falling back to a generic property
row when the ground-truth file actually identifies something else in that same
plan (`_has_plan_truth`).

**2. Repeat captures grouped by room *name*, not physical space.** `_repeat_groups`
keyed purely on `(tier, room.id)` — three unrelated captures that each happened
to name a room `living_room` were treated as three repeats of the same room.
Fixed by grouping on the declared `space_id` (from `run_manifest.json`) as well.
This is also what removed the `apartment_lidar` capture from the `room_a`
group in stage 1's own `ceiling_spread` row — it was never really a repeat.

**3. Wall ids compared by position label, not identity.** `gate_repeatability`
matched walls between two captures by `wall_id` string equality
(`f"{room_id}_w{index}"`, assigned by fitting order, not physical identity).
Fixed with `scipy.optimize.linear_sum_assignment` — true minimum-cost matching
by wall length — with unmatched walls (a scan that split or merged a boundary
differently) now explicitly reported as failures instead of silently ignored.

**Plus one new gate:** `gate_drift_accountability` — checks a named
drift-correction method (never `poses_as_is`) and a real on/off footprint
ablation exist for every LiDAR and multi-room capture. The photo/video stitch
path (`cozmo/stitch/graph.py`, `cozmo/pipeline/photo.py`) gained a genuine
`drift_correction=False` arm to produce that ablation — previously only LiDAR
had one.

### The whole-benchmark diff, stage 2

```
before (fixloop/round1_before)   26 PASS / 23 FAIL / 45 SKIP
after  (fixloop/round1_after)    30 PASS /  2 FAIL / 65 SKIP
```

Verified by independent regeneration, not asserted. What actually moved:

| Change | Count | What it means |
|---|---|---|
| `footprint` FAIL → SKIP | 9 rows (every real capture) | No longer comparing real properties against an unrelated fixture's area. Honest "no ground truth", not a pass. |
| `interval_coverage` FAIL/PASS → SKIP | 9 rows | Same reason — their only checkable measurement was the invalid footprint comparison, including 3 rows that were never a real pass either. |
| `interval_coverage_by_kind: lidar:footprint_area` FAIL → PASS | 1 row | Genuine: only the 2 valid synthetic comparisons remain, and they were always passing underneath the noise. |
| `repeatability` / `ceiling_spread` groups that were never real repeats | 3 groups removed | `photo:living_room`, `video:room_3`, `video:room_4` no longer appear at all — they were room-name coincidences, not the same physical room scanned twice. |
| `drift_accountability` added | 11 rows (9 PASS, 2 SKIP) | New gate; PASS everywhere a named method + real ablation exist, SKIP only for genuinely single-room photo captures where the gate does not apply. |
| `repeatability: lidar:apartment_full:room_a` | still FAIL | 718.4 cm → **158.0 cm** (12 matched pairs, 8 unmatched) — better matching, same real disagreement. |
| `ceiling_spread: lidar:apartment_full:room_a` | still FAIL | 62.9 cm → **34.5 cm**, now reported directly (see stage 1's note above) — this is the one row stage 1 actually targeted. |

**The 2 remaining FAILs in the entire benchmark, both `apartment_full:room_a`:**

```
ceiling_spread   lidar:apartment_full:room_a   spread 34.5 cm across 2 captures     FAIL  (<= 1 cm)
repeatability    lidar:apartment_full:room_a   worst 158.0 cm, 12 matched, 8 unmatched  FAIL  (<= 1 cm or 0.5%)
```

Both are the same flat, same known cause: `scan_floor_only` never captured the
ceiling (operator kept the phone low the whole walk), so it disagrees with
`scan_with_ceiling` on both ceiling height and, downstream, on wall geometry
near the ceiling. This is not 23 separate reconstruction bugs cleared — it is
the scorer no longer manufacturing 21 false or coincidental readings around
one already-understood, already-declared capture defect.

---

## Why it fell short of the gate

The brief's gate asks for 1 cm agreement between two captures of one room. The
declaration named the reason this is unreachable here, and the evidence has not
changed, in either stage:

`scan_floor_only`'s height histogram decays monotonically — 13,484 points in
the 1.5-1.6 m band down to 2,669 at 2.0-2.1 m, and **nothing above**. The
highest point on any wall is 2.05 m. The flat's ceiling is 3.07 m, measured
from the paired scan where 40.8% of points sit above camera height.

**The ceiling is not mis-measured in that capture. It was never captured.**
Closing 34.5 cm to 1 cm would mean either inventing a number or copying it from
the other scan — the second would make the gate pass while making the pipeline
worse, and the first is what this project exists not to do. The same logic
covers the now-improved-but-still-failing `repeatability` row: better wall
matching (stage 2) narrows the gap from 718 cm to 158 cm, but it cannot recover
wall geometry the capture never observed near an uncaptured ceiling.

What remains is the residual of an absent observation, and the honest output for
it is a wide interval that contains the truth plus a note telling the operator
what to do differently:

> "The only measurement here is the lower bound 2.05 m; the reported height is
> the midpoint of the band between that bound and a plausible maximum ceiling,
> not an observation. Read the interval, not the value, and sweep the phone
> upward to turn this into a measurement."

**The real remedy is a capture protocol instructing the operator to sweep
upward** — now shipped as [capture_protocol.md](../capture_protocol.md)'s rule
1 ("sweep the phone upward at every corner... skip this and there is no ceiling
height, only a wide interval"). A code change cannot recover data a past walk
did not collect; the remaining 2 FAILs are expected to clear on a recapture
that follows that rule, not on a further scorer or estimator change.
