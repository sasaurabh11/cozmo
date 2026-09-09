# Fix loop: before → after

Declaration (written first, committed first): [declaration.md](declaration.md)

| | Command | Tag |
|---|---|---|
| Before | `bash scripts/run_benchmark.sh fixloop/before` | `fixloop-before` |
| After | `bash scripts/run_benchmark.sh fixloop/after` | `fixloop-after` |

Both regenerate byte-comparably — `SOURCE_DATE_EPOCH` is pinned inside the
script, so `diff` is a valid check, not just inspection.

---

## What changed in the code

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

## What changed in the numbers

### The declared gate

| | Before | After | Gate |
|---|---|---|---|
| `ceiling_spread`, genuine pair (`apartment_full`) | **62.9 cm** | **34.5 cm** | ≤1 cm |
| `ceiling_spread`, as grouped by the scorer (3 captures) | 62.9 cm | **37.4 cm** | ≤1 cm |
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

### The whole-benchmark diff is one row

Same 11 captures both sides, scored identically:

```
before   26 PASS / 23 FAIL / 45 SKIP
after    26 PASS / 23 FAIL / 45 SKIP

gate rows that changed: 1
  ceiling_spread / lidar:room_a
     before: FAIL  spread 62.9 cm across 3 captures
     after : FAIL  spread 37.4 cm across 3 captures
```

One row, the declared one. Nothing else in the benchmark moved — which is the
point: the change is confined to a single branch of a single function, and the
diff is attributable to it and nothing else.

### Unchanged, as intended

| Capture | Ceiling method | Before | After |
|---|---|---|---|
| `synthetic_room` | `measured_plane` | 2.496 m | 2.496 m |
| `synthetic_no_ceiling` | `wall_extrapolation` | 2.495 m | 2.495 m |
| `scan_with_ceiling` | `measured_plane` | 3.069 m | 3.069 m |

`ceiling_height` gate: 2 PASS on the fixtures, before and after — worst error
0.4 cm / 0.5 cm against a 1.5 cm gate, unmoved. `interval_coverage_by_kind`
for `ceiling_height`: 2/2 covered (100%), before and after.

No other gate moves. The photo and video tiers do not call `estimate_ceiling`
at all — their ceiling comes from the scale prior — so the diff is confined to
the three LiDAR captures that took the `scan_cutoff_prior` branch.

## Why it fell short of the gate

The brief's gate asks for 1 cm agreement between two captures of one room. The
declaration named the reason this is unreachable here, and the evidence has not
changed:

`scan_floor_only`'s height histogram decays monotonically — 13,484 points in
the 1.5-1.6 m band down to 2,669 at 2.0-2.1 m, and **nothing above**. The
highest point on any wall is 2.05 m. The flat's ceiling is 3.07 m, measured
from the paired scan where 40.8% of points sit above camera height.

**The ceiling is not mis-measured in that capture. It was never captured.**
Closing 34.5 cm to 1 cm would mean either inventing a number or copying it from
the other scan — the second would make the gate pass while making the pipeline
worse, and the first is what this project exists not to do.

What remains is the residual of an absent observation, and the honest output for
it is a wide interval that contains the truth plus a note telling the operator
what to do differently:

> "The only measurement here is the lower bound 2.05 m; the reported height is
> the midpoint of the band between that bound and a plausible maximum ceiling,
> not an observation. Read the interval, not the value, and sweep the phone
> upward to turn this into a measurement."

**The real remedy is a capture protocol instructing the operator to sweep
upward** — a Part 1 deliverable, tracked as not met in
[compliance_matrix.md](../compliance_matrix.md) row 1.1. A code change cannot
recover data the walk did not collect, and claiming otherwise would be the
"confident garbage on thin input" the brief penalises.
