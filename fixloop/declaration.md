# Fix loop declaration

Written **before** the fix was implemented. The commit that adds this file
precedes the commit that changes the code, so the order is auditable in
`git log`.

Before run: `fixloop/before/` — regenerate with
`bash scripts/run_benchmark.sh fixloop/before`.

---

## 1. The single worst-performing gate, with the failing number

**`ceiling_spread`**, on the one genuine repeat pair in the benchmark
(`scan_floor_only` and `scan_with_ceiling`, two LiDAR walkthroughs of the same
flat, `space_id: apartment_full`):

```
ceiling_spread   lidar:room_a   spread 62.9 cm across 2 captures   gate: <= 1 cm   FAIL
```

**62.9 cm against a 1 cm gate — 63× over.**

`repeatability` fails harder in absolute terms (718.4 cm), but that number is
not trustworthy: wall ids are positional, the two scans produced polygons with
12 and 20 walls, and the gate therefore compares walls that are not the same
wall. `ceiling_spread` compares a single scalar per capture, so its failure is
real and unambiguous. That is why it is the declared target.

## 2. Root-cause hypothesis, and the evidence

**Hypothesis: the two captures do not disagree about a measurement. One of them
never made the measurement, and the pipeline substituted a generic 2.44 m
structural prior that this building does not have.**

Evidence, in order of strength:

**(a) The two captures reach the number by different methods.**

| Capture | Ceiling | `ceiling_method` | Points above camera height |
|---|---|---|---|
| `scan_with_ceiling` | **3.069 m** | `measured_plane` | 40.8% |
| `scan_floor_only` | **2.440 m** | `scan_cutoff_prior` | 8.5% |

2.440 is exactly `prior_height_m`, the default in
`cozmo/geometry/planes.py::estimate_ceiling`. It is not an observation.

**(b) The flat's ceiling really is ~3.07 m.** Height histogram of
`scan_with_ceiling`, above the fitted floor:

```
2.3-2.4m: 46,083    2.4-2.5m: 45,551    2.2-2.3m: 39,351
3.0-3.1m: 35,024  <- a distinct band, well separated from wall content
```

**(c) `scan_floor_only` contains no ceiling information whatsoever.** Same
histogram, same flat:

```
1.5-1.6m: 13,484
1.6-1.7m: 11,412
1.7-1.8m:  8,530
1.8-1.9m:  7,016
1.9-2.0m:  4,691
2.0-2.1m:  2,669
above:     nothing
```

Monotonic decay to nothing by 2.1 m. The operator kept the phone low for the
whole walk. Wall tops confirm it: they disagree by 0.46 m (median 1.79 m,
highest 2.05 m), which fails the wall-extrapolation consensus test, so the
estimator falls through to the prior.

**(d) Therefore the 62.9 cm is not an estimation error and cannot be closed by
estimating better.** The highest point observed on any wall is 2.05 m, a full
metre below the true ceiling. No estimator over that data recovers 3.07 m.

**(e) What *is* a defect: the interval excludes reality.** `scan_floor_only`
reports

```
ceiling_height: 2.440  ci_95: [2.049, 2.990]
```

The truth, from the paired scan, is **3.069 m — outside the claimed 95%
interval.** The upper bound is built as `prior + 0.55 = 2.99`, so the fallback
is structurally incapable of covering any ceiling above 2.99 m. A 3.0 m
apartment ceiling is entirely ordinary. This is a calibration failure, and it
is the part that is genuinely broken.

## 3. The fix I intend to ship, and the number I predict

**Fix:** re-anchor the unmeasured-ceiling fallback in
`cozmo/geometry/planes.py::estimate_ceiling`. When no ceiling plane is fitted
and the wall tops do not agree, the capture has established only a *lower
bound* (the highest point actually seen on a wall). Every height between that
bound and a plausible maximum residential ceiling is equally consistent with
the data, so:

- the interval becomes `[observed lower bound, plausible maximum]` — anchored on
  a realistic tall-ceiling limit rather than on `prior + 0.55`, so it can
  actually contain a 3.0-3.4 m ceiling;
- the point estimate becomes the **midpoint of that band** rather than a fixed
  2.44 m prior, because with no observation above the bound the midpoint is the
  estimate that minimises worst-case error.

Scope: one function, one branch — the `scan_cutoff_prior` path only. The
`measured_plane` and `wall_extrapolation` paths are untouched.

### Predicted numbers

| Metric | Before | Predicted after |
|---|---|---|
| `scan_floor_only` ceiling value | 2.440 m | **≈2.72 m** |
| `scan_floor_only` ceiling interval | [2.049, 2.990] — **excludes** truth | **[≈2.05, ≈3.40] — contains 3.069** |
| `ceiling_spread` (genuine pair) | 62.9 cm | **≈35 cm** |
| `ceiling_spread` gate | FAIL | **still FAIL** |
| `apartment_lidar` ceiling value | 2.440 m | ≈2.69 m |
| Synthetic fixtures (`measured_plane`, `wall_extrapolation`) | 2.500 / 2.495 m | **unchanged** |
| Test suite | 202 pass | **202 pass** |

### I am predicting this gate will not pass, and stating why up front

The gate asks for 1 cm agreement. One of the two captures does not contain the
ceiling — the information is absent, not mis-estimated. Closing a 62.9 cm gap
to 1 cm would require inventing a number, or reading the answer off the other
capture, and either would make the gate pass while making the pipeline worse.

What the fix repairs is the part that is actually wrong: an interval that
claimed 95% confidence in a range that did not contain the truth. After the fix
the plan is honest about the ceiling being unknown-within-a-band, and the
residual spread is a property of the capture, not of the estimator.

**The real remedy for this gate is a capture protocol that tells the operator to
sweep the phone upward** — which is a Part 1 deliverable, not a code change, and
is named as such in `compliance_matrix.md` (row 1.1, not met).
