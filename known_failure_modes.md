# Known failure modes

The brief names four: **mirrors, glass, wet-look surfaces and low light**. This
section reports what each actually did to our output, measured from the captures
we hold rather than described in general terms.

The instrument is the Stray Scanner **confidence channel** — a per-pixel depth
confidence map (0 = no usable return, 1 = medium, 2 = high) written alongside
every depth frame. It is the sensor's own admission of where it failed, and it
is available for every LiDAR capture without any extra work.

Evidence figure: `docs/evidence/confidence_failure_modes.jpg` — RGB alongside the
confidence map for the six worst frames across three real captures. Black in the
confidence map is zero-confidence depth.

Reproduce with:

```bash
python3 - <<'PY'
import cv2, numpy as np, glob
from pathlib import Path
for cap in ['apartment_lidar','scan_with_ceiling','scan_floor_only']:
    files = sorted(glob.glob(f'captures/{cap}/confidence/*.png'))
    rows = []
    for f in files[::max(1, len(files)//120)]:
        c = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        rows.append((int(Path(f).stem), float((c == 0).mean())))
    a = np.array([r[1] for r in rows])
    worst = sorted(rows, key=lambda r: -r[1])[:5]
    print(f'{cap}: mean {a.mean()*100:.1f}%  p90 {np.percentile(a,90)*100:.1f}%  max {a.max()*100:.1f}%')
    print('   worst frames:', [(w[0], round(w[1]*100)) for w in worst])
PY
```

## Baseline: how much depth is lost, per capture

| Capture | Zero-confidence pixels (mean) | p90 | Worst frame | High-confidence (mean) |
|---|---|---|---|---|
| `apartment_lidar` | 1.9% | 4.1% | **27.8%** | 93.6% |
| `scan_with_ceiling` | 4.4% | 16.3% | **38.5%** | 90.5% |
| `scan_floor_only` | 4.8% | 14.4% | **44.1%** | 89.8% |

Depth loss is not spread evenly. It is concentrated in a small number of frames,
and those frames are the ones pointed at the four named surfaces.

## Mirrors

`scan_floor_only`, frame 2021 — **37% of the frame returns no depth.** The RGB
shows a wall mirror with a person and the room reflected in it. The confidence
map goes black across exactly the mirror's extent while the surrounding wall
stays high-confidence.

The LiDAR measures the *reflected* optical path, so a mirror reads as a hole
punched into the wall opening onto a room that does not exist. Our behaviour:
those pixels drop out at ingest (confidence-gated), so the mirror does not become
phantom geometry — the wall behind it is simply under-observed. The cost shows up
one stage later, in `wall_observation_fractions`: the wall is not observed across
enough of its length, and `detect_openings` then **refuses to claim openings in
it** rather than reporting the mirror as a doorway.

That refusal is visible in the plans:

```
scan_floor_only: walls [3, 4, 11, 14, 15, 18] observed across less than 60% of
their length; absence of openings in them is not evidence they are solid
```

## Glass

`scan_with_ceiling`, frame 81 — **36% zero-confidence.** The RGB is a glazed
partition with a reflected person and reflected ceiling lights. Confidence
collapses over the glass panels and holds on the frames around them.

`scan_floor_only`, frame 4257 — **44% zero-confidence**, the worst frame in the
whole set: a dark glossy cabinet front. Dark *and* specular is the worst case for
an IR time-of-flight sensor — little light returns, and what does is
off-axis.

Consequence, same mechanism as mirrors: glazed walls are under-observed, so
openings in them are not claimed. This is a false-negative bias — **we would
rather miss a glass door than invent one** — and it is the reason all three real
LiDAR captures report **zero openings** despite plainly having doors.

## Wet-look / polished surfaces

`apartment_lidar`, frame 1190 — **28% zero-confidence.** A bathroom: polished
tile floor plus a glass shower screen. The confidence map shows the loss across
the glossy floor, not the matte walls.

This one degrades the *floor*, which matters more than it sounds: the floor plane
is what `fit_floor` fits, what the room polygon's interior-evidence test counts,
and what wall positions are snapped against. Sparse floor returns under a glossy
finish reduce the evidence the polygon is built from, which is the same failure
path that produced a bathroom with 9.5% floor coverage in the photo tier.

## Low light — the measured result contradicts the expectation

We tested luminance against depth confidence across all three captures:

| Capture | corr(luminance, zero-conf) | Darkest quartile | Brightest quartile | Ratio |
|---|---|---|---|---|
| `scan_with_ceiling` | **−0.08** | 4.1% | 3.3% | 1.2× |
| `scan_floor_only` | **−0.12** | 6.9% | 3.2% | 2.2× |
| `apartment_lidar` | **−0.09** | 2.4% | 1.8% | 1.3× |

**Low light barely affects LiDAR depth.** The correlation is near zero and the
darkest quartile is only 1.2–2.2× worse than the brightest. This is the expected
physics once stated plainly: the LiDAR is an *active* sensor that emits its own
IR, so ambient light is close to irrelevant to it.

Where low light does hurt is every **passive** stage, and we have not isolated it
there with the same rigour:

- **Photo and video tiers.** VGGT reconstructs from RGB alone. Low light means
  noise and motion blur, and motion blur is exactly what the video sampler's
  Laplacian-variance filter removes — on the real 1,105-frame walkthrough it
  **discarded 47 of 74 sampled frames as too blurred**, which is a direct,
  quantified low-light/motion cost measured on our own capture.
- **The damage and opening detector.** Grounding DINO and SAM 2 run on RGB;
  their thresholds were not tuned for low light and their precision is
  unvalidated (no damage ground truth).

Stating it the other way round: **the tier the brief expects to be hurt by low
light is the one that is immune, and the tiers that are hurt are the ones with no
ground truth to measure the harm on.**

## Summary of behaviour under each mode

| Mode | Sensor effect | Our behaviour | Failure direction |
|---|---|---|---|
| Mirror | Depth measures the reflected path | Confidence-gated out at ingest; wall marked under-observed | False negative (missed openings), never phantom geometry |
| Glass | Little or no return; up to 44% of frame | Same path; openings not claimed in under-observed walls | False negative |
| Wet-look / polished | Sparse floor returns | Weaker floor evidence for the polygon | Under- or mis-sized rooms |
| Low light | **Negligible for LiDAR** (r ≈ −0.1) | — | Hurts photo/video and the detector instead; blurred frames dropped (47/74 on the real clip) |

## What we do not handle

- **No mirror or glass *detection*.** We degrade gracefully because the
  confidence channel happens to mark these surfaces, not because anything
  recognises them. A mirror large enough to dominate a wall would leave that wall
  unmeasured, reported as under-observed, and nothing would say "mirror".
- **Photo and video tiers have no confidence channel at all.** There is no
  equivalent signal in VGGT's output that we treat as an admission of failure, so
  the same mirror in a photo capture degrades silently.
- **Low-light impact on the passive tiers is asserted, not measured.** We can
  quantify frames dropped for blur; we cannot yet quantify the resulting
  dimensional error, because that needs ground truth those tiers do not have.
