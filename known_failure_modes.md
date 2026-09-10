# Known failure modes

The brief names four: **mirrors, glass, wet-look surfaces, low light**. This is
what each actually did to our output, measured on the captures we hold.

**Instrument:** the Stray Scanner **confidence channel** — a per-pixel depth
confidence map (0 = no usable return, 1 = medium, 2 = high) written beside every
depth frame. It is the sensor's own admission of where it failed.

- Figure: `docs/evidence/confidence_failure_modes.jpg` — RGB beside the
  confidence map for the six worst frames across three real captures. Black =
  zero-confidence depth.
- Reproduce every number below: `python scripts/confidence_stats.py`

## Baseline: depth lost per capture

| Capture | Zero-conf (mean) | p90 | Worst frame | High-conf (mean) |
|---|---|---|---|---|
| `apartment_lidar` | 1.9% | 4.1% | **27.8%** | 93.6% |
| `scan_with_ceiling` | 4.4% | 16.3% | **38.5%** | 90.5% |
| `scan_floor_only` | 4.8% | 14.4% | **44.1%** | 89.8% |

Loss is not spread evenly. It concentrates in a few frames, and those frames are
the ones pointed at the four named surfaces.

## Mirrors

`scan_floor_only`, frame 2021 — **37% of the frame returns no depth.** The RGB
shows a wall mirror with the room reflected in it; the confidence map goes black
across exactly the mirror's extent while the surrounding wall stays high.

LiDAR measures the *reflected* optical path, so a mirror reads as a hole punched
into the wall opening onto a room that does not exist.

**Our behaviour:** those pixels drop out at ingest (confidence-gated), so the
mirror never becomes phantom geometry — the wall behind it is simply
under-observed. The cost appears one stage later in `wall_observation_fractions`,
where `detect_openings` then **refuses to claim openings in that wall**:

```
scan_floor_only: walls [3, 4, 11, 14, 15, 18] observed across less than 60% of
their length; absence of openings in them is not evidence they are solid
```

## Glass

- `scan_with_ceiling`, frame 81 — **36% zero-confidence.** A glazed partition
  with reflected ceiling lights. Confidence collapses over the panels and holds
  on the frame around them.
- `scan_floor_only`, frame 4257 — **44%, the worst frame in the whole set:** a
  dark glossy cabinet front. Dark *and* specular is the worst case for an IR
  time-of-flight sensor — little light returns, and what does is off-axis.

Same mechanism as mirrors: glazed walls are under-observed, so openings in them
are not claimed. This is a deliberate false-negative bias — **we would rather
miss a glass door than invent one** — and it is why all three real LiDAR captures
report **zero openings** despite plainly having doors.

## Wet-look / polished surfaces

`apartment_lidar`, frame 1190 — **28% zero-confidence.** A bathroom: polished
tile floor plus a glass shower screen. The loss is across the glossy floor, not
the matte walls.

This degrades the *floor*, which matters more than it sounds: the floor plane is
what `fit_floor` fits, what the polygon's interior-evidence test counts, and what
wall positions are snapped against. Sparse floor returns under a glossy finish
reduce the evidence the polygon is built from — the same path that produced a
bathroom with 9.5% floor coverage in the photo tier.

## Low light — the measurement contradicts the expectation

Luminance against depth confidence, all three captures:

| Capture | corr(luminance, zero-conf) | Darkest quartile | Brightest quartile | Ratio |
|---|---|---|---|---|
| `scan_with_ceiling` | **−0.08** | 4.1% | 3.3% | 1.2× |
| `scan_floor_only` | **−0.12** | 6.9% | 3.2% | 2.2× |
| `apartment_lidar` | **−0.09** | 2.4% | 1.8% | 1.3× |

**Low light barely affects LiDAR depth.** Correlation is near zero; the darkest
quartile is only 1.2–2.2× worse than the brightest. That is the expected physics
once stated plainly — LiDAR is an *active* sensor emitting its own IR, so ambient
light is close to irrelevant.

Low light hurts every **passive** stage instead, and we have not isolated it there
with the same rigour:

- **Photo and video tiers.** VGGT reconstructs from RGB alone. Low light means
  noise and motion blur, and blur is what the video sampler's Laplacian-variance
  filter removes — on the real 1,105-frame walkthrough it **discarded 47 of 74
  sampled frames as too blurred**. A direct, quantified cost on our own capture.
- **The damage and opening detector.** Grounding DINO and SAM 2 run on RGB; their
  thresholds were not tuned for low light and their precision is unvalidated.

Put the other way round: **the tier the brief expects low light to hurt is the one
that is immune, and the tiers that are hurt are the ones with no ground truth to
measure the harm on.**

## Summary

| Mode | Sensor effect | Our behaviour | Failure direction |
|---|---|---|---|
| Mirror | Depth measures the reflected path | Confidence-gated at ingest; wall marked under-observed | False negative (missed openings), never phantom geometry |
| Glass | Little or no return, up to 44% of frame | Same path; openings not claimed in under-observed walls | False negative |
| Wet-look / polished | Sparse floor returns | Weaker floor evidence for the polygon | Under- or mis-sized rooms |
| Low light | **Negligible for LiDAR** (r ≈ −0.1) | — | Hurts photo/video and the detector; 47/74 frames dropped for blur |

## What we do not handle

- **No mirror or glass *detection*.** We degrade gracefully because the
  confidence channel happens to mark these surfaces, not because anything
  recognises them. A mirror dominating a wall would leave that wall unmeasured
  and reported as under-observed — and nothing would say "mirror".
- **Photo and video have no confidence channel.** Nothing in VGGT's output is
  treated as an admission of failure, so the same mirror in a photo capture
  degrades **silently**.
- **Low-light impact on the passive tiers is asserted, not measured.** We can
  quantify frames dropped for blur; we cannot yet quantify the resulting
  dimensional error, because that needs ground truth those tiers do not have.
