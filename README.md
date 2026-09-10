# Cozmo AI — floor-plan reconstruction pipeline

A phone capture goes in — **LiDAR scan**, **photos**, or a **walkthrough video** —
and a dimensioned floor plan comes out: walls, ceiling height, floor area, doors
and windows, damage regions, and a drawing. Every measurement carries a
confidence interval.

One command per capture. The tier is read from the capture, never passed as a flag.

---

## 1. Install

**You need:** Python 3.9–3.12, `git`, `curl`, ~1 GB disk. macOS or Linux. No GPU required.

```bash
git clone <repo> cozmo && cd cozmo
scripts/setup.sh
source .venv/bin/activate
```

That's it — **about 2 minutes** on a fresh clone. The script picks a supported
interpreter, creates the virtualenv, installs dependencies, builds the test
fixtures, and runs a smoke reconstruction to prove it worked.

> Needs **CPython 3.9–3.12**. `open3d` publishes no wheel for 3.13+, so if your
> `python3` is newer the script finds an older one for you, or tells you to
> `brew install python@3.12`. Force one with `PYTHON=/path/to/python3.12 scripts/setup.sh`.

Check it:

```bash
cozmo version
```

### Optional: model weights

**The LiDAR tier needs no weights at all.** Only add these if you want the photo
or video tiers, or damage detection:

```bash
scripts/fetch_weights.sh --group photo       # photo + video tiers  (~5.2 GB)
scripts/setup_recon_env.sh                   # second venv VGGT needs

scripts/fetch_weights.sh --group semantics   # damage detection     (~1.7 GB)
```

Verify what you downloaded: `scripts/fetch_weights.sh --check`

---

## 2. Run it

### Try it now (no weights, no capture needed)

```bash
cozmo run --input tests/fixtures/captures/synthetic_room --out out/room
open out/room/plan.png
```

This room is ray-traced, so the true answer is known exactly — 3.60 × 2.80 m,
2.50 m ceiling. Score the reconstruction against it:

```bash
cozmo benchmark \
    --results out/room \
    --ground-truth tests/fixtures/benchmark/ground_truth_synthetic.csv \
    --out out/bench
```

Every gate passes; worst wall error 0.1 cm.

### On your own capture

**Step 1 — put the files in a folder** in the layout for your tier:

```
LiDAR (Stray Scanner)          Photos                     Video
my_capture/                    my_capture/                my_capture/
├── capture.json               ├── capture.json           ├── capture.json
├── rgb.mp4                    └── rooms/                 └── walkthrough.mov
├── camera_matrix.csv              ├── kitchen/
├── odometry.csv                   │   ├── 01.jpg
├── depth/                         │   └── 02.jpg
└── confidence/                    └── bedroom/
                                       └── ...
```

**Step 2 — write `capture.json`** in that folder:

```json
{
  "capture_id": "my_capture",
  "tier": "lidar",
  "device": {"model": "iPhone 15 Pro", "has_lidar": true},
  "declared_rooms": ["kitchen"]
}
```

`tier` is `lidar`, `photo`, or `video`. That one field decides everything.

**Step 3 — run:**

```bash
cozmo run --input my_capture --out out/my_capture --verbose
```

Photo and video tiers need the weights from step 1 and take ~40–50 s per room;
LiDAR takes 3–70 s depending on frame count.

---

## 3. What you get

In your `--out` folder:

| File | Contents |
|---|---|
| **`plan.json`** | The full output: walls, ceiling, floor area, openings, damage, concealed-damage flags, scope items — every quantity with a 95% interval |
| **`plan.png`** / `.svg` | The drawing |
| **`run_manifest.json`** | Git commit, exact command, seed, SHA-256 of the input — so any number traces back to the bytes that produced it |

Reading `plan.json`:

```jsonc
"ceiling_height": { "value": 2.44, "ci_95": [1.99, 2.99], "unit": "m" }
```

`value` is the estimate; `ci_95` is an **absolute** 95% interval in the same
unit. A wide interval is the pipeline telling you it is not confident — check
`quality.degradations` and `quality.ceiling_method` for why.

---

## 4. Commands

```bash
cozmo run       --input DIR --out DIR      # reconstruct one capture
cozmo benchmark --results DIR --ground-truth CSV --out DIR   # score against truth
cozmo calibrate --captures DIR --ground-truth CSV --out FILE # fit intervals
cozmo version
```

Useful flags — `cozmo run --help` lists them all:

| Flag | Effect |
|---|---|
| `--verbose` | Log every stage as it runs |
| `--drift-correction on\|off` | The drift ablation (LiDAR). `off` is the comparison arm |
| `--no-semantics` | Skip damage detection — faster, no weights needed |
| `--backbone stub` | Skip VGGT entirely — instant, for testing plumbing |

Regenerate the whole benchmark, all captures, timed and scored:

```bash
bash scripts/run_benchmark.sh benchmark_runs
```

Run the tests:

```bash
pytest                                              # 202 tests, ~75 s
COZMO_TEST_RECON=1 COZMO_TEST_DETECTOR=1 pytest     # + 4 more, needs weights
```

---

## 5. Everything else

| Document | Contents |
|---|---|
| **[capture_protocol.md](capture_protocol.md)** | The one-page capture protocol — which app, how to walk, how to hand the files over |
| **[docs/design.md](docs/design.md)** | How each tier works and why — per-view planes, scale recovery, stitching, the rule engine, capture provenance |
| **[technical_report.md](technical_report.md)** | Architecture, error budget, calibration, drift, fix loop, failure modes |
| **[benchmark_report.md](benchmark_report.md)** | Gates at all three tiers, repeatability, timing |
| **[compliance_matrix.md](compliance_matrix.md)** | Requirement-by-requirement status |
| **[known_failure_modes.md](known_failure_modes.md)** | Mirrors, glass, wet-look surfaces, low light — with measured evidence |
| **[fixloop/](fixloop/)** | The fix loop: declaration, before/after runs, diff |

### Honest summary

The LiDAR tier is exact where it can be checked (0.1 cm on ray-traced truth) and
needs no models. Photo and video reconstruct real rooms but carry ±60% intervals,
because scale rests on a single assumed-ceiling cue — the intervals are wide on
purpose rather than confidently wrong.

**No real capture has laser or tape ground truth**, so 45 of 94 benchmark gate
rows report SKIP, which is never a pass. See
[benchmark_report.md](benchmark_report.md) for what that does and does not prove.
