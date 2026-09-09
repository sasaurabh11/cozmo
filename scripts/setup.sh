#!/usr/bin/env bash
# One-command setup for the base pipeline (LiDAR tier + benchmark + tests).
#
#     scripts/setup.sh && source .venv/bin/activate
#
# Deliberately does NOT fetch model weights: the LiDAR tier is pure geometry and
# needs none, so a reviewer can clone, set up and reconstruct a real room without
# waiting on 6.9 GB. The photo/video tiers and the damage detector need weights
# and their own interpreter; both are opt-in:
#
#     scripts/fetch_weights.sh --group photo      # VGGT + matcher   (~5.2 GB)
#     scripts/fetch_weights.sh --group semantics  # detector + depth (~1.7 GB)
#     scripts/setup_recon_env.sh                  # .venv-recon for VGGT
#
# A fresh venv built from a system Python 3.9 ships pip 21.x, which predates
# PEP 660 and fails `pip install -e` outright with "editable mode currently
# requires a setuptools-based build". Upgrading pip first is not cosmetic; it is
# the difference between this repo installing and not installing on a clean
# machine.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"

echo "==> creating $VENV with $($PYTHON --version 2>&1)"
[ -d "$VENV" ] || "$PYTHON" -m venv "$VENV"

echo "==> upgrading pip (PEP 660 editable installs need pip >= 21.3)"
"$VENV/bin/python" -m pip install --quiet --upgrade pip setuptools wheel

echo "==> installing cozmo and its dependencies"
"$VENV/bin/python" -m pip install --quiet -e ".[dev]"

echo "==> building the ray-traced capture fixtures (exact ground truth)"
"$VENV/bin/python" tests/fixtures/synthesize.py >/dev/null

echo "==> smoke test: reconstructing the synthetic room"
"$VENV/bin/python" -m cozmo.cli run \
    --input tests/fixtures/captures/synthetic_room \
    --out out/_setup_check --quiet

"$VENV/bin/python" - <<'PY'
import json
plan = json.load(open("out/_setup_check/plan.json"))
walls = sorted(round(w["length"]["value"], 2) for w in plan["rooms"][0]["walls"])
area = plan["property_totals"]["footprint_area"]["value"]
print(f"    walls {walls}  footprint {area:.2f} m2   (truth: [2.8, 2.8, 3.6, 3.6], 10.08 m2)")
PY

cat <<'EOF'

Setup complete.

    source .venv/bin/activate
    cozmo run --input <capture-dir> --out out/run

Optional, for the photo and video tiers:
    scripts/fetch_weights.sh --group photo
    scripts/setup_recon_env.sh
EOF
