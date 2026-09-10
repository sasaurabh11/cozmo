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

VENV="${VENV:-.venv}"

# --- pick an interpreter open3d actually has wheels for -------------------
#
# open3d publishes wheels for CPython 3.8-3.12 only (0.19.0, the newest
# release, tops out at cp312). There is no 3.13 or 3.14 wheel and no source
# fallback, so on a machine where `python3` is 3.13+ -- Homebrew's default on
# current macOS -- `pip install open3d` fails outright with "No matching
# distribution found". Taking whatever `python3` happens to be therefore
# breaks setup on new Macs, which is exactly the clean-machine case that
# matters most. So: search for a supported interpreter instead of assuming.
#
# Override with PYTHON=/path/to/python3.11 to force a specific one.
PY_MIN_MINOR=9
PY_MAX_MINOR=12

python_ok() {                     # $1 = interpreter; 0 if version is in range
  local v
  v="$("$1" -c 'import sys;print(f"{sys.version_info.major} {sys.version_info.minor}")' 2>/dev/null)" || return 1
  local major="${v% *}" minor="${v#* }"
  [ "$major" = "3" ] && [ "$minor" -ge "$PY_MIN_MINOR" ] && [ "$minor" -le "$PY_MAX_MINOR" ]
}

PYTHON="${PYTHON:-}"
if [ -n "$PYTHON" ]; then
  python_ok "$PYTHON" || {
    echo "PYTHON=$PYTHON is $("$PYTHON" --version 2>&1), outside the supported 3.$PY_MIN_MINOR-3.$PY_MAX_MINOR range." >&2
    echo "open3d publishes no wheel for it. Pass a supported interpreter instead." >&2
    exit 1
  }
else
  for candidate in python3.12 python3.11 python3.10 python3.9 python3 /usr/bin/python3; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    if python_ok "$candidate"; then PYTHON="$candidate"; break; fi
  done
fi

if [ -z "$PYTHON" ]; then
  echo "No supported Python found. This project needs CPython 3.$PY_MIN_MINOR-3.$PY_MAX_MINOR," >&2
  echo "because open3d ships no wheel for 3.13 or newer." >&2
  echo >&2
  echo "  Interpreters found on this machine:" >&2
  for candidate in python3 python3.9 python3.10 python3.11 python3.12 python3.13 python3.14; do
    command -v "$candidate" >/dev/null 2>&1 && echo "    $candidate  $("$candidate" --version 2>&1)" >&2
  done
  echo >&2
  echo "  Install one and re-run, e.g.:" >&2
  echo "    brew install python@3.12   &&  scripts/setup.sh" >&2
  echo "  or point at an existing one:" >&2
  echo "    PYTHON=/usr/bin/python3 scripts/setup.sh" >&2
  exit 1
fi

echo "==> creating $VENV with $PYTHON ($($PYTHON --version 2>&1))"
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
