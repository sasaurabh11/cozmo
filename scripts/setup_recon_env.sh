#!/usr/bin/env bash
# Build the separate virtualenv the VGGT backbone runs under.
#
# Why a second venv: the upstream `vggt` package pins numpy<2 and needs
# Python >= 3.10. The rest of this project runs numpy 2.x on Python 3.9 --
# open3d and several geometry fixes (see cozmo/geometry/planes.py, layout.py)
# depend on the numpy-2 API. Forcing a numpy downgrade into the main venv to
# satisfy one backbone would silently change behaviour everywhere else, so
# VGGT gets its own interpreter and talks to the main process over a small
# JSON+npz protocol (cozmo/recon/worker.py) -- the same pattern already used
# to keep open3d and torch apart (cozmo/semantics/worker.py), applied here to
# keep two numpy majors apart instead.
#
# Usage: scripts/setup_recon_env.sh [PYTHON311_BIN]

set -euo pipefail

PYTHON="${1:-python3.11}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Need Python >= 3.10 (vggt's own requirement). Install one (e.g. \`brew install python@3.11\`)" >&2
  echo "and pass it: scripts/setup_recon_env.sh /path/to/python3.11" >&2
  exit 1
fi

VGGT_SRC="${COZMO_VGGT_SRC:-}"
if [[ -z "$VGGT_SRC" ]]; then
  VGGT_SRC="$(mktemp -d)/vggt"
  echo "fetching vggt source into $VGGT_SRC"
  curl -sL "https://codeload.github.com/facebookresearch/vggt/zip/refs/heads/main" -o "$VGGT_SRC.zip"
  unzip -q "$VGGT_SRC.zip" -d "$(dirname "$VGGT_SRC")"
  mv "$(dirname "$VGGT_SRC")/vggt-main" "$VGGT_SRC"
fi

"$PYTHON" -m venv .venv-recon
.venv-recon/bin/pip install -q --upgrade pip
.venv-recon/bin/pip install -q torch torchvision 'numpy<2' einops safetensors \
    scipy shapely opencv-python-headless Pillow huggingface_hub -e "$VGGT_SRC"

echo
echo "done. .venv-recon/bin/python is used automatically by VGGTReconstructor,"
echo "or point at a different interpreter with COZMO_RECON_PYTHON=/path/to/python."
