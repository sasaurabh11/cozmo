#!/usr/bin/env bash
# Fetch model weights into ./weights.
#
# Nothing to fetch yet: the pipeline is a stub with no learned components. The
# script exists now, and is wired into the Docker build and the README, so that
# adding the first model is one entry in the table below rather than a change to
# how the project is run. Weights are never committed -- the brief requires large
# binaries to arrive by script or volume.
#
# Usage:  scripts/fetch_weights.sh [DEST]        (default: ./weights)
#         scripts/fetch_weights.sh --check       verify what is already there

set -euo pipefail

DEST="${1:-weights}"
CHECK_ONLY=0
if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=1
  DEST="weights"
fi

# name|url|sha256   -- add a row per model; the loop below does the rest.
MODELS=(
)

mkdir -p "$DEST"

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "no weights required at this pipeline version (stub reconstruction)"
  echo "destination ready: $DEST"
  exit 0
fi

for row in "${MODELS[@]}"; do
  IFS='|' read -r name url want <<< "$row"
  target="$DEST/$name"

  if [[ -f "$target" ]]; then
    got="$(sha256_of "$target")"
    if [[ "$got" == "$want" ]]; then
      echo "ok       $name"
      continue
    fi
    echo "stale    $name (sha256 $got != $want)" >&2
    [[ $CHECK_ONLY -eq 1 ]] && exit 1
    rm -f "$target"
  elif [[ $CHECK_ONLY -eq 1 ]]; then
    echo "missing  $name" >&2
    exit 1
  fi

  echo "fetching $name"
  curl --fail --location --retry 3 --output "$target" "$url"

  got="$(sha256_of "$target")"
  if [[ "$got" != "$want" ]]; then
    echo "checksum mismatch for $name: got $got, expected $want" >&2
    rm -f "$target"
    exit 1
  fi
  echo "ok       $name"
done
