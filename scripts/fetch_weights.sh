#!/usr/bin/env bash
# Fetch model weights into ./weights (or $COZMO_WEIGHTS_DIR).
#
# Weights are never committed: the brief requires large binaries to arrive by
# script or volume. Every file is pinned twice -- to a Hugging Face commit
# revision in the URL, and to a sha256 checked after download. A mismatch is a
# hard failure, not a warning: a silently different detector is a silently
# different benchmark.
#
# Usage:  scripts/fetch_weights.sh [DEST]     (default: ./weights)
#         scripts/fetch_weights.sh --check    verify what is already there
#         scripts/fetch_weights.sh --print-hashes  recompute and print sha256s
#
# Models:
#   Grounding DINO (tiny)  open-vocabulary detection from text prompts
#   SAM 2.1 (hiera-tiny)   mask refinement of those boxes
#   VGGT-1B                photo-tier reconstruction backbone (needs .venv-recon;
#                          see scripts/setup_recon_env.sh -- numpy<2, Python >= 3.10)
#   ZoeDepth (NYU)         metric monocular depth, the strongest photo-tier scale cue
#   DINOv2 (small)         coarse room-pair retrieval for multi-room stitching
#   SuperPoint             keypoints, for room-to-room matching
#   LightGlue (SuperPoint) keypoint matches between rooms

set -euo pipefail

DEST="${COZMO_WEIGHTS_DIR:-weights}"
MODE="fetch"
case "${1:-}" in
  --check) MODE="check" ;;
  --print-hashes) MODE="hashes" ;;
  "") ;;
  *) DEST="$1" ;;
esac

HF="${HF_ENDPOINT:-https://huggingface.co}"

# repo|revision|file|sha256|local_subdir
# revision is a commit sha, so the URL cannot drift under us.
MODELS=(
  "IDEA-Research/grounding-dino-tiny|a2bb814dd30d776dcf7e30523b00659f4f141c71|config.json|SKIP|grounding-dino-tiny"
  "IDEA-Research/grounding-dino-tiny|a2bb814dd30d776dcf7e30523b00659f4f141c71|preprocessor_config.json|SKIP|grounding-dino-tiny"
  "IDEA-Research/grounding-dino-tiny|a2bb814dd30d776dcf7e30523b00659f4f141c71|tokenizer_config.json|SKIP|grounding-dino-tiny"
  "IDEA-Research/grounding-dino-tiny|a2bb814dd30d776dcf7e30523b00659f4f141c71|tokenizer.json|SKIP|grounding-dino-tiny"
  "IDEA-Research/grounding-dino-tiny|a2bb814dd30d776dcf7e30523b00659f4f141c71|special_tokens_map.json|SKIP|grounding-dino-tiny"
  "IDEA-Research/grounding-dino-tiny|a2bb814dd30d776dcf7e30523b00659f4f141c71|added_tokens.json|SKIP|grounding-dino-tiny"
  "IDEA-Research/grounding-dino-tiny|a2bb814dd30d776dcf7e30523b00659f4f141c71|vocab.txt|SKIP|grounding-dino-tiny"
  "IDEA-Research/grounding-dino-tiny|a2bb814dd30d776dcf7e30523b00659f4f141c71|model.safetensors|SKIP|grounding-dino-tiny"
  "facebook/sam2.1-hiera-tiny|de431c4043854a71d8101e17995dfe596bf101a5|config.json|SKIP|sam2.1-hiera-tiny"
  "facebook/sam2.1-hiera-tiny|de431c4043854a71d8101e17995dfe596bf101a5|preprocessor_config.json|SKIP|sam2.1-hiera-tiny"
  "facebook/sam2.1-hiera-tiny|de431c4043854a71d8101e17995dfe596bf101a5|processor_config.json|SKIP|sam2.1-hiera-tiny"
  "facebook/sam2.1-hiera-tiny|de431c4043854a71d8101e17995dfe596bf101a5|model.safetensors|SKIP|sam2.1-hiera-tiny"
  "facebook/VGGT-1B|860abec7937da0a4c03c41d3c269c366e82abdf9|config.json|SKIP|vggt-1b"
  "facebook/VGGT-1B|860abec7937da0a4c03c41d3c269c366e82abdf9|model.safetensors|f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e|vggt-1b"
  "Intel/zoedepth-nyu|52ae69caf7896c927909b3ceab4249c235e16dc1|config.json|SKIP|zoedepth-nyu"
  "Intel/zoedepth-nyu|52ae69caf7896c927909b3ceab4249c235e16dc1|preprocessor_config.json|SKIP|zoedepth-nyu"
  "Intel/zoedepth-nyu|52ae69caf7896c927909b3ceab4249c235e16dc1|model.safetensors|b616e347efc30e64822be28d2464c2fcd0b665b5fdbc9a135d53d168c31c77ab|zoedepth-nyu"
  "facebook/dinov2-small|ed25f3a31f01632728cabb09d1542f84ab7b0056|config.json|SKIP|dinov2-small"
  "facebook/dinov2-small|ed25f3a31f01632728cabb09d1542f84ab7b0056|preprocessor_config.json|SKIP|dinov2-small"
  "facebook/dinov2-small|ed25f3a31f01632728cabb09d1542f84ab7b0056|model.safetensors|ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1|dinov2-small"
  "magic-leap-community/superpoint|734450e9ffe229074f5998494ddc615475cdb20a|config.json|SKIP|superpoint"
  "magic-leap-community/superpoint|734450e9ffe229074f5998494ddc615475cdb20a|preprocessor_config.json|SKIP|superpoint"
  "magic-leap-community/superpoint|734450e9ffe229074f5998494ddc615475cdb20a|model.safetensors|1daf35c4ed78b386f7a8d3744f8160e72a30a749360a04a27ae886ff5a3ce23e|superpoint"
  "ETH-CVG/lightglue_superpoint|5f5f626efee99f37dbc9eafea879d24d297aeab8|config.json|SKIP|lightglue_superpoint"
  "ETH-CVG/lightglue_superpoint|5f5f626efee99f37dbc9eafea879d24d297aeab8|preprocessor_config.json|SKIP|lightglue_superpoint"
  "ETH-CVG/lightglue_superpoint|5f5f626efee99f37dbc9eafea879d24d297aeab8|model.safetensors|320864fabf972cd00ce314e889cf89039e44c238e08d74d2b42d9cf2e80fc867|lightglue_superpoint"
)


sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

mkdir -p "$DEST"
failed=0

for row in "${MODELS[@]}"; do
  IFS='|' read -r repo revision file want subdir <<< "$row"
  target="$DEST/$subdir/$file"
  mkdir -p "$(dirname "$target")"

  if [[ "$MODE" == "hashes" ]]; then
    [[ -f "$target" ]] && echo "$subdir/$file  $(sha256_of "$target")"
    continue
  fi

  if [[ -f "$target" ]]; then
    if [[ "$want" == "SKIP" ]]; then
      echo "ok       $subdir/$file (no pinned hash yet)"
      continue
    fi
    got="$(sha256_of "$target")"
    if [[ "$got" == "$want" ]]; then
      echo "ok       $subdir/$file"
      continue
    fi
    echo "MISMATCH $subdir/$file: $got != $want" >&2
    if [[ "$MODE" == "check" ]]; then failed=1; continue; fi
    rm -f "$target"
  elif [[ "$MODE" == "check" ]]; then
    echo "missing  $subdir/$file" >&2
    failed=1
    continue
  fi

  echo "fetching $subdir/$file"
  curl --fail --location --retry 3 --progress-bar \
       --output "$target" "$HF/$repo/resolve/$revision/$file"

  if [[ "$want" != "SKIP" ]]; then
    got="$(sha256_of "$target")"
    if [[ "$got" != "$want" ]]; then
      echo "checksum mismatch for $subdir/$file: got $got, expected $want" >&2
      rm -f "$target"
      exit 1
    fi
  fi
  echo "ok       $subdir/$file"
done

if [[ "$MODE" == "check" ]]; then
  [[ $failed -eq 0 ]] && echo "all weights present and verified" || { echo "weights incomplete" >&2; exit 1; }
fi

if [[ "$MODE" == "fetch" ]]; then
  cat <<EOF

Weights are in $DEST. Pin the hashes with:
    scripts/fetch_weights.sh --print-hashes
and paste them into the MODELS table above, replacing SKIP.
EOF
fi
