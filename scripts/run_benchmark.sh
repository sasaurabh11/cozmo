#!/usr/bin/env bash
# Regenerate the whole benchmark: every capture, every tier, timed, then scored.
#
#     scripts/run_benchmark.sh [OUT_DIR]      (default: benchmark_runs)
#
# One command, no arguments needed. Writes:
#   OUT_DIR/<capture>/plan.json          one per capture
#   OUT_DIR/<capture>/run_manifest.json  provenance: git commit, input sha256, seed
#   OUT_DIR/timing.csv                   capture,tier,seconds,exit_code
#   OUT_DIR/results.json                 the scored gate table
#
# Deterministic: SOURCE_DATE_EPOCH is pinned so plan.json is byte-identical
# across reruns, which is what makes "regenerate every reported number" checkable
# with a diff rather than by eye.
#
# The synthetic LiDAR fixtures are generated (not committed) and carry exact
# ground truth, so they are rebuilt here rather than assumed present.

set -uo pipefail

OUT="${1:-benchmark_runs}"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1756728000}"

PY="${PYTHON:-python3}"
mkdir -p "$OUT"
echo "capture,tier,seconds,exit_code" > "$OUT/timing.csv"

# Ray-traced fixtures with exact ground truth (3.60 x 2.80 m, known openings).
$PY tests/fixtures/synthesize.py >/dev/null 2>&1 || true

run_one() {
  local name="$1" input="$2" tier="$3"; shift 3
  [ -d "$input" ] || { echo "  skip $name (no $input)"; return; }
  rm -rf "${OUT:?}/$name"
  local start end secs code
  start=$($PY -c 'import time;print(time.time())')
  $PY -m cozmo.cli run --input "$input" --out "$OUT/$name" --quiet "$@" >"$OUT/$name.log" 2>&1
  code=$?
  end=$($PY -c 'import time;print(time.time())')
  secs=$($PY -c "print(f'{$end-$start:.1f}')")
  echo "$name,$tier,$secs,$code" >> "$OUT/timing.csv"
  printf '  %-24s %-6s %8ss  exit=%s\n' "$name" "$tier" "$secs" "$code"
}

echo "LiDAR tier"
run_one synthetic_room       tests/fixtures/captures/synthetic_room       lidar --no-semantics
run_one synthetic_no_ceiling tests/fixtures/captures/synthetic_no_ceiling lidar --no-semantics
run_one apartment_lidar      captures/apartment_lidar                     lidar --no-semantics
run_one scan_with_ceiling    captures/scan_with_ceiling                   lidar --no-semantics
run_one scan_floor_only      captures/scan_floor_only                     lidar --no-semantics

echo "Photo tier"
run_one demo_office          captures/demo_office        photo --backbone vggt --no-semantics
run_one saurabh_room         captures/saurabh_room       photo --backbone vggt --no-semantics
run_one saurabh_room_photo   captures/saurabh_room_photo photo --backbone vggt --no-semantics
run_one demo_fourroom        captures/demo_fourroom      photo --backbone vggt --no-semantics

echo "Video tier"
run_one apartment_video      captures/apartment_video    video --backbone vggt --no-semantics
run_one saurabh_room_video   captures/saurabh_room_video video --backbone vggt --no-semantics

echo
echo "Scoring against exact (ray-traced) ground truth"
$PY -m cozmo.cli benchmark \
    --results "$OUT" \
    --ground-truth tests/fixtures/benchmark/ground_truth_synthetic.csv \
    --out "$OUT" | tee "$OUT/gate_table.txt"

echo
echo "timing:  $OUT/timing.csv"
echo "results: $OUT/results.json"
