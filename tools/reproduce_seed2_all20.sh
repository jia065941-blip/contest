#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
IMAGE=${IMAGE:-contest-final-repro:latest}
WORKERS=${WORKERS:-20}
STAMP=${STAMP:-reproduced}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/reproduced_results}
OUTPUT=$OUTPUT_ROOT/cz_ppo_seed2_final_all20_$STAMP

mkdir -p "$OUTPUT_ROOT"
if [ -e "$OUTPUT" ]; then
    echo "Refusing to overwrite existing output: $OUTPUT" >&2
    exit 2
fi

docker build -t "$IMAGE" "$ROOT"
docker run --rm \
    -v "$OUTPUT_ROOT:/results" \
    "$IMAGE" \
    python -B tools/run_cz_ppo_seed2_20.py \
    --out "/results/cz_ppo_seed2_final_all20_$STAMP" \
    --workers "$WORKERS"

echo "Completed reproduction: $OUTPUT"
