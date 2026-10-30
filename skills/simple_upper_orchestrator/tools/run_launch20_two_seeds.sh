#!/usr/bin/env bash
set -euo pipefail

RUNNER="${RUNNER:-/home/amax/miniconda3/envs/competition/bin/simple-upper-release}"
RELEASE_ROOT="${RELEASE_ROOT:-/home/amax/ry/competition/competition-platform-env-release20261021}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/home/amax/ry/competition/jmy_ppo/contest-jmy}"
CHECKPOINT="${CHECKPOINT:-/home/amax/ry/competition/skill_generation/skills/local_avoidance/net/local_avoid_set_attention_v12_r3_timely_arrival.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/amax/ry/competition/skill_generation/skills/simple_upper_orchestrator/results/launch20_s1_s2}"
MAX_JOBS="${MAX_JOBS:-40}"
export RUNNER RELEASE_ROOT RUNTIME_ROOT CHECKPOINT OUTPUT_ROOT
export SIMPLE_UPPER_THREADS="${SIMPLE_UPPER_THREADS:-1}"

mkdir -p "$OUTPUT_ROOT/logs"
printf '%s\n' \
  launch20/easy/E{01..06} \
  launch20/medium/M{01..06} \
  launch20/hard/H{01..08} |
while read -r scenario; do
  for seed in 1 2; do
    printf '%s\t%s\n' "$scenario" "$seed"
  done
done |
xargs -P "$MAX_JOBS" -n 2 bash -c '
  set -euo pipefail
  scenario="$1"
  seed="$2"
  case_id="${scenario##*/}"
  difficulty="$(basename "$(dirname "$scenario")")"
  case "$difficulty" in
    easy) dark_nests=10; l_first=7 ;;
    medium) dark_nests=15; l_first=7 ;;
    hard) dark_nests=20; l_first=9 ;;
    *) printf "Unknown difficulty: %s\n" "$difficulty" >&2; exit 2 ;;
  esac
  run_id="launch20_${difficulty}_${case_id}_simple_upper_s${seed}"
  "$RUNNER" \
    --competition-root "$RELEASE_ROOT" \
    --runtime-root "$RUNTIME_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --scenario "$scenario" \
    --seed "$seed" \
    --render-mode none \
    --output-root "$OUTPUT_ROOT" \
    --run-id "$run_id" \
    --blue-policy b0_fixed_ratio_random \
    --blue-interceptor-ratio 2 \
    --deploy-front-axis east \
    --deploy-gap-km 30 \
    --deploy-l-depth-km 5 \
    --deploy-hm-depth-km 10 \
    --deploy-spacing-km 1 \
    --deploy-l-first-wave-fraction 1 \
    --deploy-batch-size 10 \
    --l-satellite-trigger-step 1 \
    --l-package-count "$dark_nests" \
    --l-expected-target-count "$dark_nests" \
    --l-first-wave-per-target "$l_first" \
    --l-reserve-count 0 \
    --reserve-l-deadline-step 250 \
    --hm-safety-factor 1.25 \
    --hm-reserve-h 0 \
    --hm-reserve-m 0 \
    --disable-log-color \
    >"$OUTPUT_ROOT/logs/$run_id.log" 2>&1
' _
