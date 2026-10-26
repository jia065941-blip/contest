#!/usr/bin/env bash
set -euo pipefail

RUNNER="${RUNNER:-/home/amax/miniconda3/envs/competition/bin/simple-upper-release}"
RELEASE_ROOT="${RELEASE_ROOT:-/home/amax/ry/competition/competition-platform-env-release20261021}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/home/amax/ry/competition/jmy_ppo/contest-jmy}"
CHECKPOINT="${CHECKPOINT:-/home/amax/ry/competition/skill_generation/skills/local_avoidance/net/local_avoid_set_attention_v12_r3_timely_arrival.pt}"
OUTPUT_ROOT="${SIMPLE_UPPER_OUTPUT_ROOT:-/home/amax/ry/competition/skill_generation/skills/simple_upper_orchestrator/results/e01_e04_v12_reachable_reserve_s1_s2}"
MAX_JOBS="${MAX_JOBS:-8}"
export RUNNER RELEASE_ROOT RUNTIME_ROOT CHECKPOINT OUTPUT_ROOT
export SIMPLE_UPPER_THREADS="${SIMPLE_UPPER_THREADS:-1}"
unset LOCAL_AVOID_GATE_MODE || true

mkdir -p "$OUTPUT_ROOT/logs"
for scenario in launch20/easy/E{01..04}; do
  for seed in 1 2; do
    printf '%s\t%s\n' "$scenario" "$seed"
  done
done |
xargs -P "$MAX_JOBS" -n 2 bash -c '
  set -euo pipefail
  scenario="$1"
  seed="$2"
  case_id="${scenario##*/}"
  run_id="launch20_easy_${case_id}_v12r3_reachable_reserve_s${seed}"
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
    --deploy-sentry-rendezvous-mode adaptive \
    --deploy-h-speed 200 \
    --deploy-m-speed 120 \
    --deploy-m-arrival-lead-seconds 5 \
    --deploy-min-m-lead 10 \
    --deploy-mother-crossfire-angle 35 \
    --hm-allocation-mode full_coverage_mixed \
    --hm-min-h-per-mother 2 \
    --hm-max-h-per-mother 2 \
    --hm-h-per-sentry-package 2 \
    --hm-m-per-h-mixed-package 2 \
    --hm-m-only-sentry-package 6 \
    --l-package-count 10 \
    --l-expected-target-count 10 \
    --l-first-wave-per-target 7 \
    --deploy-l-first-wave-fraction 0.824 \
    --reserve-h-count 2 \
    --reserve-deadline-step 450 \
    --reserve-h-max-distance 80 \
    --reserve-l-max-distance 100 \
    --reserve-h-max-turn 60 \
    --reserve-l-max-turn 75 \
    --reserve-holding-half-width 6 \
    --reserve-holding-heading-offset 35 \
    --reserve-holding-max-turn-steps 6 \
    --reserve-holding-straight-steps 2 \
    --disable-dynamic-retarget \
    --disable-terminal-guidance \
    --formation-audit-interval 25 \
    --enable-event \
    --disable-log-color \
    >"$OUTPUT_ROOT/logs/$run_id.log" 2>&1
' _
