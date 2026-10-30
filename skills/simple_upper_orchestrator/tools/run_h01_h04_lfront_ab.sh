#!/usr/bin/env bash
set -euo pipefail

RUNNER="${RUNNER:-/home/amax/miniconda3/envs/competition/bin/simple-upper-release}"
RELEASE_ROOT="${RELEASE_ROOT:-/home/amax/ry/competition/competition-platform-env-release20261021}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/home/amax/ry/competition/jmy_ppo/contest-jmy}"
CHECKPOINT="${CHECKPOINT:-/home/amax/ry/competition/skill_generation/skills/local_avoidance/net/local_avoid_set_attention_v12_r3_timely_arrival.pt}"
OUTPUT_ROOT="${SIMPLE_UPPER_OUTPUT_ROOT:-/home/amax/ry/competition/skill_generation/skills/simple_upper_orchestrator/results/h01_h04_v12_lfront_ab_s1}"
MAX_JOBS="${MAX_JOBS:-8}"
SEED="${SEED:-1}"
export RUNNER RELEASE_ROOT RUNTIME_ROOT CHECKPOINT OUTPUT_ROOT SEED
export SIMPLE_UPPER_THREADS="${SIMPLE_UPPER_THREADS:-1}"
unset LOCAL_AVOID_GATE_MODE || true

mkdir -p "$OUTPUT_ROOT/logs"
for scenario in launch20/hard/H{01..04}; do
  printf '%s\t%s\t%s\n' "$scenario" front100 1.0
  printf '%s\t%s\t%s\n' "$scenario" front80 0.8
done |
xargs -P "$MAX_JOBS" -n 3 bash -c '
  set -euo pipefail
  scenario="$1"
  variant="$2"
  l_fraction="$3"
  case_id="${scenario##*/}"
  run_id="launch20_hard_${case_id}_v12r3_${variant}_s${SEED}"
  "$RUNNER" \
    --competition-root "$RELEASE_ROOT" \
    --runtime-root "$RUNTIME_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --scenario "$scenario" \
    --seed "$SEED" \
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
    --deploy-l-first-wave-fraction "$l_fraction" \
    --hm-allocation-mode full_coverage_mixed \
    --hm-min-h-per-mother 2 \
    --hm-max-h-per-mother 2 \
    --hm-h-per-sentry-package 2 \
    --hm-m-per-h-mixed-package 2 \
    --hm-m-only-sentry-package 6 \
    --l-package-count 20 \
    --l-expected-target-count 20 \
    --l-first-wave-per-target 9 \
    --reserve-h-count 6 \
    --reserve-deadline-step 450 \
    --reserve-l-deadline-step 250 \
    --reserve-h-max-distance 80 \
    --reserve-l-max-distance 100 \
    --reserve-h-turn-rate 20 \
    --reserve-l-turn-rate 15 \
    --reserve-turn-safety-seconds 5 \
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
