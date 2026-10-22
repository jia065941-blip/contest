#!/usr/bin/env bash
set -euo pipefail

RUNNER="${RUNNER:-/home/amax/miniconda3/envs/competition/bin/simple-upper-release}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/amax/ry/competition/skill_generation/skills/simple_upper_orchestrator/results/e01_extra_threads}"
SEED="${SEED:-101}"
export SIMPLE_UPPER_THREADS="${SIMPLE_UPPER_THREADS:-4}"
mkdir -p "$OUTPUT_ROOT/logs"
run_id="launch20_easy_E01_simple_upper_s${SEED}_t${SIMPLE_UPPER_THREADS}"

"$RUNNER" \
  --competition-root /home/amax/ry/competition/competition-platform-env-release20261021 \
  --runtime-root /home/amax/ry/competition/jmy_ppo/contest-jmy \
  --checkpoint /home/amax/ry/competition/skill_generation/skills/local_avoidance/net/local_avoid_set_attention_v12_r3_timely_arrival.pt \
  --scenario launch20/easy/E01 \
  --seed "$SEED" \
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
  --l-satellite-trigger-step 1 \
  --l-package-count 10 \
  --l-expected-target-count 10 \
  --hm-safety-factor 1.25 \
  --disable-log-color \
  >"$OUTPUT_ROOT/logs/$run_id.log" 2>&1
