#!/usr/bin/env bash
set -euo pipefail

RUNNER="${RUNNER:-/home/amax/miniconda3/envs/competition/bin/simple-upper-release}"
RELEASE_ROOT="${RELEASE_ROOT:-/home/amax/ry/competition/competition-platform-env-release20261021}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/home/amax/ry/competition/jmy_ppo/contest-jmy}"
NET_ROOT="${NET_ROOT:-/home/amax/ry/competition/skill_generation/skills/local_avoidance/net}"
OUTPUT_ROOT="${SIMPLE_UPPER_OUTPUT_ROOT:-/home/amax/ry/competition/skill_generation/skills/simple_upper_orchestrator/results/launch20_pulse_guidance_two_networks_s1_s2}"
MAX_JOBS="${MAX_JOBS:-40}"
export RUNNER RELEASE_ROOT RUNTIME_ROOT OUTPUT_ROOT
export SIMPLE_UPPER_THREADS="${SIMPLE_UPPER_THREADS:-1}"
unset LOCAL_AVOID_GATE_MODE || true

networks=(
  "v13|$NET_ROOT/local_avoid_ppo_v13_gate_aligned_s3_best.pt"
  "v12r3|$NET_ROOT/local_avoid_set_attention_v12_r3_timely_arrival.pt"
)
mkdir -p "$OUTPUT_ROOT/logs"
for network in "${networks[@]}"; do
  label="${network%%|*}"
  checkpoint="${network#*|}"
  for scenario in \
    launch20/easy/E{01..06} \
    launch20/medium/M{01..06} \
    launch20/hard/H{01..08}; do
    for seed in 1 2; do
      printf '%s\t%s\t%s\t%s\n' "$label" "$checkpoint" "$scenario" "$seed"
    done
  done
done |
xargs -P "$MAX_JOBS" -n 4 bash -c '
  set -euo pipefail
  label="$1"
  checkpoint="$2"
  scenario="$3"
  seed="$4"
  case_id="${scenario##*/}"
  difficulty="$(basename "$(dirname "$scenario")")"
  case "$difficulty" in
    easy) dark_nests=10; l_first=7 ;;
    medium) dark_nests=15; l_first=7 ;;
    hard) dark_nests=20; l_first=9 ;;
    *) printf "Unknown difficulty: %s\n" "$difficulty" >&2; exit 2 ;;
  esac
  if [ "$label" = "v13" ]; then mother_h=3; else mother_h=2; fi
  run_id="launch20_${difficulty}_${case_id}_${label}_pulse_guidance_s${seed}"
  "$RUNNER" \
    --competition-root "$RELEASE_ROOT" \
    --runtime-root "$RUNTIME_ROOT" \
    --checkpoint "$checkpoint" \
    --scenario "$scenario" \
    --seed "$seed" \
    --render-mode none \
    --output-root "$OUTPUT_ROOT" \
    --run-id "$run_id" \
    --blue-policy b0_fixed_ratio_random \
    --blue-interceptor-ratio 2 \
    --deploy-front-axis east \
    --deploy-gap-km 15 \
    --deploy-l-depth-km 5 \
    --deploy-hm-depth-km 10 \
    --deploy-h-depth-km 5 \
    --deploy-m-depth-km 10 \
    --deploy-hm-follow-gap-km 0 \
    --deploy-mother-crossfire-angle 35 \
    --deploy-mother-crossfire-max-offset-km 35 \
    --deploy-sentry-m-lead-km 7.5 \
    --deploy-sentry-rendezvous-mode adaptive \
    --deploy-h-speed 200 \
    --deploy-m-speed 120 \
    --deploy-m-arrival-lead-seconds 5 \
    --deploy-min-m-lead 10 \
    --deploy-spacing-km 1 \
    --deploy-l-first-wave-fraction 1 \
    --deploy-batch-size 10 \
    --l-satellite-trigger-step 1 \
    --l-allocation-mode distance_balanced \
    --l-package-count "$dark_nests" \
    --l-expected-target-count "$dark_nests" \
    --l-first-wave-per-target "$l_first" \
    --l-reserve-count 0 \
    --reserve-l-deadline-step 250 \
    --hm-allocation-mode deterministic_thresholds \
    --hm-target-kill-probability 0.85 \
    --hm-min-h-per-mother "$mother_h" \
    --hm-min-m-per-mother 0 \
    --hm-min-h-per-sentry 1 \
    --hm-max-h-per-mother "$mother_h" \
    --hm-h-per-sentry-package 2 \
    --hm-m-per-h-mixed-package 2 \
    --hm-m-only-sentry-package 6 \
    --hm-m-target-policy complete_h_sentries \
    --hm-target-priority-axis east \
    --dynamic-retarget-threshold 0.70 \
    --dynamic-retarget-max-stage 3 \
    --dynamic-retarget-min-distance 15 \
    --dynamic-retarget-cooldown 25 \
    --dynamic-retarget-max-retarget 1 \
    --dynamic-retarget-max-fraction 0.20 \
    --dynamic-retarget-distance-weight 0.05 \
    --dynamic-retarget-track-max-age 3 \
    --dynamic-retarget-max-target-separation 10 \
    --dynamic-retarget-max-turn-angle 45 \
    --dynamic-retarget-max-candidate-distance 30 \
    --dynamic-retarget-max-per-old-target 1 \
    --disable-dynamic-retarget \
    --terminal-min-stage 4 \
    --terminal-capture-distance 2000 \
    --terminal-final-straight-distance 300 \
    --terminal-max-track-age 5 \
    --terminal-stale-grace-age 8 \
    --terminal-threat-distance 3000 \
    --terminal-heading-enter 6 \
    --terminal-heading-exit 10 \
    --terminal-predicted-miss 80 \
    --terminal-turn-pulse-steps 1 \
    --terminal-straight-recovery-steps 2 \
    --terminal-no-improvement-limit 3 \
    --enable-terminal-guidance \
    --velocity-window-steps 5 \
    --seconds-per-step 1 \
    --hm-reserve-h 0 \
    --hm-reserve-m 0 \
    --enable-event \
    --disable-log-color \
    >"$OUTPUT_ROOT/logs/$run_id.log" 2>&1
' _
