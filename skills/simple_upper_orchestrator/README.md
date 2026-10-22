# Simple Upper Orchestrator

This package wires the four pure skills into the unmodified 2026-10-21 official
release. It installs a process-local replacement for only the exported
`user_agents.red.RedAgentFactory`, calls the official `prepare_run(args)` and
`run_simulation(args, bundle)`, and restores the export afterward.

The adapter uses `local_avoidance_skill.load_external_policy`; the runtime and
checkpoint are caller-owned. It does not import or depend on archived
`local_avoid_v13` code. H/M assignments use the initial public 9400/9600
directory. Dynamic target coordinates remain owner-local. L state is scoped by
the current `self.commRangeInfo` group, satellite use is globally single-shot,
and a 9500 attack is emitted only when that ID is in the command owner's current
`detectInfo`.

## Install

Install this package and its four sibling packages in the Python environment
used by the private-runtime launcher:

```bash
python -m pip install -e ../staggered_deployment \
  -e ../l_satellite_strike_packages \
  -e ../hm_public_strike_packages \
  -e ../local_avoidance \
  -e .
```

## Run

No preset deployment plan is accepted by this adapter. The staggered planner
validates that the official L and H/M deployment areas are the same single
axis-aligned rectangle and fails closed otherwise.

```bash
simple-upper-release \
  --competition-root /home/amax/ry/competition/competition-platform-env-release20261021 \
  --runtime-root /home/amax/ry/competition/jmy_ppo/contest-jmy \
  --checkpoint /home/amax/ry/competition/jmy_ppo/contest-jmy/models/local_avoid_set_attention_v12_r3_timely_arrival.pt \
  --scenario launch20/easy/E01 \
  --seed 701 \
  --max-steps 3000 \
  --render-mode none \
  --output-root /home/amax/ry/competition/results/simple-upper \
  --run-id E01-s701 \
  --blue-policy b0_fixed_ratio_random \
  --blue-interceptor-ratio 2 \
  --deploy-front-axis east \
  --deploy-gap-km 30 \
  --deploy-l-depth-km 5 \
  --deploy-hm-depth-km 10 \
  --deploy-spacing-km 1 \
  --deploy-batch-size 10 \
  --l-satellite-trigger-step 1 \
  --l-package-count 10 \
  --l-reserve-count 0 \
  --l-expected-target-count 10 \
  --hm-safety-factor 1.25 \
  --hm-reserve-h 0 \
  --hm-reserve-m 0 \
  --log-frequency 1 \
  --enable-ai-action
```

All official output and logging switches remain available, including
`--batch-size`, `--verbose`, `--enable-config`, `--enable-state`,
`--enable-event`, and `--disable-log-color`. Run `simple-upper-release --help`
for the combined official and adapter parser.

`--runtime-root` and `--checkpoint` default to the current contest-jmy checkout
and checkpoint paths, but no model class or checkpoint hash is fixed by this
package. Set `SIMPLE_UPPER_GLIBC_ROOT` to override the default private glibc 2.38
location. The launcher forces OpenMP, MKL, OpenBLAS, NumExpr, BLIS, vecLib, and
Torch thread counts to one. For a caller-provided JSON profile, legacy
`"commRangeInfo": []` values are migrated into a generated copy under the run
output directory; the source and official checkout are never modified.

## Test

```bash
python -m pytest -q
python -m compileall -q src tests
```
