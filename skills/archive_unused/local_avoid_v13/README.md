# local_avoid_v13

Independent V13 local-avoidance Skill built on the shared `skill-generation`
Runtime. It contains a trusted Actor Backend, verified Actor-only model bundle,
33-feature encoder, `mission_v12` gate and explicit adapter for the unmodified
official `competition_envs` checkout.

## Layout

```text
skills/local_avoid_v13/
├── bin/local-avoid-v13-glibc238
├── pyproject.toml
├── src/local_avoid_v13_skill/
│   ├── backend.py
│   ├── encoder.py
│   ├── gate.py
│   ├── policy.py
│   ├── official_entry.py
│   └── bundle/
├── tests/
└── tools/
```

Future concrete Skills should be sibling directories under top-level `skills/`.
They depend on the shared root package instead of copying Runtime code.

## Runtime Environment

The verified numerical environment is Python 3.11.17, NumPy 2.4.6 and PyTorch
2.6.0+cpu at `/home/amax/miniconda3/envs/competition`. This environment can run
the Actor directly. The official native simulator also needs glibc 2.38. An
installed `local-avoid-v13` command switches only its own process to the existing
private loader automatically; source checkouts can use
`bin/local-avoid-v13-glibc238`. Direct Conda Python on host glibc 2.35 cannot load
`_UtilsPy.so`.

Optional installation into the Conda environment is:

```bash
/home/amax/miniconda3/envs/competition/bin/python -m pip install --no-deps \
  /home/amax/ry/competition/skill_generation
/home/amax/miniconda3/envs/competition/bin/python -m pip install --no-deps \
  /home/amax/ry/competition/skill_generation/skills/local_avoid_v13
```

The launcher also supports source execution without modifying the Conda
environment and is the method used for acceptance.

## Model Safety

The source checkpoint SHA-256 is:

```text
632662d47fb35edfdb10d680a359cced03fab75af09cff666631a97f0ccf87f3
```

`tools/export_v13_bundle.py` requires that exact digest and uses only
`torch.load(..., weights_only=True)` with a minimal NumPy array allowlist. The
bundle contains Actor weights, actor RMS and JSON configuration. It does not
contain pickle, Critic weights, optimizer state or training code.

## Direct Inference

```python
import numpy as np
from local_avoid_v13_skill import LocalAvoidV13Skill

with LocalAvoidV13Skill() as skill:
    result = skill.infer({"observation": np.zeros(33, dtype=np.float32)})
    action = int(np.argmax(result.outputs["logits"]))
```

Actions are `0=left`, `1=straight/stop`, `2=right`. The environment adapter maps
them to `-20G`, `0`, `+20G`.

## Official Environment

R9 is already the shared upper-layer commander. To reproduce its complete
Final20 preset, select a Final20 case and pass that case's fixed grid deployment
with `--deployment-plan preset`. New `InferenceSkill` chain:

```bash
cd /home/amax/ry/competition/skill_generation/skills/local_avoid_v13
for CASE in E01 E02 E03; do
  bin/local-avoid-v13-glibc238 run \
    --competition-root /home/amax/ry/competition/competition_envs \
    --scenario "final20/easy/$CASE" \
    --deployment-plan preset \
    --implementation skill \
    --blue-policy b0_fixed_ratio_random \
    --blue-interceptor-ratio 2 \
    --red-policy r9_hierarchical_learning \
    --seed 701 \
    --run-id "new_${CASE}_s701" \
    --output-root "$PWD/results" \
    --total-rounds 1 \
    --render-mode none
done
```

Original-author interface on the same official environment and scenarios:

```bash
cd /home/amax/ry/competition/skill_generation/skills/local_avoid_v13
for CASE in E01 E02 E03; do
  bin/local-avoid-v13-glibc238 run \
    --competition-root /home/amax/ry/competition/competition_envs \
    --scenario "final20/easy/$CASE" \
    --deployment-plan preset \
    --implementation original \
    --jmy-contest-root /home/amax/ry/competition/jmy_ppo/contest \
    --checkpoint /home/amax/ry/competition/jmy_ppo/contest/models/local_avoid_ppo_v13_gate_aligned_s3_best.pt \
    --blue-policy b0_fixed_ratio_random \
    --red-policy r9_hierarchical_learning \
    --seed 701 \
    --run-id "old_${CASE}_s701" \
    --output-root "$PWD/results" \
    --total-rounds 1 \
    --render-mode none
done
```

Both modes use the same Final20 scenario, matching fixed deployment plan, R9
commander, `mission_v12` gate, blue policy, seed, engine and score code. The
`original` bridge loads the author's policy source
read-only and replaces its unsafe checkpoint loader with the same verified
`weights_only=True` payload.

The simulator policy disables per-inference Runtime INFO records by default to
avoid hundreds of thousands of JSON log lines. Runtime counters, validation and
exceptions remain active. Direct `LocalAvoidV13Skill` users can still supply a
normal `Monitor` when detailed call logging is required.

## PlanSpec-v2 Upper Layer

`local-avoid-v13-plan` uses the authorized, read-only Plan Anchor release as the
upper layer instead of R9. It runs the release's deterministic PlanSpec-v2,
Unified Agent, scenario snapshot and historical explicit-launch engine while
switching only the V13 numerical implementation between `original` and `skill`.
No file in the Plan Anchor checkout is changed.

```bash
cd /home/amax/ry/competition/skill_generation/skills/local_avoid_v13

PLAN_ROOT=/home/amax/ry/competition/plan-anchor-final20-v2
RUNNER="$PWD/.venv/bin/local-avoid-v13-plan"

for IMPLEMENTATION in original skill; do
  "$RUNNER" \
    --plan-root "$PLAN_ROOT" \
    --scenario final20/easy/E01 \
    --plan A00_balanced \
    --implementation "$IMPLEMENTATION" \
    --blue-policy b0_fixed_ratio_random \
    --seed 701 \
    --run-id "planspec_E01_A00_${IMPLEMENTATION}_s701" \
    --output-root "$PWD/results" \
    --total-rounds 1 \
    --render-mode none
```

The plan can also be an explicit JSON path. For E02/E03 use the matching
`final20/easy/E02` or `E03`; the short name resolves only within that scenario's
plan directory. The adapter fail-closes on PlanSpec context mismatches.

## Verification

```bash
.venv/bin/python -I -m pytest
.venv/bin/python tools/compare_original.py \
  --contest-root /home/amax/ry/competition/jmy_ppo/contest \
  --checkpoint /home/amax/ry/competition/jmy_ppo/contest/models/local_avoid_ppo_v13_gate_aligned_s3_best.pt \
  --bundle src/local_avoid_v13_skill/bundle
.venv/bin/python tools/compare_encoder.py \
  --contest-root /home/amax/ry/competition/jmy_ppo/contest \
  --bundle src/local_avoid_v13_skill/bundle
.venv/bin/python tools/compare_gate.py \
  --contest-root /home/amax/ry/competition/jmy_ppo/contest \
  --bundle src/local_avoid_v13_skill/bundle
```
