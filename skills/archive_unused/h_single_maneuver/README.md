# H Single Maneuver Skill

Deterministic single-H-missile controller composed from the sibling
`turn-left-skill`, `stop-turning-skill`, and `turn-right-skill` packages.
The H controller itself has no checkpoint, neural-network runtime, or dependency
on another complex skill. The comparison runner can separately load the
published local V13 runtime for M/L or for the all-V13 control arm.

## Rule

One `HSingleManeuverSkill` instance owns exactly one H missile. Every output is
produced by calling one of the three child skills.

1. Default to `StopTurningSkill`.
2. Project each live type-24000 track to a local east/north frame.
3. Trigger a break only when range is within the emergency threshold, or when
   closing speed, time to CPA, and distance at CPA jointly indicate a conflict.
4. Lock the escape side briefly so bearing noise cannot cause left/right
   oscillation.
5. Clear lateral acceleration after the short break.
6. Recover toward the assigned goal in bounded one-step pulses, then return to
   zero lateral acceleration.
7. Tighten the CPA threshold inside the terminal 12 km so target completion
   wins over marginal threats.

The default profile is `src/h_single_maneuver_skill/profiles/default.json`.
It is intentionally STOP-dominant because archived evaluations and local
trajectory diagnostics show that frequent switching sharply reduces score.

## Python API

```python
from h_single_maneuver_skill import (
    HMissileSnapshot,
    HSingleManeuverSkill,
    ThreatTrack,
    Vector2,
)

skill = HSingleManeuverSkill()
decision = skill.step(
    HMissileSnapshot(
        step=10,
        missile_id=2576,
        position=Vector2(0.0, 0.0),
        velocity=Vector2(2000.0, 0.0),
        goal_position=Vector2(100000.0, 0.0),
        threats=(
            ThreatTrack(
                entity_id=2800,
                position=Vector2(12000.0, 1000.0),
                velocity=Vector2(-2000.0, 0.0),
            ),
        ),
    )
)
engine_row = [0, 2576, decision.engine_acc_z, 0.0]
```

Call `reset()` before reusing an instance for a new episode. Reusing one
instance for a different missile ID or replaying a step is rejected.

## Official Final20 Run

Install this package into the shared development environment:

```bash
cd /home/amax/ry/competition/skill_generation
.venv/bin/python -m pip install --no-deps -e skills/h_single_maneuver
```

The runner reads scenarios and native libraries from the unmodified official
`competition_envs` checkout, uses B0 with two shots per target, and loads the
map-specific PlanSpec-v2 upper plan selected by `--plan auto`:

```bash
cd /home/amax/ry/competition/skill_generation/skills/h_single_maneuver
../../.venv/bin/h-single-maneuver-plan \
  --environment-root /home/amax/ry/competition/competition_envs \
  --plan-root /home/amax/ry/competition/plan-anchor-final20-v2 \
  --scenario final20/easy/E01 \
  --plan auto \
  --lower-policy hybrid \
  --blue-policy b0_fixed_ratio_random \
  --blue-interceptor-ratio 2 \
  --seed 1 \
  --run-id e01_h_single_s1 \
  --output-root "$PWD/results" \
  --render-mode none \
  -- --disable-log-color
```

`auto` selects:

- E01: `A03_deploy_compact`
- E02: `A10_l_screen`
- E03: `A08_formation_dense`

The current official missile models auto-launch during reset and ignore the
planner's launch command. The runner therefore performs two process-local,
auditable compatibility transforms without editing `competition_envs`:

- converts PlanSpec launch rows to supported `CHANGE_TARGET` rows;
- compiles delayed public waves to step zero and concentrates public H/M
  salvos for the B0 two-shot rule.
- gives the H child skill exact one-second calls while reproducing the local
  V13 policy's original 10-step pulse internally for M/L and the control arm.

Disable these transforms with `--no-auto-launch-retarget` and
`--no-auto-launch-salvos` when using an older explicit-launch engine.

There is no official CLI/environment/scenario parameter that enables physical
planned-wave launch in the current checkout. H/M/L call native `Launch()` from
`set_lla()` and do not handle command 200 (`MISSILE_LAUNCH`). Real wave launch
therefore requires simulator source changes or equivalent child-process code
injection; disabling the compatibility transforms does not enable it.

This runner provides the explicit opt-in `--launch-mode patched-waves` for
controlled comparison. It patches the three simulator classes only inside the
child process: initialization auto-launch is suppressed, command 200 calls the
native `Launch()` method, and the original PlanSpec release steps are retained.
No file in `competition_envs` is modified. The default remains
`--launch-mode official-auto`.

## H-Only Comparison

The two modes keep scenario, PlanSpec, seed, B0, checkpoint, launch transforms,
maximum steps, and local-policy pulse timing identical:

- `hybrid`: H uses `HSingleManeuverSkill`; M/L use local V13.
- `local-all`: H/M/L all use local V13.

```bash
cd /home/amax/ry/competition/skill_generation/skills/h_single_maneuver

COMMON=(
  --environment-root /home/amax/ry/competition/competition_envs
  --plan-root /home/amax/ry/competition/plan-anchor-final20-v2
  --scenario final20/easy/E01
  --plan auto
  --launch-mode patched-waves
  --blue-policy b0_fixed_ratio_random
  --blue-interceptor-ratio 2
  --seed 1
  --output-root "$PWD/results"
  --render-mode none
)

../../.venv/bin/h-single-maneuver-plan \
  "${COMMON[@]}" \
  --lower-policy hybrid \
  --run-id compare_E01_h_rule_ml_v13_s1 \
  -- --disable-log-color

../../.venv/bin/h-single-maneuver-plan \
  "${COMMON[@]}" \
  --lower-policy local-all \
  --run-id compare_E01_all_v13_s1 \
  -- --disable-log-color
```

The default checkpoint is
`plan-anchor-final20-v2/github_release/plan_anchor_final20_v2/snapshot/models/local_avoid_ppo_v13_gate_aligned_s3_best.pt`.
Override it with `--local-checkpoint PATH` only when both arms use the same
replacement file.

For an older Final20 checkout that already implements command-gated launch,
use `--launch-mode native-waves` and point `--environment-root` at that tree.
The retained compatible checkout is:

```text
/home/amax/ry/competition/personal_train_ppo/competition-platform-env
```

The older `/home/amax/ry/competition/jmy_ppo/contest` working tree cannot be
combined with these PlanSpec E01 files: its checked-in `easy/E01` has
30H/44M/90L while the plan contract requires 17H/24M/85L.

Each run writes:

- `<output>/<run-id>/<timestamp>/summary.json`: official weighted-damage score;
- `<output>/<run-id>/<timestamp>/hit_events.jsonl`: post-episode global-truth
  hit, angle, adjusted-rate, applied-damage, and before/after-health diagnostics;
- `<output>/<run-id>/<timestamp>/analysis_summary.json`: aggregate hit/damage
  diagnostics. Both analysis files are explicitly unavailable to online policy;
- `<output>/<run-id>/maneuver_decisions.jsonl`: H-only per-step action, phase,
  goal error, selected threat, range, closing speed, TCPA, and DCPA;
- `<output>/<run-id>/effective_plan.json`: compiled plan when using the official
  auto-launch compatibility mode; patched-wave mode uses the source plan.

## Measured Validation

Before adding the M/L comparison dispatcher, the H rule was tuned with M/L held
neutral on the current official checkout, B0 ratio 2, seed 1, the official
3000-step horizon, and the map-specific `auto` plan:

| Scenario | Plan | Score | Steps | H turn share |
|---|---|---:|---:|---:|
| `final20/easy/E01` | `A03_deploy_compact` | 13.793103 | 1801 | 5.87% |
| `final20/easy/E02` | `A10_l_screen` | 10.344828 | 1873 | 7.92% |
| `final20/easy/E03` | `A08_formation_dense` | 13.793103 | 1801 | 4.21% |

The decision traces contain 17 independently controlled H missiles in every
case. Long or repeatedly re-armed turns were also sampled; they produced large
goal errors and no public-target gain, so the bounded STOP-dominant profile was
retained.

These numbers must not be compared directly with the publication report. The
published planner was evaluated with a matching explicit-launch simulator that
is not included in the bundle, while the current official native missile models
auto-launch and ignore launch commands. In the current checkout all scored
damage in these three replays came from hidden 9500 responses; no public
9400/9600 target was damaged. The runner records this limitation instead of
claiming reproduction of the publication scores.

### Patched-Wave Validation

With `--launch-mode patched-waves`, E01 H control began in three groups at
steps `1`, `121`, and `241`, matching PlanSpec release steps `0`, `120`, and
`240` after the normal one-step observation delay. Full seed-1 results were:

| Lower policy | Score | Steps | Red alive |
|---|---:|---:|---:|
| H rule, M/L local V13 | 13.793103 | 1852 | 0 |
| H/M/L local V13 | 12.643678 | 3000 | 12 |

Public 9400/9600 objectives were still undamaged. This demonstrates that the
current checkout's low score is not caused only by the missing physical wave
gate; its native physics/interception/runtime also differ materially from the
environment used by the PlanSpec publication.

### Old Final20 + PlanSpec Comparison

Using the retained command-gated Final20 checkout, E01/A03, B0, and seed 701:

| Lower policy | Score | Red alive | H turn share |
|---|---:|---:|---:|
| H rule, M/L local V13 | 13.793103 | 1 | 4.96% |
| H/M/L local V13 | 12.643678 | 10 | n/a |

Both arms used plan ID
`8e8fdde06d1e8b7698aafd38d332f64865cbf0527313c7f5a1948d1947405521`
and emitted all 126 planned launches. H control started in groups of 6/6/5 at
steps 1/121/241. Public targets again received no damage, showing that merely
combining the retained old engine with PlanSpec does not reproduce either the
historical R9/V13 result or the separate PlanSpec publication runtime.

### JMY Old Engine + External PlanSpec Scenario

`/home/amax/ry/competition/jmy_ppo/contest` contains the old command-gated
engine but no longer contains its historical Final20 E01 scenario. Its local
E01 copies have 30H/44M/90L and 30H/42M/108L, so neither satisfies the PlanSpec
17H/24M/85L contract. A controlled comparison was still possible by supplying
the PlanSpec bundle's own E01 scenario as an absolute path.

The old engine's step-0 clock causes L missiles launched at step 0 to expire
immediately, so every wave was shifted by one step with
`--native-wave-start-delay 1`. Both arms therefore used the same derived plan
ID `d9979ab1307d9a6fd4ba810e46117949cfeac21ef62b0dd138132f312be8e1b8`.

| Lower policy | Score | Steps | Red alive | Destroyed |
|---|---:|---:|---:|---|
| H rule, M/L local V13 | 6.896552 | 967 | 0 | 2573, 2575 |
| H/M/L local V13 | 6.896552 | 967 | 0 | 2573, 2575 |

H control began at steps 2/122/242 and used non-zero maneuvers on 1.44% of
recorded H decisions. This engine-plus-external-scenario composition also does
not reproduce the historical 74.47 result; that result depended on the missing
historical Final20 scenario/runtime copy and R9 rather than PlanSpec.

### CZ PKU Checkout + PlanSpec Comparison

The current checkout at `/home/amax/cz/competition-platform-env` is branch
`pku`, commit `2350170`, after the auto-launch change. Its own Final20 E01 is
PlanSpec-compatible, so `--launch-mode patched-waves` was used to restore
physical command-gated release for both arms. E01/A03, B0, and seed 701 gave:

| Lower policy | Score | Red alive | Destroyed |
|---|---:|---:|---|
| H rule, M/L local V13 | 13.793103 | 1 | 168, 2570, 2573, 2575 |
| H/M/L local V13 | 13.793103 | 8 | 168, 2570, 2573, 2575 |

Both used source plan ID
`8e8fdde06d1e8b7698aafd38d332f64865cbf0527313c7f5a1948d1947405521`,
126 launches, and H waves at steps 1/121/241. The H rule used non-zero actions
on 4.96% of 3,686 decisions and consumed up to 97 visible interceptor tracks.
Official score tied exactly; all-V13 retained seven more red platforms.

### Matched Old-PKU Skill Comparison

`/home/amax/ry/competition/competition-platform-env-pku` reproduces the
historical E01 R9/V13 score exactly (`74.473180`, 968 steps). Using this
behavior-matched environment, B0 ratio 2, and seed 701:

| Upper | H rule + M/L V13 | H/M/L V13 | Delta | Red alive |
|---|---:|---:|---:|---|
| R9 | 46.836383 | 74.473180 | -27.636798 | 0 / 0 |
| PlanSpec A03 | 76.436782 | 95.402299 | -18.965517 | 2 / 8 |

The H rule emitted non-zero maneuvers on 4.62% of R9 decisions and 5.09% of
PlanSpec decisions. In the matched runtime this conservative rule consistently
underperformed V13: R9/V13 destroyed four high-value public roots versus one
for the H rule; PlanSpec/V13 destroyed six public roots versus three.

R9 comparisons use the dedicated entry point:

```bash
../../.venv/bin/h-single-maneuver-r9 \
  --environment-root /home/amax/ry/competition/competition-platform-env-pku \
  --jmy-contest-root /home/amax/ry/competition/jmy_ppo/contest \
  --checkpoint /home/amax/ry/competition/jmy_ppo/contest/models/local_avoid_ppo_v13_gate_aligned_s3_best.pt \
  --scenario final20/easy/E01 \
  --lower-policy hybrid \
  --seed 701 \
  --run-id oldpku_r9_h_skill_ml_v13_E01_s701 \
  --output-root "$PWD/results" \
  --render-mode none
```

### Variable H Strike-Package Mode

`--lower-policy joint-h-v13` forms immutable 2-6 H packages from the effective
PlanSpec bindings. Members must have type 21000, the same public target ID, and
planned release steps within `--joint-release-window-steps` (default 10).
Single H missiles and degraded packages remain on direct V13. Packages larger
than `--joint-max-members` (default 6) are split by the coordinator.

The current E01-E03 plans split each four-H target allocation across three
widely separated waves. With a ten-step release window, each map therefore
produces three same-wave two-H packages; later singletons remain direct V13.
Future package planners can supply 3-6 member near-synchronous groups without
changing the lower-skill API.

### L Search And Response Modes

`--lower-policy joint-l-local` keeps every L maneuver on the selected local
network while adding two legal-observation coordinators:

- `LSearchCoordinationSkill`: fixed 4x8 cell ownership, discovery quorum,
  member-loss reassignment requests, and delayed satellite requests.
- `LResponsePackageSkill`: immutable 2-4 member packages per
  `response_group_index`, hidden-target health belief, stop-commitment,
  supplement, and authorized-retarget requests.

`--lower-policy joint-hl-local` enables both H and L package layers. Search
waypoints, launches, retarget commands, and ACC_Z rows remain agent-owned; the L
skills emit status and requests only. Easy plans create eight four-cell search
teams and twenty response packages sized 3/2. Hard plans create eighteen search
teams and thirty-six response packages sized 4/3.

### M Strike Packages

`--lower-policy joint-m-local` enables immutable 2-6 member M packages over the
selected local network. Members must share a public 9400/9600 target and a
ten-step release window. The M package never overrides maneuver actions; it
emits roles, status, conservative stop/supplement requests, and authorized
retarget requests from legal PlanSpec damage belief.

`--lower-policy joint-hml-local` enables H, M, and L package/coordination
layers together. `--mission-replan all-packages` additionally consumes
high-confidence M stop requests for still-unlaunched members while retaining
the staged L-response controller. Supplement and retarget requests remain
trace-only until global allocation arbitration is implemented.

E01/H03 400-step smokes produced five same-wave M pairs and twenty H03 M
packages sized 3/4 respectively. All 4,448 package status rows preserved
`motion_overrides=[]`, proving byte-level maneuver ownership remains with the
new local network. Status is written to `m_strike_status.jsonl`.

Trace files are `l_search_status.jsonl` and `l_response_status.jsonl`. E01 and
H02 400-step smokes activated response packages at step 52 after discovery at
step 31, preserved all local-network actions, and produced no illegal context
or duplicate command.

### Mission Replan L-Response Staging

`--mission-replan l-response` installs an event-driven launch filter above
PlanSpec while leaving R9 untouched. After all required legal discoveries and
the original 20-step delay, MissionReplan v2 releases whole response packages
until at least 50% of each target allocation (minimum three members) is active.
It uses one shared `TargetDamageBeliefSkill`, supplements after ETA-calibrated
timeouts, applies shorter follow-up intervals, and cancels only on confirmed
death or high-confidence 1.25x delivered overkill. Projected in-flight damage
may hold briefly but never permanently cancel a package.

E01/H02 smokes discovered targets at step 31 and granted initial packages at
step 51. E01 v2 launched 28 response L (4+3 per target) instead of all 53 and
held nine later packages. H02 launched 72 response L (four per target) instead
of all 142 and held eighteen later packages. Easy/hard initial supplement
timeouts are 600/800 steps, subsequent checks 250/300 steps, and the hard
release deadline is 900 steps. Decisions and belief provenance are written to
`mission_replan.jsonl`; the pure skill never reads PlanSpec or simulator state.

MissionReplan also supports one-time whole-package cleanup retargeting after a
source target is confirmed or has high-confidence 1.25x delivered overkill.
Only launched, live, pre-terminal members are eligible. H/M select legal
9400/9600 targets and L response packages select legally discovered 9500
targets; candidates must have positive expected and projected HP. Selection is
deterministic by same-type preference, ETA, objective weight, then target ID.
Each package retargets at most once, each destination has a 60-step cooldown,
and at most 25% of live launched inventory (minimum one atomic package) can be
retargeted per 100 steps. Accepted commands atomically transfer the shared
TargetDamageBelief commitment and are audited as `mission_retarget_events`, not
coordinate refreshes.

The coordinator is deliberately sparse:

- V13 remains the final controller on normal steps.
- A pure geometric shadow pass records minimum peer separation and ETA spread.
- V13 is not called early when peer separation stays at least 150 m.
- Only a predicted sub-150 m collision can activate an atomic three-member
  decision, and at most one action may change.
- ETA actuation is disabled in the initial release.

Seed-701 E01-E03 validation produced zero action overrides: minimum predicted
separations were 480.97 m, 3096.74 m, and 370.17 m. Scores were 97.70, 97.70,
and 90.80 versus one-process V13 references 93.68, 91.95, and 91.95. Because
the simulator uses process-object IDs in same-time event ordering and the joint
runs changed no maneuver action, these score deltas are runtime variance, not a
causal joint-skill gain. Multi-seed non-inferiority is required before enabling
ETA synchronization or wider deconfliction.

An explicit `--joint-rule-profile experimental` retains threat-owner and
override-budget guards but widens the prediction horizon to 6 seconds and the
peer trigger to 1,000 m. It requires at least 100 m improvement and 400 m final
separation. Two E01/B0 seed-701 processes scored 100.0 (3 overrides, 688
steps) and 98.850575 (6 overrides, 3,000 steps), versus the immediately
preceding safe-joint and all-V13 pair at 82.183908 with zero overrides. This is
promising but remains a single-seed result in a process-order-sensitive
simulator; `safe` remains the default pending multi-seed E01-E03 validation.

The subsequent seed-1/2/3 E01-E03 matrix did not confirm a general gain:

| Map | All V13 mean | Experimental mean | Paired delta |
|---|---:|---:|---:|
| E01 | 96.551724 | 98.084291 | +1.532567 |
| E02 | 97.701149 | 95.019157 | -2.681992 |
| E03 | 94.252874 | 94.636015 | +0.383142 |
| Overall | 96.168582 | 95.913155 | -0.255428 |

Experimental overrides ranged from zero to ten per run and did not correlate
consistently with score. E02 regressed for every seed, so the profile is not a
non-inferior replacement for V13 and remains diagnostic-only.

After generalizing grouping to same-target, release-window-bound 2-6 H
packages, a fresh 36-run matrix (E01-E03, M01-M03, seeds 1-3) produced:

| Map | All V13 mean | Variable-package mean | Paired delta |
|---|---:|---:|---:|
| E01 | 94.827586 | 98.659004 | +3.831418 |
| E02 | 96.551724 | 96.934866 | +0.383142 |
| E03 | 93.103448 | 95.785441 | +2.681992 |
| M01 | 91.898148 | 93.750000 | +1.851852 |
| M02 | 94.907407 | 93.402778 | -1.504630 |
| M03 | 95.138889 | 93.981481 | -1.157407 |
| Overall | 94.404534 | 95.418928 | +1.014394 |

Paired outcomes were 9 wins, 5 losses, and 4 ties. Sidecar diagnostics show
that angle bonuses were rare (4 joint versus 8 V13 events over all 18 runs),
so the gain is not explained by the angle multiplier. Joint runs averaged more
total/public hit events but slightly less total public damage, indicating a
mixture of target-distribution, overkill, and interception-order effects. The
profile remains experimental because M02 and M03 regressed on average.

The expanded nine-map matrix added H01-H03:

| Map | All V13 mean | Variable-package mean | Paired delta |
|---|---:|---:|---:|
| H01 | 86.010710 | 87.014726 | +1.004016 |
| H02 | 92.570281 | 86.345382 | -6.224900 |
| H03 | 89.156627 | 91.030790 | +1.874163 |
| Nine-map overall | 92.684980 | 92.989385 | +0.304405 |

Across all 27 pairs the package profile won 12, lost 8, and tied 7. H02 is
the largest regression and confirms that the issue is not merely the trigger:
negative runs had larger predicted separation improvement than positive runs.
The action objective itself is misaligned with mission score. It optimizes
peer separation while ignoring V13 confidence, target value, ETA, satellite
window, and expected damage. Post-episode sidecars show nearly unchanged H hit
counts but lower H/public applied damage and fewer satellite-assisted hits.

The first 90-run mission-aware batch exposed and then invalidated an
entity/action-ordering bug: bindings were release-ordered while the pure skill
returned entity-ID-ordered actions. After freezing runtime groups in entity-ID
order, a fresh `missionbench2` 9-map x 5-seed x 2-mode matrix produced:

| Map | All V13 mean | Mission-aware mean | Paired delta |
|---|---:|---:|---:|
| E01 | 96.781609 | 98.160920 | +1.379310 |
| E02 | 95.632184 | 96.781609 | +1.149425 |
| E03 | 93.103448 | 91.264368 | -1.839080 |
| M01 | 91.041667 | 95.277778 | +4.236111 |
| M02 | 93.333333 | 90.972222 | -2.361111 |
| M03 | 93.472222 | 93.888889 | +0.416667 |
| H01 | 85.502008 | 87.710843 | +2.208835 |
| H02 | 90.361446 | 89.799197 | -0.562249 |
| H03 | 90.826305 | 90.722892 | -0.103414 |
| Overall | 92.228247 | 92.730969 | +0.502722 |

The 45 paired games contained 14 wins, 22 losses, and 9 neutral outcomes. More
importantly, the 24 pairs with actual overrides averaged only +0.158 score and
lost 14 versus 7 wins; pairs with more than one override averaged -0.771.
Mission-aware correctly eliminated direct LEFT/RIGHT reversal, but most active
changes stopped a V13 turn without knowing whether that turn was threat-driven.
Until V13 exposes proposal confidence/reason, active overriding remains unsafe.

### Original V13 vs V12-R3 Set-Attention

A 135-run matrix compared the original V13 runtime/checkpoint, the standalone
V12-R3 set-attention network, and V12-R3 plus mission-aware H packages across
E01-E03/M01-M03/H01-H03 with seeds 1-5:

| Condition | 45-run mean |
|---|---:|
| Original V13 | 93.289355 |
| V12-R3 | 93.917227 |
| V12-R3 + package | 94.015096 |

V12-R3 improved +0.627872 over V13 (21 wins, 17 losses, 7 ties). The package
layer added only +0.097869 over V12-R3 (14 wins, 12 losses, 19 ties), which is
within observed process variance. V12-R3 produced fewer total hits but more H
public hits, higher H/public damage, and more satellite-assisted hits; its gain
comes from hit quality rather than hit count. H02 remains the largest network
regression (-2.289157 versus V13).

### Full H/M/L + Mission Stack Against B4-B8

A 450-run matrix compared pure V12-R3 with `joint-hml-local` plus
`--mission-replan all-packages` across nine maps, five seeds, and B4-B8. The
full stack averaged 94.891626 versus 94.567362 (+0.324264), with 88 wins, 54
losses, and 83 ties. It was positive for B4/B5/B6/B8 but regressed -0.972254
against B7. Mission staging reduced mean launched inventory by roughly 26-39
missiles per map while increasing hidden-target kill rate for every blue
policy. M stop/supplement requests never activated in this matrix, so the
observed treatment is primarily H package action plus staged L response. The
B7 regression indicates that holding L removes useful interceptor saturation;
future MissionReplan must optimize global defense consumption, not hidden
target overkill in isolation.

## Tests

```bash
cd /home/amax/ry/competition/skill_generation/skills/h_single_maneuver
../../.venv/bin/python -I -m pytest -q
```
