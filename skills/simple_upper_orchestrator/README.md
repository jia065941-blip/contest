# Simple Upper Orchestrator

This package wires nine pure skills into the unmodified 2026-10-21 official
release. It installs a process-local replacement for only the exported
`user_agents.red.RedAgentFactory`, calls the official `prepare_run(args)` and
`run_simulation(args, bundle)`, and restores the export afterward.

The adapter uses `local_avoidance_skill.load_external_policy`; the runtime and
checkpoint are caller-owned. It does not import or depend on archived
`local_avoid_v13` code. During deployment, the deploy agent legally sees the
complete friendly roster and public 9400/9600 directory. It runs
`HMPublicStrikePackagesSkill` once and publishes an immutable pre-mission plan:
the H/M assignment map, frozen entity core attributes/types, sorted complete L
roster, and any H reserve packages. Reserve H packages start in `STAGING` with
public provisional/fallback metadata and a latitude responsibility region. In
2.5, provisional targets remain package metadata and reserve members are absent
from both the core H/M assignment map and target-aware deployment requests. They
do not enter runtime assignments, goals, or type-4 actions while `STAGING`.
Only a confirmed deficit dispatch or dynamic deadline fallback creates the
runtime assignment and action. Core H/M behavior is unchanged. The combat
coordinator never calls the global H/M allocator and does not redistribute core
H/M from current observations. Each core H/M owner emits type 4 for its fixed
target on its first action; later goal refreshes for that ID use only that
owner's `detectInfo`.

For explicitly enabled dynamic retargeting, legal fresh candidate IDs remain
strictly owner-local. Only members in the same current communication group may
merge ledgers and fresh 9400/9600 coordinates; the newest legal group report
wins. Public initial coordinates are never marked fresh. Commitments are
registered from the pre-mission plan before combat. Terminal evidence is added
when a previously active owner reports terminal health or disappears from a
later step, retained in owner ledgers, and disclosed only to that owner's last
official communication group. Disappearance alone uses the conservative
delivery fallback; it is never treated as confirmed destruction. Flexible M
missiles may retarget from a sufficiently likely delivered kill; H commitments
remain core and projected probability never triggers release. Provisional
reserve H package targets are excluded from this core commitment ledger.

Each L uses only its own isolated legal 9500 records. Legal records are merged
into an owner-local persistent catalogue and are never copied across owners.
Once its local catalogue
meets `--l-expected-target-count`, it locks
`sorted_target_ids[roster_rank % target_count]`. No other L position, track,
survival state, fleet common-track set, or Hungarian assignment affects that
choice. Reserve dispatch and fallback may use a legal 9500 entry retained from
an earlier step even when it is absent from current `detectInfo`. The lowest
deployment-time L roster ID is the only satellite requester,
the request remains globally single-shot, and every hidden-target type-4 action
requires that ID in the action owner's legal catalogue.

When `--l-first-wave-per-target` is positive, the first `N * target_count`
members of the sorted L roster are deterministic core owners after satellite
discovery. Any remainder becomes singleton `STAGING` reserve packages. A reserve
L remains without an assignment or target action; its provisional
and fallback dark nests remain owner-local catalogue metadata until dispatch.
The default value `0` retains the prior all-members rank allocation exactly.

## Install

Install this package and its nine sibling packages in the Python environment
used by the private-runtime launcher:

```bash
python -m pip install -e ../staggered_deployment \
  -e ../l_satellite_strike_packages \
  -e ../hm_public_strike_packages \
  -e ../target_damage_probability \
  -e ../dynamic_retarget \
  -e ../local_avoidance \
  -e ../terminal_guidance_override \
  -e ../reserve_echelon \
  -e ../reserve_holding_maneuver \
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
  --checkpoint /home/amax/ry/competition/skill_generation/skills/local_avoidance/net/local_avoid_set_attention_v12_r3_timely_arrival.pt \
  --scenario launch20/easy/E01 \
  --seed 701 \
  --max-steps 3000 \
  --render-mode none \
  --output-root /home/amax/ry/competition/results/simple-upper \
  --run-id E01-s701 \
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
  --deploy-l-first-wave-fraction 0.70 \
  --deploy-batch-size 10 \
  --l-satellite-trigger-step 1 \
  --l-expected-target-count 10 \
  --hm-allocation-mode deterministic_thresholds \
  --hm-target-kill-probability 0.85 \
  --hm-min-h-per-mother 2 \
  --hm-min-m-per-mother 0 \
  --hm-min-h-per-sentry 1 \
  --hm-max-h-per-mother 2 \
  --hm-h-per-sentry-package 2 \
  --hm-m-per-h-mixed-package 2 \
  --hm-m-only-sentry-package 6 \
  --hm-m-target-policy complete_h_sentries \
  --hm-target-priority-axis east \
  --hm-reserve-h 0 \
  --hm-reserve-m 0 \
  --reserve-h-count 0 \
  --l-first-wave-per-target 0 \
  --reserve-deadline-step 300 \
  --reserve-h-max-distance 80 \
  --reserve-l-max-distance 100 \
  --reserve-h-turn-rate 20 \
  --reserve-l-turn-rate 15 \
  --reserve-turn-safety-seconds 5 \
  --reserve-holding-half-width 6 \
  --reserve-holding-heading-offset 35 \
  --reserve-holding-max-turn-steps 6 \
  --reserve-holding-straight-steps 2 \
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
  --enable-terminal-guidance \
  --terminal-min-stage 4 \
  --terminal-h-capture300 300 \
  --terminal-h-final150 150 \
  --terminal-m-capture600 600 \
  --terminal-m-final200 200 \
  --terminal-l-capture600 600 \
  --terminal-l-final200 200 \
  --terminal-max-track-age 5 \
  --terminal-stale-grace-age 8 \
  --terminal-threat-distance 3000 \
  --terminal-heading-enter 6 \
  --terminal-heading-exit 10 \
  --terminal-predicted-miss 80 \
  --terminal-turn-pulse-steps 1 \
  --terminal-straight-recovery-steps 2 \
  --terminal-no-improvement-limit 3 \
  --velocity-window-steps 5 \
  --formation-audit-interval 25 \
  --missile-trace-interval 0 \
  --missile-terminal-history 20 \
  --log-frequency 1 \
  --enable-ai-action
```

All official output and logging switches remain available, including
`--batch-size`, `--verbose`, `--enable-config`, `--enable-state`,
`--enable-event`, and `--disable-log-color`. Run `simple-upper-release --help`
for the combined official and adapter parser.

Dynamic retargeting is disabled by default. Pass `--enable-dynamic-retarget`
only for an explicit experiment. Evidence accounting, communication-ledger
isolation and audit output remain active while retargeting is disabled.

`--reserve-h-count` is the canonical H reserve control and defaults to zero. The
legacy `--hm-reserve-h` remains accepted when the canonical option is zero;
setting both nonzero is rejected. M reserves are not supported by
`reserve-echelon-skill==0.4.0`, so a nonzero `--hm-reserve-m` fails closed.
`--reserve-max-stage` is optional and defaults to no stage limit. Supplying a
nonnegative integer restores an explicit stage gate. Both deficit dispatch and
deadline selection must pass `reserve-echelon-skill==0.4.0` physical
reachability for every package member. H/L distance defaults remain 80/100 km.
The `--reserve-h-max-turn` and `--reserve-l-max-turn` risk caps now default to
`None`; supplying `60`/`75` explicitly restores the old hard angle limits.
Otherwise any angle is eligible when its turn-aware ETA fits. ETA uses nominal
H/L speeds of 200/100 m/s, turn rates from `--reserve-h-turn-rate` (default
`20` deg/s) and `--reserve-l-turn-rate` (default `15` deg/s), plus
`--reserve-turn-safety-seconds` (default `5`). The remaining budget is
`(max_steps - step) * --seconds-per-step`. Negative pre-launch stages normalize
to zero.

A deficit is keyed by current communication group and target. Its first complete
step creates `PendingDeficit(first_step, evidence_signature, missing_damage)` and
emits `reserve_pending(status=pending)` without dispatch. The same legal target
and evidence must survive into a later step before it becomes `confirmed` and is
submitted to Reserve Echelon. Changed or vanished evidence emits `cancelled` and
cannot produce action 4. Terminal evidence and group damage beliefs are fully
merged before this update. Delivered destruction probability at least `0.9`
cancels a public-target deficit. A confirmed deficit may commit the package's
provisional target ID directly even though `STAGING` itself has no assignment.
Confirmation and the final action 4 both revalidate the current group track,
delivered probability, active core commitments, and Reserve/core ETA. A target
at `0.9` is audited as `reserve_target_satisfied`; a previously fresh target
that expires after high-confidence close core delivery is audited as
`reserve_target_stale_cancel`. Neither path reads target health.

At `--reserve-deadline-step` (default `300`), the adapter does not blindly use
the package's fixed fallback metadata. It scores the current group's legal
targets from delivered destruction probability, current/confirmed deficit,
target weight, distance, and active core commitment coverage. Targets at or
above `0.9` delivered probability are excluded, confirmed deficits rank first,
and sufficient active core commitments carry a penalty. Candidates are tried in
score order through 0.4 reachability. Fresh moving tracks rank ahead of static
coordinates, and the score compares core completion ETA with Reserve ETA. The selected target's current legal
position is passed into that ETA check. This reduces selection of a target likely
to be destroyed by core missiles before the reserve arrives.

While an H reserve remains `STAGING` and unassigned, the external local policy
still runs first. `reserve-holding-maneuver-skill==0.1.0` may then replace its
baseline category with a bounded S-pattern action. The first owner-local
position is the holding anchor and the package's public provisional target is
the forward axis, with its public fallback used when the provisional axis is
degenerate. Defaults are a 6 km half-width, 35 degree heading offset, six
consecutive turn steps, and two straight transition steps; configure them with
`--reserve-holding-half-width`, `--reserve-holding-heading-offset`,
`--reserve-holding-max-turn-steps`, and
`--reserve-holding-straight-steps`. L reserves use only the local-policy
baseline. During a pending warning, a matching H reserve enters READY: holding
is stopped, the local policy still owns the motion category, and only its
temporary `goal_position` changes to the pending target's public coordinate for
pre-alignment. READY emits `reserve_ready_align` but never action 4. A cancelled
pending warning restores normal holding. Dispatch/fallback stops and resets
holding in the same planning step
before type 4 is emitted. A reserve not yet exposed by the environment remains
staged without any scoring target command.

Core H/M terminal evidence becomes an early loss when `stage <= 3` and its
minimum target distance remains above `10 km`, or when delivery probability is
at most `0.15`. Core L tracks its real stage, position, fixed assigned dark nest,
and minimum distance; disappearance is an early loss only at `stage <= 3` and
above `5 km`. Deficits are rebuilt for each current communication group from
visible unique early-loss events after the current step's terminal/belief merge.
Target health is `24/18/1.2` for
9400/9600/9500; initial core nominal commitment is reduced by the visible lost
nominal damage (`H=16/12`, `M=4/3`, `L=0.8`) before missing damage is computed.
Evidence is assigned a unique event ID, explicitly audited, and written to the
legal ledgers of the lost owner's last communication-group recipients. Current
out-of-group reserves cannot consume that evidence. A reserve member is eligible
only while current, within the stage gate, and with the requested target in its
owner-local legal catalogue; action 4 is always routed under and audited for
that member's owner ID. A multi-member package owner is its lowest current
surviving ID. Losing that owner emits `reserve_owner_changed` and preserves the
state, target, and assignments; only loss of every member emits `TERMINATED`.
After H Reserve action 4, a completed or qualifying stale target may switch once
to a reachable confirmed deficit or dynamic fallback. `reserve_retarget` updates
the package `target_id` and `retarget_count`; Core and M/L behavior is unchanged.

Terminal guidance is disabled by default and requires
`--enable-terminal-guidance`. After the external local policy has produced each
owner's baseline `0/1/2` category, the adapter passes an owner-local snapshot to
`terminal-guidance-override-skill==0.4.0` and maps its final category back to
`acc_z = category - 1`. H, M, and L all use the profiled controller. H/M require
their assigned public target's fresh fix from that owner's current `detectInfo`.
L requires the assigned legal type-9500 fix from its own current `detectInfo`;
because dark nests are stationary, an unchanged stale fix remains eligible.
The detected fix, not a public-directory or cross-owner coordinate, is the
snapshot aimpoint. Missing or invalid assignment, position, fix, or aimpoint
fails closed to the baseline category.
Velocity is selected strictly from that same owner's isolated observation:
finite official ECF `x/y/z` is converted to local east/north first, finite
`speed/heading` is second, and an owner-local bounded position-window estimate
is the only fallback. Official velocity fields are audited as `observed`; the
fallback is audited and passed to the pure skill as `estimated_window`. Its
per-owner `deque(maxlen=--velocity-window-steps)` defaults to five samples and
computes east/north velocity from the current and oldest retained valid sample,
using WGS84 local radii, wrapped longitude deltas, and `--seconds-per-step`
(default `1`). Non-finite, non-forward, or over-limit samples retain the last
valid estimate; a speed below `1 m/s` clears it, and a speed above `1000 m/s`
starts a new position window without replacing it. The first position-only
sample has no velocity and therefore fails closed. It never reads simulator
state, another owner's observation, or another owner's history.

The 0.4 defaults are H `300/150` m, M `600/200` m, and L `600/200` m for
capture/final-straight distance. H/M fixes must be fresh with maximum age `5`;
L permits a stale legal detect fix. All profiles otherwise inherit stage `4`,
heading hysteresis at `6`/`10` degrees, predicted miss threshold `80` m, a
one-step turn pulse, two straight-recovery steps, and a three-sample
no-improvement limit. Override type distances with
`--terminal-h-capture300`/`--terminal-h-final150`,
`--terminal-m-capture600`/`--terminal-m-final200`, and
`--terminal-l-capture600`/`--terminal-l-final200`; the shorter aliases without
the numeric suffix are also accepted. The generic
`--terminal-capture-distance`, `--terminal-final-straight-distance`, and all
legacy terminal flags remain compatible with the 0.4 global/M settings.
Interceptor threat remains diagnostic only. `--disable-terminal-guidance`
remains accepted for compatibility but explicit enablement is required.

The pre-mission plan is immutable and retained across episode resets. Reset
clears observations, actions, goals, target refresh versions, dynamic evidence,
satellite state, and other battle history, then restores only the fixed H/M
core assignments and their initial owner-local commitment records. Staged reserve
members return without runtime assignments. Terminal-guidance
latches, position/velocity histories, pending/confirmed deficits, READY targets,
and audit de-duplication state are also cleared. Reserve-holding phases and
first-position anchors are cleared on
terminal state and episode reset. Terminal owners also have their
position/velocity history removed. A missing plan
fails closed: the coordinator may still produce owner-local motion, but it emits
no H/M or L target action and does not infer a replacement assignment.
Reserve skill state, early-loss evidence and ledger event IDs, deficits,
dispatch counts, terminal package status, owner-local L catalogues, and dynamic
L packages are battle state and are cleared. Pre-mission H reserve definitions
and the deployment L roster are retained; H packages return to uncommitted
`STAGING`.

The adapter uses `dynamic-retarget-skill==0.3.0`. A detected public track is
fresh only when it has a legal ID/type/finite position and a non-future age of
at most `--dynamic-retarget-track-max-age` steps. Candidate freshness is checked
against the requesting owner's current `detectInfo`, not the public directory or
another group member. The communication group may contribute a newer fresh
coordinate to the pure skill's target snapshot. Retargeting also requires finite
`x`/`y` components from `self.velocity` or `self.vel_lla`; a missing or zero
direction fails closed. The 0.3 geometry gates default to a 15 km old/new target
separation, a 60 degree turn, a 60 km member/candidate distance, and one request
per old target per step. Stage 3 is rejected by the default maximum stage of 2.

Terminal delivery distance and stage heuristics are used only when the last
assigned-target coordinate was `detect_fresh`. A public fallback or stale track
sets an explicit delivery probability of `0.15`, even if its coordinate appears
near the missile. The recorded progress keeps the coordinate source and track
age used for the current assignment.

## Retarget audit

Each process writes `retarget_audit.jsonl` directly under its resolved run
output root (`args.output_dir`, including `--run-id` when supplied). The file is
truncated when the process-local coordinator is initialized, then appended
through episode resets. Pass `--disable-retarget-audit` to disable creation and
writing entirely.

Every line is a JSON object serialized with sorted keys and contains at least
`event` and `step`. Event-specific fields are:

| `event` | Fields |
| --- | --- |
| `initial_commitment` | `owner`, `type`, `target`, `commitment`, `group`, `event_id` |
| `initial_reserve` / `reserve_staging` | `package`, `members`, `types`, `provisional_target`, `fallback_target`, `responsibility_region`, `source`; staging also records `state=STAGING`, `committed=false` |
| `reserve_early_loss` | `owner`, `target`, `type`, `target_type`, `nominal_damage`, `stage`, `min_distance_m`, `probability`, `last_group`, `recipients`, `event_id` |
| `reserve_deficit` | `package`, `owners`, `group`, `target`, `type`, `missing`, `target_health`, `initial_core_nominal`, `early_lost_nominal`, `responsibility_region`, `evidence_ids`, `event_ids` |
| `reserve_pending` | `status` (`pending`, `confirmed`, or `cancelled`), `group`, `target`, `first_step`, `evidence_signature`, `missing`, plus package/responsibility or cancellation reason when applicable |
| `reserve_rejected` | `package`, `members`, `reason`, plus applicable group or unavailable-member context; unchanged rejection states are not repeated |
| `reserve_dispatch` | `package`, `members`, `old_target`, `new_target`, `reason`, `group`, `evidence_ids`, `evidence_event_ids`, `missing`, `event_id`, and 0.3 reachability fields |
| `reserve_fallback` | `package`, `members`, `old_target`, `new_target`, `reason`, `group`, `event_id`, and 0.3 reachability fields |
| `reserve_fallback_candidate` | `package`, `group`, `target`, `position`, `score`, `confirmed_deficit`, `missing`, `weight`, `distance_km`, `delivered_probability`, `active_core_ids`, `active_core_damage`, `active_core_coverage`, `eligible`, `excluded_reason`, `selected`, `reachability_reason` |
| `reserve_action4` | `package`, `owner`, `target`, `action_type`, `reason`, `event_id` |
| `reserve_owner_changed` | `package`, `old_owner`, `new_owner`, `surviving_members`, `state`, `target`, `reason`, `group` |
| `reserve_target_satisfied` / `reserve_target_stale_cancel` | `package`, `owner`, `members`, `group`, `target`, delivered probability, track freshness, core delivery and ETA fields |
| `reserve_retarget` | `package`, `owner`, `members`, `old_target`, `new_target`, `retarget_count`, `group`, `reason`, `event_id`, and reachability fields |
| `reserve_terminal` | `package`, `members`, `owner`, `previous_state`, `status`, `reason`, `group` |
| `reserve_holding` | `package`, `owner`, `phase`, `cross_track`, `progress`, `baseline`, `final`, `overridden`, `reason`, plus owner-local velocity and axis provenance when available |
| `reserve_ready_align` | `package`, `owner`, `target`, public `goal`, local-policy `baseline`/`final`, `holding_paused=true`, `action4=false` |
| `terminal_evidence` | `owner`, `type`, `target`, `last_stage`, `min_distance_m`, `source`, `age`, `forced_low_confidence`, `last_group`, `delivery_probability`, `evidence_recorded`, `terminal_reason`, `missile_terminal_event_id`, `event_id` |
| `belief_change` | `group`, `target`, `type`, `delivered_probability`, `projected_probability`, `committed_ids`, `delivery_ids` |
| `retarget_status` | `owner`, `type`, `current_target`, `delivered_probability`, `threshold`, `stage`, `distance_km`, `group`, `retarget_count`, `last_retarget_step`, `fresh_target_ids`, `velocity_x`, `velocity_y`, `max_target_separation_km`, `max_turn_angle_degrees`, `max_candidate_distance_km`, `max_per_old_target_per_step`, `reason` |
| `retarget_request` / `retarget_applied` | `owner`, `type`, `old_target`, `new_target`, `probability`, `group`, `stage`, `distance_km`, `candidate_position`, `candidate_position_source`, `reason`, `event_id` |
| `formation_status` | `m_id`, `target_id`, `m_stage`, `m_lon`, `m_lat`, `m_track_source`, `m_track_age`, `h_ids`, `nearest_h_id`, `separation_km`, `lateral_km`, `forward_km`, `same_communication_group`, `group`, `movement`, `velocity`; `reason=no_active_h_lead` when no active same-target H exists |
| `terminal_guidance` | `owner`, `type`, `profile`, `profile_capture_distance`, `profile_final_straight_distance`, `baseline`, `final`, `stage`, `distance_m`, `track_source`, `track_age`, `nearest_interceptor_distance_m`, `threat`, `latched`, `overridden`, `reason`, `mode`, `error`, `predicted_miss_distance`, `control_state`, `no_improvement_count`, `heading_error`, `velocity_source`, `velocity_east`, `velocity_north`; guided decisions also retain compatible `heading_error_degrees` and optional `turn_direction` |
| `missile_trace` | `id`, `type`, `profile`, `profile_capture_distance`, `profile_final_straight_distance`, `stage`, `position`, `velocity`, `velocity_source`, `velocity_east`, `velocity_north`, `current_target_id`, `aimpoint`, `track_source`, `track_fingerprint`, `track_age`, `aimpoint_distance_km`, `comm_group`, `same_target_active_h_ids`, `nearest_h_id`, `nearest_h_separation_km`, `same_group`, `launch_step`, `steps_since_launch`, `maneuver_state`, `progress_step`, `minimum_aimpoint_distance_m`, `retarget_count`, `last_retarget_step`, `baseline_action`, `final_action`, `override_reason`, `mode`, `error`, `predicted_miss_distance`, `control_state`, `no_improvement_count`, `heading_error` |
| `missile_terminal` | `id`, `type`, `current_target_id`, `terminal_reason`, `last_sample`, `history`, `history_length`, `event_id`, `terminal_evidence_event_id` |
| `reset` | `owner` |

The 0.4 Reserve reachability fields are `distance`, `turn_angle`, `turn_time`,
`travel_time`, `total_eta`, and `remaining_time`. Unit-explicit aliases use
`distance_km`, `turn_angle_degrees`, and the corresponding `*_seconds` names;
the configured maximums are also recorded. `travel_steps` and `remaining_steps`
remain as compatibility aliases.

Belief rows are emitted only when a probability or missile-ID set changes for
that communication group and target. Status rows are emitted when any audited
status field changes; reasons include `target_probability_below_threshold`,
`stage_too_late`, `too_close`, `max_retargets`, `cooldown`,
`candidate_track_stale`, `target_separation`, `candidate_too_far`, `turn_angle`,
`old_target_step_limit`, `no_eligible_candidate`, and `eligible`. Enabled-mode
status reasons are taken directly from `RetargetDecision.rejected_reasons`; only
members that cannot form a legal snapshot use `insufficient_legal_data`. When
dynamic retargeting is disabled, one status row is emitted initially and then
only if its reason changes; changing positions and distances do not create a
row every step.

Terminal-guidance rows are emitted when an owner's `reason`, `latched`, `final`,
`control_state`, or `no_improvement_count` value changes, and on every decision
that actually overrides the local policy category. This records pulse/recovery,
straight, stale-grace, and final-straight state transitions without emitting an
unchanged pass-through decision merely because its geometry changed.

Formation rows are emitted at steps divisible by
`--formation-audit-interval` (default `25`), at most once per active M and
interval step. `h_ids` contains only active H missiles with a current assignment
matching that M and present in the M's current communication group. The
nearest-H separation uses haversine distance; signed
`lateral_km` and `forward_km` are local-tangent approximations along the M's
owner-local target track when all required positions are legal. `movement` is
the current maneuver state and `velocity` is the finite current observation
payload. Formation de-duplication is cleared on episode reset.

M trace auditing is disabled by default (`--missile-trace-interval 0`). A
positive interval records each active M immediately on its first owner-local
sample and then at most once per interval steps. Repeated planning of the same
step is de-duplicated. Each M has a
`deque(maxlen=--missile-terminal-history)`; the default terminal window is 20
samples and the history value must be a positive integer. When that M later
reports terminal health or is absent from a new current step,
`missile_terminal` embeds only this bounded pre-terminal window and its last
sample. Setting the trace interval back to zero emits no `missile_trace` or
`missile_terminal` rows and allocates no history deque.

The trace aimpoint, source, fingerprint and age come only from the M owner's
current legal public-directory/`detectInfo` view. Same-target active H IDs and
positions come from those H owners' current isolated observations;
`same_group` states whether the nearest such H is in the M owner's current
official communication group. No simulator-global state, event stream or
ground-truth entity table is consulted. `baseline_action`, `final_action`, and
`override_reason` record the same owner-local motion decision applied that step.

The audit is an explanation of the coordinator's evidence-based decisions, not
simulator ground truth. It uses only the coordinator's existing public target
directory, isolated owner observations, commitment ledgers, and derived
beliefs; it does not read official event logs, summaries, or simulator state.

The default `0.70` threshold requires two fresh-track stage-4 H deliveries
(`0.85²`) or two fresh-track terminal-close H deliveries (`0.98²`). A single H
cannot satisfy a 24-HP mother, public/stale coordinate fallback cannot claim a
high-confidence delivery, and projected commitments never trigger release.

`--runtime-root` and `--checkpoint` default to the current contest-jmy checkout
and checkpoint paths, but no model class or checkpoint hash is fixed by this
package. Set `SIMPLE_UPPER_GLIBC_ROOT` to override the default private glibc 2.38
location. The launcher forces OpenMP, MKL, OpenBLAS, NumExpr, BLIS, vecLib, and
Torch thread counts to `SIMPLE_UPPER_THREADS` (default one). Checkpoint metadata
selects its own local gate; the orchestrator does not force one gate across
different networks. For a caller-provided JSON profile, legacy
`"commRangeInfo": []` values are migrated into a generated copy under the run
output directory; the source and official checkout are never modified.

The `staggered-deployment-skill==0.9.0` planner supports a rearmost 5 km
`reserve_h` role while retaining target-aligned core H/M geometry. Release
scripts pass `l_first_wave_fraction=1`, so all L missiles use the front band.
Sentry rendezvous can be `adaptive` or `fixed`.
Adaptive mode uses H/M speed, desired M arrival lead, and minimum/optional
maximum M lead parameters. Deployment-time H/M allocation is fixed for the
episode; post-discovery L distribution comes solely from the deterministic
deployment spatial-rank rule above, with owner-local targets ordered laterally.

## Batch scripts

```bash
MAX_JOBS=40 SIMPLE_UPPER_THREADS=1 \
  bash tools/run_launch20_two_networks_two_seeds.sh

SIMPLE_UPPER_THREADS=1 SEED=1 \
  bash tools/run_e01_two_networks.sh
```

Use `SIMPLE_UPPER_OUTPUT_ROOT=/custom/path` to override a script output root.
The scripts intentionally ignore a generic inherited `OUTPUT_ROOT` so an old
shell export cannot mix strategy revisions.

## Test

```bash
python -m pytest -q
python -m compileall -q src tests
```
