# Dynamic Retarget

`DynamicRetargetSkill` is a pure, deterministic controller that proposes
semantic H/M retarget requests for public `9400` and `9600` targets. It never
creates simulator or engine action rows.

## Release And Candidate Rules

Only a live, flexible (`is_core=False`) member can be released from its current
target. The current target's delivered `destruction_probability` must reach the
configured threshold; `projected_probability` is never sufficient to release a
member. Stage, current-target distance, cooldown, and lifetime retarget limits
are also enforced. The default maximum release stage is `2`, so stage `3` is
rejected.

Every `decide` call represents exactly one communication group. All member
snapshots must name the same group. A destination is eligible only when all of
the following are true:

- Its ID differs from the current target and occurs in both
  `legal_target_ids` and `fresh_target_ids`.
- Its delivered and projected destruction probabilities are below the
  configured threshold.
- Its WGS84 haversine distance from the old target is no more than
  `max_target_separation_km` (default `15`).
- Its WGS84 haversine distance from the member is no more than
  `max_candidate_distance_km` (default `60`).
- The angle from the member's current `(velocity_x, velocity_y)` direction to
  the destination is no more than `max_turn_angle_degrees` (default `60`).

`velocity_x` is the local east/longitude component and `velocity_y` is the
local north/latitude component. Both may be `None` for construction
compatibility, but an absent or zero direction is not eligible. Likewise,
`fresh_target_ids` defaults to an empty tuple. This fail-closed default keeps
old snapshot construction valid without treating an unconfirmed track as
fresh.

Candidate ranking combines low delivered destruction probability, high target
value, projected probability shortfall, WGS84 distance, and low assigned count.
Input members and targets are normalized by ID, virtual assigned counts are
updated after each request, and IDs provide final tie breaks, so input ordering
cannot change the result. The global per-step budget remains the ceiling of the
eligible fraction. Independently, at most `max_per_old_target_per_step`
(default `1`) members are selected from the same old target in one call.

## Configuration

```python
DynamicRetargetConfig(
    destruction_threshold=0.90,
    max_stage=2,
    min_current_distance_km=15.0,
    cooldown_steps=25,
    max_retargets=1,
    max_fraction_per_step=0.20,
    distance_weight=0.05,
    max_target_separation_km=15.0,
    max_turn_angle_degrees=60.0,
    max_candidate_distance_km=60.0,
    max_per_old_target_per_step=1,
)
```

The three new geometric limits must be finite and positive;
`max_turn_angle_degrees` must also be at most `180`.
`max_per_old_target_per_step` must be a positive integer.

## Usage

```python
from dynamic_retarget_skill import (
    DynamicRetargetSkill,
    RetargetMemberSnapshot,
    RetargetTargetSnapshot,
)

member = RetargetMemberSnapshot(
    entity_id=20,
    missile_type=21001,
    current_target_id=901,
    alive=True,
    stage=2,
    current_distance_km=30.0,
    communication_group="group-a",
    is_core=False,
    retarget_count=0,
    last_retarget_step=None,
    lon=120.0,
    lat=30.0,
    legal_target_ids=(901, 902),
    fresh_target_ids=(901, 902),
    velocity_x=1.0,
    velocity_y=0.0,
)
targets = (
    RetargetTargetSnapshot(901, 9400, 5.0, 120.1, 30.0, 0.95, 0.99, 6),
    RetargetTargetSnapshot(902, 9600, 2.0, 120.2, 30.0, 0.10, 0.25, 0),
)

skill = DynamicRetargetSkill()
decision = skill.decide(step=100, members=(member,), targets=targets)
request = decision.requests[0]
assert (request.entity_id, request.new_target_id) == (20, 902)
```

`RetargetDecision.rejected_reasons` is a read-only mapping from every
non-selected member ID to its deterministic reason. Candidate-specific reasons
include `candidate_track_stale`, `target_separation`, `candidate_too_far`, and
`turn_angle`. Existing release reasons include `stage_too_late`, `too_close`,
`max_retargets`, `cooldown`, `target_probability_below_threshold`, and
`no_eligible_candidate`. Budget outcomes use `step_budget` and
`old_target_step_limit`.

The same pre-budget filtering result is available without duplicating logic:

```python
reason = skill.eligibility_reason(member, targets[0], targets, step=100)
assert reason == "eligible"
```

The caller remains responsible for accepting a request, emitting an
owner-routed engine action, and reflecting accepted changes in later snapshots.
