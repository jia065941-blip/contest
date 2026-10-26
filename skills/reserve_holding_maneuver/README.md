# Reserve Holding Maneuver

`reserve_holding_maneuver` version `0.1.0` is a pure, deterministic controller
for H (`21000`) reserves while their lifecycle state is `STAGING`. It loads no
network or checkpoint and has no runtime dependency outside the Python standard
library. The caller retains observation authority and command dispatch.

The caller must stop invoking this skill after dispatch. `DISPATCHED`,
`FALLBACK`, and every non-H type are fail-closed passthrough cases: the returned
action is exactly `baseline_action`.

## Data Contract

`HoldingSnapshot` is frozen and contains:

```text
entity_id, type, step,
position=(longitude, latitude),
velocity=(east, north),
anchor=(longitude, latitude),
forward_target=(longitude, latitude),
baseline_action, state
```

Longitude/latitude values are degrees. Velocity may use any consistent speed
unit because only its direction is used, but it must be nonzero. Actions are
`LEFT=0`, `STRAIGHT=1`, and `RIGHT=2`. Lifecycle state is one of `STAGING`,
`DISPATCHED`, or `FALLBACK`.

The controller projects both the position and the `anchor -> forward_target`
axis into a local east/north tangent plane. `forward_progress` is positive in
the target direction. Positive `cross_track` is left of that direction, so the
corridor boundaries are `+corridor_half_width_km` and
`-corridor_half_width_km`.

`Decision` contains:

```text
action, overridden, phase, cross_track, forward_progress, reason
```

`phase` is `left_leg`, `right_leg`, or `transition`.

## Control

The initial leg is selected from position and lateral velocity. A left leg
tracks `heading_offset_deg` left of the forward axis; a right leg tracks the
same offset to the right. On reaching the positive boundary, the controller
enters `transition`, commands exactly `straight_transition_steps` straight
actions, and then starts `right_leg`. The negative boundary performs the
symmetric change back to `left_leg`.

No more than `max_turn_steps` consecutive turn actions are emitted. If heading
correction still requests a turn at that limit, one `STRAIGHT` guard action is
inserted before turning can continue.

State is isolated by `entity_id`. With `latch=True`, a temporary passthrough
observation retains that owner's phase without expanding control authority;
the action is still passed through. With `latch=False`, passthrough clears that
owner. `reset(entity_id)` clears one owner and `reset()` clears all owners.

## Configuration

```python
from reserve_holding_maneuver_skill import Config

config = Config(
    enabled_type=21000,
    corridor_half_width_km=6.0,
    heading_offset_deg=35.0,
    max_turn_steps=6,
    straight_transition_steps=2,
    latch=True,
)
```

Version `0.1.0` deliberately fixes `enabled_type` to H (`21000`). The corridor
half-width must be positive, the heading offset must be in `(0, 90)` degrees,
and both step counts must be positive integers.

## Usage

```python
from reserve_holding_maneuver_skill import (
    HoldingSnapshot,
    ReserveHoldingManeuverSkill,
    STRAIGHT,
)

skill = ReserveHoldingManeuverSkill()
decision = skill.decide(HoldingSnapshot(
    entity_id=10,
    type=21000,
    step=20,
    position=(120.01, 30.0),
    velocity=(200.0, 0.0),
    anchor=(120.0, 30.0),
    forward_target=(121.0, 30.0),
    baseline_action=STRAIGHT,
    state="STAGING",
))

action = decision.action
```
