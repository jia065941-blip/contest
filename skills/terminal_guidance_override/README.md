# Terminal Guidance Override

`TerminalGuidanceSkill` is a pure, deterministic pulse terminal-guidance
controller composed after `local_avoidance`. Before capture, the caller's
baseline action passes through. The fixed action mapping is:

- `0 = LEFT`
- `1 = STRAIGHT`
- `2 = RIGHT`

## V0.3 pulse guidance

An M missile (`21001`) may enter capture at stage 4 or later when it has an
aimpoint, a `detect_fresh` track no more than five steps old, and range at or
below 2000 m. Capture is owner-local and latched by default. Interceptor threats
remain available as diagnostics but never change capture or guidance.

The controller projects the aimpoint line of sight and missile velocity into
the local horizontal plane. It reports signed heading error and the constant-
velocity future closest-approach miss distance. Closest approach is restricted
to nonnegative time, so a missile moving away reports its current horizontal
separation as the predicted miss.

Control is evaluated again on every snapshot in this order:

1. A latched track older than 5 and no older than 8 steps commands `STRAIGHT`
   with control state `STALE_GRACE`. A track older than 8 releases capture and
   passes through.
2. Range at or below 300 m commands `FINAL_STRAIGHT`.
3. A heading-error sign crossing clears the pending turn and begins two
   `STRAIGHT_RECOVERY` steps. An opposite turn is never emitted without a
   straight transition.
4. Predicted miss at or below 80 m commands `STRAIGHT`.
5. Heading error at or below 6 degrees enters straight hysteresis. It remains
   straight until heading error reaches 10 degrees.
6. Three consecutive observations whose absolute heading error does not improve
   restart straight recovery.
7. Otherwise the controller emits a one-step `TURN_PULSE`, followed by two
   `STRAIGHT_RECOVERY` steps before another pulse can be selected.

Per-owner state contains `mode`, `previous_error`, `error_sign`,
`no_improvement`, `pulse_remaining`, `recovery_remaining`,
`straight_hysteresis`, and `turn_direction`. `reset(entity_id)` clears one
owner; `reset()` clears all owners.

`TerminalGuidanceDecision` retains the v0.2 fields and adds
`predicted_miss_distance`, `control_state`, `no_improvement_count`, and
`velocity_source`. Control states are `PASSTHROUGH`, `TURN_PULSE`,
`STRAIGHT_RECOVERY`, `STRAIGHT`, `STALE_GRACE`, and `FINAL_STRAIGHT`.

## Configuration

The v0.3 defaults are:

| Field | Default |
| --- | ---: |
| `capture_distance` | `2000` |
| `final_straight_distance` | `300` |
| `max_track_age` | `5` |
| `stale_grace_age` | `8` |
| `heading_enter_straight_deg` | `6` |
| `heading_exit_straight_deg` | `10` |
| `predicted_miss_threshold_m` | `80` |
| `turn_pulse_steps` | `1` |
| `straight_recovery_steps` | `2` |
| `no_improvement_limit` | `3` |
| `latch` | `True` |

The following v0.1/v0.2 fields remain accepted for source compatibility but
are deprecated:

| Deprecated field | V0.3 behavior |
| --- | --- |
| `capture_distance_m`, `enter_distance_m` | Alias `capture_distance` |
| `straight_distance_m` | Alias `final_straight_distance` |
| `max_track_age_steps` | Alias `max_track_age` |
| `heading_deadband_degrees` | Sets both v0.3 heading thresholds |
| `exit_distance_m` | Validated and exposed, but does not release a latch |
| `threat_distance_m` | Only defines the `threat` diagnostic threshold |
| `ignore_interceptors_after_latch` | Validated and exposed, but threats are always ignored |
| `straight_action` | Accepted only as the fixed compatible value `1` |

Conflicting old and new values are rejected.

## Snapshot geometry

Position and aimpoint may each use either WGS84 longitude/latitude degrees or
an ECF/ECEF `(x, y, z)` tuple in metres. Velocity may use local
`velocity_east_mps`/`velocity_north_mps`, or an ECF/ECEF vector in
`velocity_ecf`. ECEF vectors are projected into the current local east/north
tangent plane. `velocity_source` is a nonempty diagnostic string and defaults
to `"provided"` for compatibility.

Longitude/latitude range uses great-circle distance; ranges involving ECEF use
three-dimensional Euclidean distance. An unavailable aimpoint is represented
by leaving both aimpoint forms `None`. Only one representation of each
available value may be supplied.

```python
from terminal_guidance_override_skill import (
    TerminalGuidanceSkill,
    TerminalGuidanceSnapshot,
)

skill = TerminalGuidanceSkill()
decision = skill.decide(TerminalGuidanceSnapshot(
    entity_id=17,
    missile_type=21001,
    stage=4,
    baseline_action=1,
    aimpoint_lon=120.01,
    aimpoint_lat=30.0,
    track_source="detect_fresh",
    track_age=0,
    step=100,
    position_lon=120.0,
    position_lat=30.0,
    velocity_east_mps=0.0,
    velocity_north_mps=300.0,
    velocity_source="observed_heading",
))
```

This package does not load a model, inspect a simulator, or read simulator
state. Geometry, freshness, stage, velocity, aimpoint, and baseline action are
caller-provided inputs.
