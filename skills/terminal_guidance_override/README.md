# Terminal Guidance Override

`TerminalGuidanceSkill` is a pure, deterministic pulse terminal-guidance
controller composed after `local_avoidance`. Before capture, the caller's
baseline action passes through. The fixed action mapping is:

- `0 = LEFT`
- `1 = STRAIGHT`
- `2 = RIGHT`

## V0.4 type profiles

V0.4 selects an independent `TerminalTypeProfile` from the snapshot's
`missile_type`. A type without a profile passes through. The default profiles
are:

| Type | Capture | Final straight | Fresh detect required | Maximum track age |
| --- | ---: | ---: | --- | ---: |
| H `21000` | 300 m | 150 m | Yes | 5 |
| M `21001` | 600 m | 200 m | Yes | 5 |
| L `21002` | 600 m | 200 m | No | None |

`max_track_age=None` permits a detect track of any age. It does not permit a
missing or non-detect track: L still requires an aimpoint and a track whose
source is `detect_fresh` or `detect_stale`. H and M require `detect_fresh` to
enter capture and reject tracks older than five steps. A latched finite-age
profile may use the inherited stale grace through age eight; after that it
releases and passes through.

All three default types use the same pulse tracking state machine. State is
isolated by `(missile_type, entity_id)`, so the same entity ID in two type
profiles has independent pulse, recovery, hysteresis, and latch state.
`reset(entity_id)` clears that owner in every type; `reset()` clears all state.
Interceptor proximity remains diagnostic and cannot interrupt a latch.

## Pulse guidance

The controller projects the aimpoint line of sight and missile velocity into
the local horizontal plane. It reports signed heading error and the constant-
velocity future closest-approach miss distance. Closest approach is restricted
to nonnegative time, so a missile moving away reports its current horizontal
separation as the predicted miss.

Control is evaluated again on every snapshot in this order:

1. A finite-age profile in latched stale grace commands `STRAIGHT` with control
   state `STALE_GRACE`; an expired track releases capture.
2. Range at or below the profile's final-straight distance commands
   `FINAL_STRAIGHT`.
3. A heading-error sign crossing clears the pending turn and begins the
   configured `STRAIGHT_RECOVERY` period. An opposite turn is never emitted
   without a straight transition.
4. Predicted miss at or below the configured threshold commands `STRAIGHT`.
5. Heading error at or below the enter threshold starts straight hysteresis;
   it remains straight until the exit threshold is reached.
6. The configured number of non-improving heading observations restarts
   straight recovery.
7. Otherwise the controller emits a `TURN_PULSE`, followed by straight recovery
   before another pulse is selected.

The global state-machine defaults remain: stage 4, stale grace age 8, heading
enter/exit thresholds 6/10 degrees, predicted miss threshold 80 m, one pulse
step, two recovery steps, three non-improving observations, and latch enabled.

## Configuration

`TerminalTypeProfile` is a frozen DTO. Core distance/freshness settings and the
following state-machine settings may be overridden per type:

- `min_stage`
- `stale_grace_age`
- `heading_enter_straight_deg`
- `heading_exit_straight_deg`
- `predicted_miss_threshold_m`
- `turn_pulse_steps`
- `straight_recovery_steps`
- `no_improvement_limit`
- `latch`

For profile fields other than `max_track_age`, `None` inherits the global
`TerminalGuidanceConfig` value. `max_track_age="inherit"` inherits the global
value, while `max_track_age=None` explicitly allows tracks of any age.

`profiles` accepts either a mapping or a tuple of `(missile_type, profile)`
pairs:

```python
from terminal_guidance_override_skill import (
    TerminalGuidanceConfig,
    TerminalGuidanceSkill,
    TerminalTypeProfile,
)

config = TerminalGuidanceConfig(profiles={
    21001: TerminalTypeProfile(
        capture_distance=750,
        final_straight_distance=225,
        max_track_age=5,
        require_fresh_track=True,
    ),
    21002: TerminalTypeProfile(
        capture_distance=650,
        final_straight_distance=200,
        max_track_age=None,
        require_fresh_track=False,
    ),
})
skill = TerminalGuidanceSkill(config)
```

The v0.1-v0.3 single-profile API remains accepted. Supplying
`enabled_types=(21001,)` with global thresholds constructs one M profile. The
deprecated aliases `capture_distance_m`, `enter_distance_m`,
`straight_distance_m`, `max_track_age_steps`, and
`heading_deadband_degrees` still map to their current fields. Conflicting old
and new values are rejected. `exit_distance_m`, `threat_distance_m`,
`ignore_interceptors_after_latch`, and `straight_action` retain their v0.3
compatibility behavior.

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

This package does not load a model, inspect a simulator, or read simulator
state. Geometry, freshness, stage, velocity, aimpoint, and baseline action are
caller-provided inputs.
