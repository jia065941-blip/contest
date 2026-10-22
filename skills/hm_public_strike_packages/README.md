# H/M Public Strike Packages

`HMPublicStrikePackagesSkill` is a standalone, deterministic controller for H
(`21000`) and M (`21001`) missiles attacking only public `9400` and `9600`
targets. It composes with the caller's local maneuver categories `0/1/2`; it
does not import R9, another skill, a simulator, or target current-health state.

The initial directory is caller-owned public information. Its immutable
`initial_health` is the only health input. Later coordinates can be refreshed
only from isolated H/M `detectInfo` for an existing matching `target_id` and
type. Hidden `9500` detections and unknown public IDs are ignored. A newer
coordinate for an assigned target re-emits an attack request even when the
locked assignment itself is unchanged.

Default assumptions are H damage `20`, M damage `5`, hit rates `0.8` for 9400
and `0.6` for 9600, initial health `24/18`, values `5/2`, and safety factor
`1.25`. Minimum packages are planned from expected effective damage. If all
targets cannot be covered, allocation repeatedly serves the largest normalized
shortfall, then value and target ID. Remaining non-reserved missiles use value,
shortfall and ID ordering.

```python
from hm_public_strike_packages_skill import (
    HMMissileSnapshot,
    HMPublicStrikePackagesSkill,
    PublicStrikeConfig,
    materialize_engine_actions,
    public_targets_from_directory,
    track_updates_from_isolated_observations,
)

config = PublicStrikeConfig(reserve_h=1, reserve_m=2, lock_assignments=True)
public_targets = public_targets_from_directory(public_directory, config=config)
skill = HMPublicStrikePackagesSkill(config, public_targets=public_targets)

updates = track_updates_from_isolated_observations(
    isolated_hm_observations, public_targets=public_targets, step=2,
)
decision = skill.decide(step=2, members=hm_snapshots, track_updates=updates)
actions_by_owner = materialize_engine_actions(decision)
```

Engine rows are owner-routed and ordered baseline first, attack second:

```text
[0, missile_id, -1/0/1, 0]  caller's baseline category
[4, missile_id, target_id, 0]  semantic public-target attack request
```
