# L Satellite Strike Packages

`LSatelliteStrikePackagesSkill` uses one designated L missile to request the
global satellite, accepts hidden-nest tracks only from caller-supplied isolated
observations, and assigns L missiles to the discovered `9500` target IDs.

The skill does not deploy missiles, control H/M missiles, read simulator state,
or replace the local avoidance policy. Caller-supplied maneuver categories are
preserved as semantic actions and a separate adapter routes them by owner.

The default allocation sorts targets and missiles by ID and distributes all
available L missiles as evenly as possible. `fixed_size` packages, reserves,
package limits, an expected target count, and reassignment after member loss
are configurable through `SatelliteStrikeConfig`.

```python
SatelliteStrikeConfig(
    satellite_trigger_step=1,
    satellite_leader_id=None,       # lowest active L ID
    allocation_mode="equal",       # or "fixed_size"
    package_size=None,              # required by fixed_size
    package_count=None,             # None means every admitted target
    reserve_count=0,
    reserve_fraction=0.0,           # mutually exclusive with reserve_count
    min_package_size=1,
    max_package_size=None,
    expected_target_count=None,     # wait until this many tracks are admitted
    lock_assignments=True,
    reassign_lost_members=False,    # requires lock_assignments=False
)
```

With `equal`, quotient/remainder allocation gives the lowest target IDs one
extra member. With `fixed_size`, every selected target must receive the exact
configured size or the decision fails closed. Reserved or maximum-capped L
missiles are returned in `unassigned_member_ids`.

```python
from l_satellite_strike_packages_skill import (
    LMissileSnapshot,
    LSatelliteStrikePackagesSkill,
    SatelliteStrikeConfig,
    materialize_engine_actions,
    tracks_from_isolated_observations,
)

skill = LSatelliteStrikePackagesSkill(SatelliteStrikeConfig())
tracks = tracks_from_isolated_observations(observations, step=1)
decision = skill.decide(
    step=1,
    members=(LMissileSnapshot(101, baseline_action=1), LMissileSnapshot(102, baseline_action=0)),
    tracks=tracks,
)
actions_by_entity = materialize_engine_actions(decision)
```

Engine action rows emitted by the skill are:

```text
[3, leader_id, 0, 0]       request the global satellite once
[4, missile_id, target_id, 0]  attack a legally received hidden target
```
