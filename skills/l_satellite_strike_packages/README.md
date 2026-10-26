# L Satellite Strike Packages

`LSatelliteStrikePackagesSkill` uses one designated L missile to request the
global satellite, accepts hidden-nest tracks only from caller-supplied isolated
observations, and assigns L missiles to the discovered `9500` target IDs.

The skill does not deploy missiles, control H/M missiles, read simulator state,
or replace the local avoidance policy. Caller-supplied maneuver categories are
preserved as semantic actions and a separate adapter routes them by owner.

The default `distance_balanced` allocation keeps total package quotas balanced,
balances wave 1 and wave 2 quotas independently, then minimizes approximate
WGS84 distance within each wave. Matching is deterministic regardless of input
order. Legacy ID-ordered `equal` allocation, `fixed_size` packages, reserves,
package limits, an expected target count, and reassignment after member loss
are configurable through `SatelliteStrikeConfig`.

```python
SatelliteStrikeConfig(
    satellite_trigger_step=1,
    satellite_leader_id=None,       # lowest active L ID
    allocation_mode="distance_balanced",  # or "equal" / "fixed_size"
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

With `distance_balanced`, quotient/remainder allocation gives every target a
total quota differing by at most one and does the same separately for both
waves. A deterministic minimum-cost assignment then fills those quotas using
haversine distance on a WGS84 mean-radius approximation. With `equal`, the
legacy ID ordering gives the lowest target IDs one extra member. With
`fixed_size`, every selected target must receive the exact configured size or
the decision fails closed. Reserved or maximum-capped L missiles are returned
in `unassigned_member_ids`.

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
    members=(
        LMissileSnapshot(101, baseline_action=1, lon=120.1, lat=30.2, wave=1),
        LMissileSnapshot(102, baseline_action=0, lon=120.2, lat=30.3, wave=2),
    ),
    tracks=tracks,
)
actions_by_entity = materialize_engine_actions(decision)
```

`LMissileSnapshot.lon` and `.lat` must be finite and `wave` must be `1` or
`2`. All three fields have backward-compatible defaults (`0.0`, `0.0`, and
`1`), and the existing positional arguments remain unchanged.

Engine action rows emitted by the skill are:

```text
[3, leader_id, 0, 0]       request the global satellite once
[4, missile_id, target_id, 0]  attack a legally received hidden target
```
