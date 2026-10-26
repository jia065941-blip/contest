# Reserve Echelon

`ReserveEchelonSkill` is a pure, deterministic second-echelon state machine. It
emits semantic package-level target changes only; callers retain command
ownership and engine-action routing.

Version `0.2.0` supports atomic H-only reserve packages against public `9400`
and `9600` deficits, and atomic L-only packages against dark `9500` deficits.
It adds per-member physical reachability gates while retaining the type vectors
needed for future H/M package support. Mixed or M-containing packages still fail
closed in this release.

## State Machine

Every package starts in `STAGING` and reaches exactly one terminal state:

```text
STAGING -- eligible deficit --> DISPATCHED
STAGING -- deadline, no eligible deficit --> FALLBACK
```

State is retained by the skill across `decide` calls. Repeating stale caller
snapshots cannot dispatch a terminal package again. `reset()` clears all
registered definitions and state; the read-only `state` property returns the
current package snapshots ordered by package ID.

The whole package must pass every gate. All members must be below the
dispatch/retarget limit, name the target in their `legal_target_ids`, and share
one group. A deficit must match the package responsibility region and package
type. With same-group evidence enabled, at least one `evidence_ids` entry must
identify an input member in the package's group. The caller remains responsible
for supplying authorized deficit evidence.

The default `max_dispatch_stage=None` does not impose a stage gate, so stage 3
and 4 reserves remain dispatchable when physically reachable. Supplying an
integer retains the v0.1 behavior: any package member above that stage rejects
the whole package with `stage_too_late`.

Every logically eligible destination is then checked from every member's
position. H members default to an `80 km` maximum distance and `60 degree`
maximum turn; L members default to `100 km` and `75 degree`. `heading_east` and
`heading_north` are direction components in the same local east/north frame as
`position`. Both may be omitted for construction compatibility, but a nonzero
trip then rejects with `missing_heading`; a zero heading does the same.

When `remaining_steps` is present, estimated travel steps are calculated as:

```text
distance_km * 1000 / nominal_speed_mps / seconds_per_step
```

The estimate must not exceed that member's remaining steps. The default nominal
speeds are the environment values `200 m/s` for H and `100 m/s` for L, and both
are configurable. `remaining_steps=None` disables only the time-budget check;
distance and turn checks still apply. Every member must pass for an atomic
dispatch.

Eligible targets are ranked from the package position (the member-position
centroid) using:

```text
score = weight * missing_damage / (1 + distance_km)
```

Higher score wins. The deficit's supplied `priority`, shorter distance, and
lower target ID are deterministic tie breakers in that order. Positions are
two-dimensional kilometre coordinates. The legacy
`max_target_distance_km` remains available as an additional package-wide cap;
when set, the smaller of it and the member-type distance limit is used.

At or after `deadline_step`, the skill still tries reachable deficit candidates
first. If none is reachable, the prevalidated legal `fallback_target_id` is
checked with exactly the same distance, turn, and remaining-time gates before
an atomic fallback. Set `fallback_position` on the package to provide its local
kilometre coordinates. For compatibility, the skill can also obtain that
position from an input deficit with the fallback ID. Without either source it
fails closed with `fallback_position_unavailable`. Before the deadline the
package remains staged.

`Decision.rejected_reasons` reports `distance_exceeded`,
`turn_angle_exceeded`, `insufficient_time`, or `missing_heading` when physical
reachability prevents selection.

## Configuration

```python
ReserveEchelonConfig(
    max_dispatch_stage=None,
    max_dispatches=1,
    deadline_step=300,
    require_same_group_evidence=True,
    max_target_distance_km=None,
    h_max_distance_km=80.0,
    l_max_distance_km=100.0,
    h_max_turn_angle_degrees=60.0,
    l_max_turn_angle_degrees=75.0,
    seconds_per_step=1.0,
    h_nominal_speed_mps=200.0,
    l_nominal_speed_mps=100.0,
)
```

## Usage

```python
from reserve_echelon_skill import (
    ReserveEchelonSkill,
    ReserveMemberSnapshot,
    ReservePackageSnapshot,
    TargetDeficitSnapshot,
)

member = ReserveMemberSnapshot(
    id=10,
    type=21000,
    stage=1,
    group="west",
    current_target=900,
    legal_target_ids=(900, 901, 999),
    position=(0.0, 0.0),
    retarget_count=0,
    heading_east=1.0,
    heading_north=0.0,
    remaining_steps=100,
)
package = ReservePackageSnapshot(
    package_id=1,
    member_ids=(10,),
    member_types=(21000,),
    state="STAGING",
    provisional_target_id=900,
    fallback_target_id=999,
    responsibility_region="west",
    dispatch_count=0,
    fallback_position=(10.0, 0.0),
)
deficit = TargetDeficitSnapshot(
    target_id=901,
    type=9400,
    weight=5.0,
    missing_damage=20.0,
    priority=0.0,
    position=(3.0, 4.0),
    responsibility_region="west",
    evidence_ids=(10,),
)

decision = ReserveEchelonSkill().decide(
    step=100,
    members=(member,),
    deficits=(deficit,),
    packages=(package,),
)
assert decision.requests[0].member_ids == (10,)
assert decision.packages[0].state == "DISPATCHED"
```
