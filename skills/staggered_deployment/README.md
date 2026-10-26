# Staggered Deployment Skill

Produces deterministic launch20 deployment coordinates only. By default it
marks the lowest-ID 70% of L platforms (type `21002`) as the `front` wave. The
remaining L platforms use the `second` band marker. H is allocated first, M is
allocated second, and both L waves are allocated last from the remaining global
deployment grid. L prefers the former front/follow regions for the respective
waves, then falls back anywhere in the legal rectangle rather than displacing
H/M. H and M retain the compatible `rear` band marker and existing `wave_role`
values. This creates a natural second-wave delay without assigning launch timing
or combat actions.

The default front axis is east and the default `deployment_gap_km` is 15 km.
Longitude distances use the cosine of the deployment rectangle's midpoint
latitude. Behind the 5 km front-L band, the default layout has a 15 km gap, a
5 km forward H/M band, no empty H/M gap, and a 10 km rear H/M/second-L band.
These defaults require 35 km and fit an exactly 40 km front axis. Untargeted,
coordinate-only, and fixed-rendezvous placements retain these grids. Adaptive
`9600` M formations may use the complete legal forward extent of the deployment
rectangle, including the preferred L range, which permits initial H/M
separations greater than 20 km. Exact coordinates remain unique. Each region
expands from one column into a grid at `lateral_spacing_km`, and rejects requests
that cannot fit without crossing a boundary. `front_axis` accepts `east`,
`west`, `north`, and `south`.

The corresponding options are `h_depth_km=5`, `m_depth_km=10`, and
`hm_follow_gap_km=0`. The former `hm_depth_km` option remains as a deprecated
alias for `m_depth_km`, including its original positional argument slot.

The L split is deterministic by ascending `entity_id` and is controlled by
`l_first_wave_fraction` in the strict range `(0, 1]` (default `0.70`). Its count
is rounded up so a non-empty L set always has a first wave. All H/M coordinates
are reserved before either L wave is spread across the remaining global grid,
so adding L does not move an H/M placement and cannot overlap one.

`PlatformDeploymentRequest` accepts optional `target_lon` and `target_lat`, plus
the paired public identity fields `target_id` and `target_type`. Coordinates
must be paired, finite, and valid WGS84 values. Identity fields must be paired,
require coordinates, accept only public types `9400`/`9600`, and are valid only
on H/M requests. Coordinate-only requests remain supported for callers that
have not yet supplied target identity.

Two or three H platforms assigned to one `9400` mother use crossfire placement.
They share a forward column and occupy `-offset,+offset` or
`-offset,0,+offset` around the target's lateral center. The requested crossfire
angle defaults to 35 degrees, while the offset is capped by the bounds and
`mother_crossfire_max_offset_km=35`. If a 30-degree angle cannot be achieved,
the largest legal offset is retained. `DeploymentPoint.crossfire_role` and
`crossfire_angle` report the role and achieved geometric angle. A formation for
a target on the exact lateral boundary is shifted inward to keep coordinates
unique.

When a `9600` target has both H and M members, the default
`sentry_rendezvous_mode="adaptive"` first reserves the rearmost legal H
formation. It measures the H-centroid-to-target great-circle distance `D` and
requests this M lead in kilometres:

```text
(1 - m_speed_mps / h_speed_mps) * D
    + m_speed_mps / 1000 * desired_m_arrival_lead_seconds
```

The defaults are `h_speed_mps=200`, `m_speed_mps=120`, and
`desired_m_arrival_lead_seconds=5`, so the request is `0.4 * D + 0.6`.
`min_sentry_m_lead_km=10` supplies the policy floor.
`max_sentry_m_lead_km=None` leaves the upper limit to the legal deployment
geometry; a finite value supplies an explicit policy cap. M is placed at the
requested forward centroid as precisely as possible in the same target corridor,
with lateral centroids within 3 km. Only rectangle boundaries, coordinate
occupancy/capacity, and the corridor constrain adaptive geometry; the former M
follow band does not. Earlier H/M reservations and uniqueness are always
respected.

Set `sentry_rendezvous_mode="fixed"` to retain the v0.6 band placement and
`sentry_m_lead_km=7.5` default. That option still accepts finite values in
`[0, 15]`. A `9600` target with only H or only M retains the compatible legacy
band in either mode.

`DeploymentPoint.formation_role` reports `m_lead` or `h_follow` for paired
`9600` formations. `desired_separation_km` reports the policy-adjusted lead and
`actual_separation_km` reports the achieved forward centroid separation. The
same points also expose `requested_lead` (the raw distance formula),
`actual_lead`, `predicted_m_arrival_delta_seconds`,
`predicted_separation_at_m_20km`, `geometry_clamped`, and `clamp_reason`.
`clamp_reason` is `boundary`, `capacity`, or `corridor` when geometry is
constrained, and otherwise `None`. Arrival delta is
`H arrival time - M arrival time`, so the ideal default is `+5` seconds. At the
instant M is 20 km from its target, ideal geometry predicts about `14.33` km of
H/M separation. `geometry_clamped` is true only when bounds or occupancy prevent
the policy-adjusted request; applying the configured minimum or maximum alone is
not a geometry clamp. These diagnostics remain `None` for unpaired targets and
`9400` mother formations. H/M cells, including unchanged `9400` crossfire
reservations, are allocated before both L waves use remaining global-grid cells.
All target-aware geometry supports east, west, north, and south front axes.

```python
from staggered_deployment_skill import (
    DeploymentBounds,
    PlatformDeploymentRequest,
    StaggeredDeploymentSkill,
    to_deployment_actions,
)

plan = StaggeredDeploymentSkill().plan(
    DeploymentBounds(119.7, 120.3, 29.8, 30.2, altitude=1000.0),
    (
        PlatformDeploymentRequest(
            1, 21000, target_lon=120.2, target_lat=30.1,
            target_id=901, target_type=9400,
        ),
        PlatformDeploymentRequest(
            2, 21001, target_lon=119.8, target_lat=29.9,
            target_id=902, target_type=9600,
        ),
        PlatformDeploymentRequest(3, 21002),
    ),
)
actions = to_deployment_actions(plan)
# ((0, 1, lon, lat, 1000.0), ...)
```

`route_deployment_actions(plan, owners)` returns the same action rows grouped
by caller-supplied owner. Neither the core nor the adapter imports an
environment or simulator. The action ABI remains the five-item tuple
`(0, entity_id, longitude, latitude, altitude)`; compatible band metadata stays
on `DeploymentPoint.band`, while the precise layer is in
`DeploymentPoint.wave_role`.

## Tests

```bash
cd /home/amax/ry/competition/skill_generation/skills/staggered_deployment
PYTHONPATH=src ../../.venv/bin/python -m pytest -q
```
