# Staggered Deployment Skill

Produces deterministic launch20 deployment coordinates only. It places L
platforms (type `21002`) in a front band and H/M platforms (types `21000` and
`21001`) in a rear band, without target assignment, launch timing, or combat
planning.

The default front axis is east. Longitude distances use the cosine of the
deployment rectangle's midpoint latitude. The rear band's front edge is kept
at least `deployment_gap_km` behind the L band's rear edge. If the rectangle is
only 40 km deep, the rear band is shortened as needed; 50 km and 60 km regions
retain the configured depths. Each band expands from one column into a grid at
`lateral_spacing_km`, and rejects requests that cannot fit without crossing a
boundary.

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
        PlatformDeploymentRequest(1, 21000),
        PlatformDeploymentRequest(2, 21001),
        PlatformDeploymentRequest(3, 21002),
    ),
)
actions = to_deployment_actions(plan)
# ((0, 1, lon, lat, 1000.0), ...)
```

`route_deployment_actions(plan, owners)` returns the same action rows grouped
by caller-supplied owner. Neither the core nor the adapter imports an
environment or simulator.

## Tests

```bash
cd /home/amax/ry/competition/skill_generation/skills/staggered_deployment
PYTHONPATH=src ../../.venv/bin/python -m pytest -q
```
