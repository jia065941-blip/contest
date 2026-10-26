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
`1.25`. The default `allocation_mode="full_coverage_mixed"` treats the
environment's applied damage as deterministic: H applies `16/12` to 9400/9600
and M applies `4/3`. The hit-rate fields are multipliers in this planner, not
Bernoulli trials.

The default planner uses every non-reserve missile and follows these steps:

- Seed each 9400 mother with `min_h_per_mother` H in value/target-ID rounds.
  V12 uses `2`. M is not sent to a mother while sentries are available.
- Visit every 9600 sentry by `target_priority_axis` (`east` by default) and give
  it `1H` before any sentry is upgraded. Scarce H therefore produces a stable
  priority prefix without upgrading one target while another could be seeded.
- Use remaining H to upgrade the farthest/highest-priority `1H` sentries to
  `h_per_sentry_package=2`. Each `2H` package applies `24 >= 18`.
- Give each remaining `1H` sentry `m_per_h_mixed_package=2` M when inventory
  permits. A `1H+2M` package applies exactly `12 + 2*3 = 18`.
- If H could not seed every sentry, use groups of
  `m_only_sentry_package=6` M to cover additional zero-H sentries after the
  cheaper mixed packages are complete.
- Balance remaining M by current M count, target priority, and target ID. When
  `1H` sentries exist, only those targets receive the tail; no M is wasted on a
  `2H` sentry. If every sentry already has `2H`, M is balanced as redundancy.
- Every non-reserve missile is assigned whenever at least one public target is
  available.

`PublicStrikePackage.nominal_applied_damage` reports this deterministic damage;
`threshold_satisfied` compares it with immutable `initial_health`. The standard
inventory tests cover these plans:

- Easy V12 (`17H/24M`): four mothers get `2H`; sentries receive
  `2H,2H,1H,1H,1H,1H,1H`, with M counts `0,0,5,5,5,5,4`. All seven sentries
  meet nominal damage.
- Medium V12 (`27H/37M`): six mothers get `2H`; three sentries get `2H` and nine
  get `1H`, with M counts `5,4,4,4,4,4,4,4,4` on the `1H` targets. All twelve
  sentries meet nominal damage.
- Hard V12 (`48H/66M`): eight mothers get `2H`; twelve sentries get `2H` and
  eight get `1H`, with M counts `9,9,8,8,8,8,8,8` on the `1H` targets. All
  twenty sentries meet nominal damage.

The prior planners remain available without behavior changes:

- `allocation_mode="deterministic_thresholds"` retains the v0.8 sequential
  complete-package behavior and `allow_m_only_sentry_packages` option.
- `allocation_mode="threshold_packages"` retains the v0.6 H-package and
  `m_target_policy` allocation.
- `allocation_mode="kill_probability"` retains exact Bernoulli
  probability-marginal allocation.
- `allocation_mode="expected_damage"` retains safety-factor and expected-damage
  allocation.

Every package still reports the legacy `kill_probability` diagnostic. For
example, `kill_probability(2, 0, 9600)` is `0.84`,
`kill_probability(3, 0, 9600)` is `0.936`, and
`kill_probability(2, 2, 9400)` is `0.9472`.

```python
from hm_public_strike_packages_skill import (
    HMMissileSnapshot,
    HMPublicStrikePackagesSkill,
    PublicStrikeConfig,
    kill_probability,
    materialize_engine_actions,
    public_targets_from_directory,
    track_updates_from_isolated_observations,
)

config = PublicStrikeConfig(
    reserve_h=1,
    reserve_m=2,
    min_h_per_mother=2,
    h_per_sentry_package=2,
    m_per_h_mixed_package=2,
    m_only_sentry_package=6,
    target_priority_axis="east",
    lock_assignments=True,
)
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
