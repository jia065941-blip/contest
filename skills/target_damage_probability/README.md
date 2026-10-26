# Target Damage Probability Skill

Computes target destruction probability and deterministic damage-range
diagnostics from caller-owned evidence only. The package is a pure Python
DTO/state component and does not import an environment, simulator, or global
game state.

## Damage model

Only these missile/target pairs are valid:

| Missile | Target | Lower | Nominal | Upper |
| --- | --- | ---: | ---: | ---: |
| H (`21000`) | `9400` | 9.6 | 16 | 19.2 |
| H (`21000`) | `9600` | 12 | 12 | 12 |
| M (`21001`) | `9400` | 2.4 | 4 | 4.8 |
| M (`21001`) | `9600` | 3 | 3 | 3 |
| L (`21002`) | `9500` | 0.8 | 0.8 | 0.8 |

These are deterministic conditional-damage diagnostics. Applied damage is
`base_damage * adjusted_rate`; the lower, nominal, and upper values are not hit
rates or probabilities. For `9400`, the H base `20` and M base `5` use adjusted
rates `0.48`, `0.8`, and `0.96`; the other supported pairs use their single
fixed adjusted rate. H-to-`9400` damage is never assumed to be `20` unless that
delivery explicitly supplies `satellite_active=True` (adjusted rate `1.0`). In
that case its conditional damage is exactly `20`. `adjusted_damage_override`
supplies an exact conditional damage value and takes precedence over the
standard interval. An exact `HitReceipt.applied_damage` takes precedence over
all delivery fields.

Delivery probabilities default to `0.98` within 500 m, `0.85` at stage 4 or
later, `0.65` within 1200 m, `0.15` for a farther disappearance, and `0.0`
otherwise. `DeliveryEvidence.probability` explicitly overrides that heuristic.
It is the probability that the delivery occurred and its deterministic damage
was applied; it is not a hit rate and does not scale `adjusted_rate`.

`destruction_probability` convolves independent delivery/non-delivery outcomes
using nominal deterministic damage and checks the resulting totals against the
health threshold. It is therefore a threshold probability under delivery
uncertainty, not a hit probability. A receipt replaces delivery uncertainty for
the same missile with exact applied damage.

`lower_expected_damage`, `nominal_expected_damage`, and
`upper_expected_damage` weight each conditional-damage bound by its delivery
probability. `nominal_expected_damage` is the same value retained under the
existing `delivered_expected_damage` name. The corresponding `projected_*`
fields additionally include active commitments.

`projected_destruction_probability` additionally includes active `core` and
`flexible` commitments. Their probabilities are configured independently when
constructing `TargetDamageProbabilitySkill` (defaults: core `1.0`, flexible
`0.5`). Projection is always reported separately and never marks a target as
confirmed destroyed.

## Interface

```python
from target_damage_probability_skill import (
    CommitmentEvidence,
    DeliveryEvidence,
    TargetDamageProbabilitySkill,
    TargetPriorEvidence,
)

tracker = TargetDamageProbabilitySkill(
    core_commitment_probability=0.9,
    flexible_commitment_probability=0.4,
)
tracker.register(TargetPriorEvidence(52, 9400, 24.0))
tracker.commit(CommitmentEvidence(1, 21000, 52, "core"))
tracker.report_delivery(
    DeliveryEvidence(
        1,
        21000,
        52,
        100,
        4,
        75.0,
        event_id="delivery:1",
        satellite_active=False,
    )
)
belief = tracker.estimate(52)
```

State operations are `register`, `commit`, `transfer`, `release`,
`report_delivery`, `report_receipt`, `confirm`, `reset`, and `snapshot`.
`export` returns an immutable `EvidenceSnapshot`; `from_snapshot` and `replay`
rebuild equivalent state for an orchestrator-owned communication domain.
The default delivery heuristic treats a final observed target distance of
`500 m` or less as `0.98` delivery probability. This accounts for the official
one-second observation cadence; stage 4 uses `0.85`, distance up to `1200 m`
uses `0.65`, and a farther disappearance uses `0.15`.
