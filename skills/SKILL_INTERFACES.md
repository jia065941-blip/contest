# Active Simple-Upper Interfaces

The active stack is deliberately split by lifecycle and authority.

| Package | Lifecycle | Authority |
|---|---|---|
| `staggered_deployment` | Pre-combat deployment | H/M/L initial coordinates only |
| `l_satellite_strike_packages` | Combat planning | One satellite request and L-to-9500 assignments |
| `hm_public_strike_packages` | Combat planning | H/M-to-public-9400/9600 assignments |
| `local_avoidance` | Combat motion | Caller-supplied three-action local policy |
| `simple_upper_orchestrator` | Official runtime adapter | Owner routing and lifecycle composition |

Deployment rows are five-column arrays:

```text
[0, entity_id, longitude, latitude, altitude]
```

Combat rows are four-column arrays:

```text
[0, entity_id, acceleration, 0]
[3, entity_id, 0, 0]
[4, entity_id, target_id, 0]
```

The orchestrator consumes only official isolated observations. Public 9400/9600
catalogue information is available fleet-wide. A hidden 9500 target command is
emitted only when that target ID exists in the command owner's current
`self.detectInfo`. The official environment remains responsible for
communication-domain construction and satellite fusion.

`local_avoidance` ships no checkpoint or fixed network. The caller supplies a
compatible runtime root and trained checkpoint at launch.
