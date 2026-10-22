# Local Avoidance

Network-agnostic runtime bridge for a caller-supplied local-avoidance checkout
and checkpoint. This package ships no model, checkpoint, architecture, V13 hash
gate, or fixed bundle. The external runtime must provide
`policies/red/skills/evasion_ppo/local_runtime.py` with a
`LocalAvoidRuntimePolicy` implementing `load`, `set_training`, observation
encoding, action masking, and three-action inference.

```python
from local_avoidance_skill import load_external_policy

policy = load_external_policy(
    "/path/to/contest-runtime",
    "/path/to/trained-network.pt",
    max_steps=3000,
)
```

The returned policy remains owned by the caller and should be closed when the
episode runner exits if it exposes `close()`.
