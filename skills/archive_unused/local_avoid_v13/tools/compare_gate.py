"""Compare mission-v12 masks with the author's original gate implementation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from local_avoid_v13_skill.encoder import EnvironmentConfig
from local_avoid_v13_skill.gate import prepare_action_mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contest-root", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    sys.path.insert(0, str(args.contest_root.resolve()))
    from policies.red.skills.evasion_ppo.local_env import LocalAvoidConfig
    from policies.red.skills.evasion_ppo.local_runtime import LocalAvoidRuntimePolicy

    raw = json.loads((args.bundle / "artifacts/environment_config.json").read_text())["environment_config"]
    original_config = LocalAvoidConfig(**{
        **raw, "unit_speeds_mps": tuple(raw["unit_speeds_mps"]),
        "unit_turn_rates_deg_s": tuple(raw["unit_turn_rates_deg_s"]),
    })
    converted_config = EnvironmentConfig.from_dict(raw)
    original = LocalAvoidRuntimePolicy()
    original.environment_config = original_config
    original._current_agent_id = 1
    rng = np.random.default_rng(20261015)
    for _ in range(args.samples):
        observation = rng.uniform(-2.0, 2.0, 33).astype(np.float32)
        observation[3] = rng.uniform(0.0, 2.0)
        observation[29] = rng.uniform(0.0, 1.0)
        observation[30:33] = 0.0
        observation[30 + int(rng.integers(0, 3))] = 1.0
        for offset in (6, 15):
            observation[offset] = float(rng.integers(0, 2))
            observation[offset + 1] = rng.uniform(0.0, 2.0)
            observation[offset + 5] = rng.uniform(-2.0, 2.0)
            observation[offset + 6] = rng.uniform(0.0, 2.0)
            observation[offset + 7] = rng.uniform(0.0, 2.0)
        pending = bool(rng.integers(0, 2))
        original._recovery_pending[1] = pending
        expected = original.prepare_action_mask(observation, np.ones(3, dtype=np.bool_))
        actual, actual_pending = prepare_action_mask(observation, converted_config, pending)
        np.testing.assert_array_equal(actual, expected)
        if actual_pending != original._recovery_pending.get(1, False):
            raise AssertionError("Recovery state mismatch")
    print(f"equivalent_gate_cases={args.samples}")


if __name__ == "__main__":
    main()
