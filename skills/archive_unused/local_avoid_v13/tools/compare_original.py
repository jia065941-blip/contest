"""Compare the converted Backend with the author's Actor implementation safely."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from export_v13_bundle import _safe_payload
from local_avoid_v13_skill import LocalAvoidV13Skill


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contest-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=256)
    args = parser.parse_args()
    sys.path.insert(0, str(args.contest_root.resolve()))
    from policies.red.skills.evasion_ppo.local_ppo import LocalAvoidPPO, LocalPPOConfig

    payload = _safe_payload(args.checkpoint)
    config_values = dict(payload["ppo_config"])
    config_values["hidden_dims"] = tuple(config_values["hidden_dims"])
    config_values["device"] = "cpu"
    original = LocalAvoidPPO(LocalPPOConfig(**config_values))
    original.network.load_state_dict(payload["network"])
    original.actor_rms.load_state_dict(payload["actor_rms"])
    original.network.eval()
    rng = np.random.default_rng(20261015)
    observations = np.concatenate((
        np.zeros((1, 33), dtype=np.float32),
        np.ones((1, 33), dtype=np.float32),
        rng.uniform(-2.0, 2.0, size=(args.samples, 33)).astype(np.float32),
    ))
    max_error = 0.0
    with LocalAvoidV13Skill(args.bundle) as converted:
        for observation in observations:
            normalized = original._normalize_actor(observation, False)
            with torch.inference_mode():
                expected = original.network.actor_logits(torch.from_numpy(normalized)).numpy()
            actual = converted.infer({"observation": observation}).outputs["logits"]
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
            if int(np.argmax(actual)) != int(np.argmax(expected)):
                raise AssertionError("Actor action mismatch")
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
    print(f"equivalent_samples={len(observations)} max_abs_error={max_error:.9g}")


if __name__ == "__main__":
    main()
