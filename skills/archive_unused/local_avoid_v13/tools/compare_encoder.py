"""Compare the self-contained encoder with the author's deployment encoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from local_avoid_v13_skill.encoder import EnvironmentConfig, LocalAvoidRuntimeEncoder


def observation(step: int, y: float, threats: int) -> dict:
    detect = {}
    for index in range(threats):
        track_id = 100 + index
        detect[str(track_id)] = {
            "entity_id": track_id,
            "entity_type": 24000,
            "detect_from": 7 if index == 0 else 999,
            "time": step,
            "pos_ecf": {"x": 6_378_137.0, "y": y + 600.0 + index * 250.0, "z": 0.0},
            "vel_ecf": {"x": 0.0, "y": -200.0 + index * 50.0, "z": 0.0},
        }
    return {
        "step": step,
        "self": {
            "type": 21000, "health": 100.0, "stage": 1,
            "position": {"lon": 0.0, "lat": 0.0, "alt": 1000.0},
            "pos_ecf": {"x": 6_378_137.0, "y": y, "z": 0.0},
            "detectInfo": detect, "commRangeInfo": [999],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contest-root", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.contest_root.resolve()))
    from policies.red.skills.evasion_ppo.local_env import LocalAvoidConfig
    from policies.red.skills.evasion_ppo.local_runtime import LocalAvoidRuntimeEncoder as Original

    payload = json.loads((args.bundle / "artifacts/environment_config.json").read_text())
    raw_config = payload["environment_config"]
    original_config = LocalAvoidConfig(**{
        **raw_config,
        "unit_speeds_mps": tuple(raw_config["unit_speeds_mps"]),
        "unit_turn_rates_deg_s": tuple(raw_config["unit_turn_rates_deg_s"]),
    })
    converted_config = EnvironmentConfig.from_dict(raw_config)
    original = Original(entity_id=7, max_steps=1200, config=original_config)
    converted = LocalAvoidRuntimeEncoder(entity_id=7, max_steps=1200,
                                         config=converted_config)
    cases = [(10, 0.0, 0), (11, 200.0, 1), (12, 400.0, 2), (13, 600.0, 3)]
    maximum = 0.0
    for step, y, threat_count in cases:
        raw = observation(step, y, threat_count)
        expected = original.encode(raw, launch_step=5, maneuver_state=0,
                                   goal_position={"lon": 0.1, "lat": 0.0})
        actual = converted.encode(raw, launch_step=5, maneuver_state=0,
                                  goal_position={"lon": 0.1, "lat": 0.0})
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
        maximum = max(maximum, float(np.max(np.abs(actual - expected))))
    print(f"equivalent_encoder_cases={len(cases)} max_abs_error={maximum:.9g}")


if __name__ == "__main__":
    main()
