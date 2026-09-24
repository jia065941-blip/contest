"""Diagnose Stage-2 avoidance failures by initial threat bearing."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policies.red.skills.evasion_ppo.local_env import LEFT, RIGHT, STRAIGHT, VectorLocalAvoidEnv
from policies.red.skills.evasion_ppo.local_ppo import LocalAvoidPPO, summarize_episodes


SECTOR_EDGES = np.asarray([-180.0, -135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0])


def initial_bearing_deg(environment: VectorLocalAvoidEnv) -> np.ndarray:
    values = []
    for env in environment.envs:
        threat = env._threat_values(env.bugs[0])
        values.append(math.degrees(threat["bearing"]))
    return np.asarray(values, dtype=np.float64)


def sector_name(value: float) -> str:
    index = int(np.clip(np.searchsorted(SECTOR_EDGES, value, side="right") - 1, 0, 7))
    return f"[{SECTOR_EDGES[index]:.0f},{SECTOR_EDGES[index + 1]:.0f})"


def summarize_buckets(rows: list[dict]) -> dict[str, dict[str, float]]:
    result = {}
    for lower, upper in zip(SECTOR_EDGES[:-1], SECTOR_EDGES[1:]):
        name = f"[{lower:.0f},{upper:.0f})"
        episodes = [item for item in rows if item["sector"] == name]
        result[name] = summarize_episodes(episodes)
    return result


def evaluate(
    policy: LocalAvoidPPO,
    environment_config,
    *,
    episodes: int,
    seed: int,
    mode: str,
) -> dict:
    num_envs = min(policy.config.num_parallel_envs, episodes)
    environment = VectorLocalAvoidEnv(num_envs, environment_config, stage=2, seed=seed)
    actor_observation, critic_state = environment.reset()
    bearings = initial_bearing_deg(environment)
    fixed_actions = np.where(bearings >= 0.0, LEFT, RIGHT).astype(np.int64)
    completed: list[dict] = []
    selected: list[np.ndarray] = []
    while len(completed) < episodes:
        if mode == "ppo":
            actions = policy.act(
                actor_observation,
                critic_state,
                deterministic=True,
                update_normalizer=False,
            )[0]
        elif mode == "straight":
            actions = np.full(num_envs, STRAIGHT, dtype=np.int64)
        elif mode == "fixed_away":
            actions = fixed_actions.copy()
        else:
            threat_sine = actor_observation[:, 8]
            actions = np.where(threat_sine >= 0.0, LEFT, RIGHT).astype(np.int64)

        actor_observation, critic_state, _rewards, dones, infos = environment.step(actions)
        selected.append(actions.copy())
        for index, done in enumerate(dones.astype(bool)):
            if not done:
                continue
            item = dict(infos[index])
            item["initial_bearing_deg"] = float(bearings[index])
            item["sector"] = sector_name(float(bearings[index]))
            completed.append(item)
            bearings[index] = initial_bearing_deg(environment)[index]
            fixed_actions[index] = LEFT if bearings[index] >= 0.0 else RIGHT

    rows = completed[:episodes]
    return {
        "overall": summarize_episodes(rows, np.concatenate(selected)),
        "by_initial_bearing": summarize_buckets(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/local_avoid_ppo/diagnosis.json",
    )
    args = parser.parse_args()
    policy, environment_config, _payload = LocalAvoidPPO.load(args.model, device=args.device)
    report = {
        "model": str(args.model.resolve()),
        "episodes_per_policy": args.episodes,
        "policies": {
            mode: evaluate(
                policy,
                environment_config,
                episodes=args.episodes,
                seed=args.seed,
                mode=mode,
            )
            for mode in ("ppo", "straight", "fixed_away", "reactive_away")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("LOCAL_AVOID_DIAGNOSIS " + json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
