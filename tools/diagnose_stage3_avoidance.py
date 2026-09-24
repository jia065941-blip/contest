"""Break down Stage-3 avoidance outcomes by encounter pattern and geometry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policies.red.skills.evasion_ppo.local_env import (
    LEFT,
    RIGHT,
    STRAIGHT,
    VectorLocalAvoidEnv,
)
from policies.red.skills.evasion_ppo.local_ppo import LocalAvoidPPO, summarize_episodes


PATTERNS = ("simultaneous", "pincer", "delayed_trail")
DISTANCE_BANDS = ((0.0, 25_000.0), (25_000.0, 35_000.0), (35_000.0, math.inf))


def _distance_band(distance: float) -> str:
    for lower, upper in DISTANCE_BANDS:
        if lower <= distance < upper:
            upper_text = "inf" if math.isinf(upper) else f"{upper / 1000.0:.0f}km"
            return f"[{lower / 1000.0:.0f}km,{upper_text})"
    raise RuntimeError("unreachable distance band")


def _snapshot(environment) -> dict[str, Any]:
    values = [environment._threat_values(bug) for bug in environment.bugs]
    bearings = [math.degrees(item["bearing"]) for item in values]
    distances = [item["distance"] for item in values]
    releases = [int(bug["release_step"]) for bug in environment.bugs]
    return {
        "encounter_pattern": environment.encounter_pattern,
        "initial_bearings_deg": bearings,
        "initial_distances_m": distances,
        "initial_min_distance_m": min(distances, default=math.inf),
        "distance_band": _distance_band(min(distances, default=math.inf)),
        "release_delay_steps": max(releases, default=0),
        "opposite_sides": bool(
            len(bearings) == 2 and bearings[0] * bearings[1] < 0.0
        ),
    }


def _reactive_actions(observations: np.ndarray) -> np.ndarray:
    actions = np.full(observations.shape[0], STRAIGHT, dtype=np.int64)
    for row_index, observation in enumerate(observations):
        candidates = []
        for start in (6, 15):
            if observation[start] <= 0.5:
                continue
            bearing_sine = float(observation[start + 2])
            t_cpa = float(observation[start + 6])
            d_cpa = float(observation[start + 7])
            candidates.append((t_cpa + d_cpa, bearing_sine))
        if candidates:
            _risk, bearing_sine = min(candidates)
            actions[row_index] = LEFT if bearing_sine >= 0.0 else RIGHT
    return actions


def _summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    result = summarize_episodes(rows)
    action_counts = np.sum(
        [item["action_counts"] for item in rows], axis=0, dtype=np.int64
    ) if rows else np.zeros(3, dtype=np.int64)
    action_total = max(int(action_counts.sum()), 1)
    result.update(
        {
            "safe_terminal_rate": sum(not bool(item.get("hit")) for item in rows)
            / max(len(rows), 1),
            "left_action_ratio": float(action_counts[LEFT] / action_total),
            "straight_action_ratio": float(action_counts[STRAIGHT] / action_total),
            "right_action_ratio": float(action_counts[RIGHT] / action_total),
        }
    )
    return result


def evaluate(
    policy: LocalAvoidPPO,
    environment_config,
    *,
    episodes: int,
    seed: int,
    mode: str,
) -> dict[str, Any]:
    num_envs = min(policy.config.num_parallel_envs, episodes)
    environment = VectorLocalAvoidEnv(
        num_envs, environment_config, stage=3, seed=seed
    )
    actor_observation, critic_state = environment.reset()
    snapshots = [_snapshot(item) for item in environment.envs]
    action_counts = np.zeros((num_envs, 3), dtype=np.int64)
    completed: list[dict[str, Any]] = []
    while len(completed) < episodes:
        if mode == "ppo":
            actions = policy.act(
                actor_observation,
                critic_state,
                deterministic=True,
                update_normalizer=False,
            )[0]
        elif mode == "reactive_away":
            actions = _reactive_actions(actor_observation)
        else:
            actions = np.full(num_envs, STRAIGHT, dtype=np.int64)
        action_counts[np.arange(num_envs), actions] += 1
        actor_observation, critic_state, _rewards, dones, infos = environment.step(actions)
        for index, done in enumerate(dones.astype(bool)):
            if not done:
                continue
            row = dict(infos[index])
            row.update(snapshots[index])
            row["action_counts"] = action_counts[index].copy()
            completed.append(row)
            action_counts[index] = 0
            snapshots[index] = _snapshot(environment.envs[index])

    rows = completed[:episodes]
    return {
        "overall": _summarize(rows),
        "by_pattern": {
            pattern: _summarize(
                [item for item in rows if item["encounter_pattern"] == pattern]
            )
            for pattern in PATTERNS
        },
        "by_distance_band": {
            _distance_band(lower): _summarize(
                [item for item in rows if item["distance_band"] == _distance_band(lower)]
            )
            for lower, _upper in DISTANCE_BANDS
        },
        "by_side_relation": {
            name: _summarize(
                [item for item in rows if item["opposite_sides"] is opposite]
            )
            for name, opposite in (("same_side", False), ("opposite_sides", True))
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20269901)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/local_avoid_ppo/stage3_diagnosis.json",
    )
    args = parser.parse_args()
    policy, environment_config, _payload = LocalAvoidPPO.load(
        args.model, device=args.device
    )
    report = {
        "model": str(args.model.resolve()),
        "episodes_per_policy": args.episodes,
        "seed": args.seed,
        "policies": {
            mode: evaluate(
                policy,
                environment_config,
                episodes=args.episodes,
                seed=args.seed,
                mode=mode,
            )
            for mode in ("ppo", "reactive_away", "straight")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("LOCAL_AVOID_STAGE3_DIAGNOSIS " + json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
