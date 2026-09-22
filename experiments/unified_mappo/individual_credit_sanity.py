"""ICM000：智能体级奖励守恒与反事实信用的数值检验。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for module_root in (PROJECT_ROOT, PROJECT_ROOT / "core"):
    module_path = str(module_root)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

REWARD_MODULE_PATH = (
    PROJECT_ROOT / "core" / "envengine" / "environment" / "individual_reward.py"
)
reward_spec = importlib.util.spec_from_file_location(
    "individual_reward_standalone", REWARD_MODULE_PATH
)
if reward_spec is None or reward_spec.loader is None:
    raise ImportError(f"无法加载奖励模块: {REWARD_MODULE_PATH}")
reward_module = importlib.util.module_from_spec(reward_spec)
sys.modules[reward_spec.name] = reward_module
reward_spec.loader.exec_module(reward_module)
allocate_direct_credit = reward_module.allocate_direct_credit
target_reward_vector = reward_module.target_reward_vector
from experiments.unified_mappo.credit_assignment import (
    normalized_counterfactual_credit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="智能体级信用数值检验")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    initial = {51: 100.0, 54: 100.0}
    previous = {51: {"health": 100.0}, 54: {"health": 100.0}}
    current = {51: {"health": 80.0}, 54: {"health": 0.0}}
    target_rewards_dict = target_reward_vector(
        current_entities=current,
        previous_entities=previous,
        initial_health=initial,
        objective_weights={51: 5.0, 54: 2.0},
    )
    direct = allocate_direct_credit(
        target_rewards=target_rewards_dict,
        hit_events={
            51: (
                {"id": 1001, "damage_point": 16.0},
                {"id": 1002, "damage_point": 4.0},
            ),
            54: ({"id": 1003, "damage_point": 100.0},),
        },
        entity_to_agent={1001: 1, 1002: 2, 1003: 3, 1004: 4},
        agent_ids=(1, 2, 3, 4),
    )

    target_rewards = torch.tensor(
        [[target_rewards_dict[51], target_rewards_dict[54]]],
        dtype=torch.float64,
        device=device,
    )
    factual_q = torch.tensor(
        [[0.8, 0.6]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    counterfactual_q = torch.tensor(
        [[[0.4, 0.6], [0.6, 0.6], [0.8, 0.1], [0.8, 0.6]]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    candidate_mask = torch.tensor(
        [[[1, 0], [1, 0], [0, 1], [0, 0]]],
        dtype=torch.bool,
        device=device,
    )
    direct_damage = torch.zeros_like(counterfactual_q)
    credit = normalized_counterfactual_credit(
        target_rewards=target_rewards,
        factual_q=factual_q,
        counterfactual_q=counterfactual_q,
        candidate_mask=candidate_mask,
        direct_damage=direct_damage,
    )
    objective = -credit.agent_rewards[0, 0]
    objective.backward()
    gradients = torch.cat(
        (
            factual_q.grad.flatten(),
            counterfactual_q.grad.flatten(),
        )
    )

    fallback = normalized_counterfactual_credit(
        target_rewards=target_rewards,
        factual_q=torch.zeros_like(factual_q),
        counterfactual_q=torch.zeros_like(counterfactual_q),
        candidate_mask=candidate_mask,
        direct_damage=torch.tensor(
            [[[16.0, 0.0], [4.0, 0.0], [0.0, 100.0], [0.0, 0.0]]],
            dtype=torch.float64,
            device=device,
        ),
    )
    checks = {
        "target_reward_identity": math.isclose(
            sum(target_rewards_dict.values()),
            3.0 / 7.0,
            abs_tol=1e-12,
            rel_tol=0.0,
        ),
        "direct_credit_conservation": direct.conservation_error < 1e-12,
        "direct_inactive_zero": direct.agent_rewards[4] == 0.0,
        "counterfactual_conservation": float(credit.conservation_error.detach().max()) < 1e-12,
        "counterfactual_inactive_zero": float(credit.agent_rewards[0, 3].detach()) == 0.0,
        "counterfactual_gradient_finite": bool(torch.isfinite(gradients).all().item()),
        "counterfactual_gradient_nonzero": bool((gradients.abs() > 0.0).any().item()),
        "direct_fallback_conservation": float(fallback.conservation_error.detach().max()) < 1e-12,
        "all_outputs_finite": bool(
            torch.isfinite(
                torch.cat(
                    (
                        credit.agent_rewards.flatten(),
                        credit.coefficients.flatten(),
                        fallback.agent_rewards.flatten(),
                    )
                )
            ).all().item()
        ),
    }
    result = {
        "run_id": "ICM000",
        "status": "passed" if all(checks.values()) else "failed",
        "device": str(device),
        "checks": checks,
        "target_rewards": {str(key): value for key, value in target_rewards_dict.items()},
        "direct_agent_rewards": {
            str(key): value for key, value in direct.agent_rewards.items()
        },
        "counterfactual_agent_rewards": credit.agent_rewards.detach().cpu().tolist(),
        "counterfactual_coefficients": credit.coefficients.detach().cpu().tolist(),
        "fallback_agent_rewards": fallback.agent_rewards.detach().cpu().tolist(),
        "gradient_norm": float(gradients.norm().item()),
    }
    output_path = args.output_dir / "ICM000_result.json"
    write_json(output_path, result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    if result["status"] != "passed":
        raise RuntimeError(f"ICM000 failed: {checks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
