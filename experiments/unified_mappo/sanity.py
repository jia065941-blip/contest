"""UHM000：混合动作分布、掩码、更新与 checkpoint 的数值闭环。"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from .model import (
    HybridActionMask,
    HybridMAPPOConfig,
    HybridMAPPOTrainer,
    HybridRolloutBatch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单网络混合动作 MAPPO 数值闭环")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--observation-dim", type=int, default=128)
    parser.add_argument("--target-slots", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--initial-coordinate-log-std", type=float, default=-2.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def parameter_vector(trainer: HybridMAPPOTrainer) -> torch.Tensor:
    return torch.cat([parameter.detach().flatten().cpu() for parameter in trainer.model.parameters()])


def build_rollout(
    trainer: HybridMAPPOTrainer,
    *,
    batch_size: int,
    observation_dim: int,
    target_slots: int,
    seed: int,
) -> tuple[HybridRolloutBatch, dict[str, float]]:
    generator = torch.Generator(device=trainer.device).manual_seed(seed)
    observations = torch.randn(
        batch_size,
        observation_dim,
        generator=generator,
        device=trainer.device,
    )
    next_observations = observations + 0.01 * torch.randn(
        batch_size,
        observation_dim,
        generator=generator,
        device=trainer.device,
    )
    target_coordinates = torch.empty(
        batch_size,
        target_slots,
        2,
        device=trainer.device,
    ).uniform_(-1.0, 1.0, generator=generator)
    target_valid_mask = torch.rand(
        batch_size,
        target_slots,
        generator=generator,
        device=trainer.device,
    ) > 0.25
    target_valid_mask[:, 0] = True

    indices = torch.arange(batch_size, device=trainer.device)
    deployment = indices % 4 == 0
    retarget = indices % 4 == 1
    maneuver = ~deployment
    masks = HybridActionMask(
        presence=deployment,
        initial_position=deployment,
        search_position=deployment | retarget,
        retarget=retarget,
        target=deployment | retarget,
        maneuver=maneuver,
        satellite=torch.ones_like(deployment),
    )
    with torch.no_grad():
        sampled = trainer.model.act(
            observations,
            target_coordinates,
            target_valid_mask,
            masks,
        )
        recomputed, _, _ = trainer.model.evaluate_actions(
            observations,
            target_coordinates,
            target_valid_mask,
            sampled.action_mask,
            sampled.action,
        )
        inactive_no_target = trainer.model.act(
            observations[:1],
            target_coordinates[:1].cpu(),
            torch.zeros(1, target_slots, dtype=torch.bool),
            HybridActionMask(
                presence=torch.zeros(1, dtype=torch.bool),
                initial_position=torch.zeros(1, dtype=torch.bool),
                search_position=torch.zeros(1, dtype=torch.bool),
                retarget=torch.zeros(1, dtype=torch.bool),
                target=torch.zeros(1, dtype=torch.bool),
                maneuver=torch.ones(1, dtype=torch.bool),
                satellite=torch.ones(1, dtype=torch.bool),
            ),
        )
        original_presence_bias = trainer.model.presence_head.bias.detach().clone()
        trainer.model.presence_head.bias.fill_(-100.0)
        forced_absent = trainer.model.act(
            observations[:1],
            target_coordinates[:1],
            target_valid_mask[:1],
            HybridActionMask(
                presence=torch.ones(1, dtype=torch.bool, device=trainer.device),
                initial_position=torch.ones(1, dtype=torch.bool, device=trainer.device),
                search_position=torch.ones(1, dtype=torch.bool, device=trainer.device),
                retarget=torch.zeros(1, dtype=torch.bool, device=trainer.device),
                target=torch.ones(1, dtype=torch.bool, device=trainer.device),
                maneuver=torch.zeros(1, dtype=torch.bool, device=trainer.device),
                satellite=torch.ones(1, dtype=torch.bool, device=trainer.device),
            ),
            deterministic=True,
        )
        trainer.model.presence_head.bias.copy_(original_presence_bias)

    semantic = sampled.action.semantic_tensor()
    selected_valid = target_valid_mask[
        torch.arange(batch_size, device=trainer.device), sampled.action.target_index
    ]
    legal = (
        ((sampled.action.presence == 0) | (sampled.action.presence == 1))
        & (sampled.action.initial_xy.abs() <= 1.0).all(dim=-1)
        & ((~sampled.action_mask.target) | selected_valid)
        & ((sampled.action.maneuver >= -1) & (sampled.action.maneuver <= 1))
    )
    mask_consistent = (
        (sampled.action_mask.presence | (semantic[:, 0] == 1.0))
        & (sampled.action_mask.retarget | (semantic[:, 1] == 0.0))
        & (sampled.action_mask.initial_position | (semantic[:, 2:4] == 0.0).all(dim=-1))
        & (sampled.action_mask.search_position | (semantic[:, 4:6] == 0.0).all(dim=-1))
        & (sampled.action_mask.target | (semantic[:, 6:8] == 0.0).all(dim=-1))
        & (sampled.action_mask.maneuver | (semantic[:, 8] == 0.0))
    )
    finite = torch.isfinite(
        torch.cat(
            (
                semantic.flatten(),
                sampled.log_prob.flatten(),
                sampled.entropy.flatten(),
                sampled.value.flatten(),
            )
        )
    )
    # 解析奖励只用于数值闭环：参与、靠近区域中心和保持动作获得较高真值。
    rewards = (
        0.3 * sampled.action.presence.float()
        - 0.1 * sampled.action.initial_xy.square().sum(dim=-1)
        + 0.2 * (sampled.action.maneuver == 0).float()
        + 0.1 * (sampled.action.target_index == 0).float()
    )
    agent_ids = indices % 16
    trajectory_steps = indices // 16
    dones = torch.zeros(batch_size, device=trainer.device)
    for agent_id in range(16):
        member_indices = torch.nonzero(agent_ids == agent_id, as_tuple=False).flatten()
        dones[member_indices[-1]] = 1.0
    rollout = HybridRolloutBatch(
        observations=observations.detach().cpu(),
        target_coordinates=target_coordinates.detach().cpu(),
        target_valid_mask=target_valid_mask.detach().cpu(),
        action_mask=HybridActionMask(
            presence=sampled.action_mask.presence.cpu(),
            initial_position=sampled.action_mask.initial_position.cpu(),
            search_position=sampled.action_mask.search_position.cpu(),
            retarget=sampled.action_mask.retarget.cpu(),
            target=sampled.action_mask.target.cpu(),
            maneuver=sampled.action_mask.maneuver.cpu(),
            satellite=sampled.action_mask.satellite.cpu(),
        ),
        actions=sampled.action,
        old_log_probs=sampled.log_prob.detach().cpu(),
        old_values=sampled.value.detach().cpu(),
        rewards=rewards.detach().cpu(),
        dones=dones.detach().cpu(),
        next_observations=next_observations.detach().cpu(),
        agent_ids=agent_ids.detach().cpu(),
        steps=trajectory_steps.detach().cpu(),
    )
    diagnostics = {
        "finite_output_rate": float(finite.float().mean().item()),
        "legal_action_rate": float(legal.float().mean().item()),
        "mask_consistency_rate": float(mask_consistent.float().mean().item()),
        "inactive_no_target_supported": float(
            torch.isfinite(inactive_no_target.log_prob).all().item()
        ),
        "log_prob_recompute_max_error": float(
            (sampled.log_prob - recomputed).abs().max().item()
        ),
        "forced_absent_conditional_mask": float(
            forced_absent.action.presence.item() == 0
            and not forced_absent.action_mask.initial_position.item()
            and forced_absent.action_mask.target.item()
            and not forced_absent.action_mask.maneuver.item()
            and torch.count_nonzero(forced_absent.action.initial_xy).item() == 0
            and target_valid_mask[
                0, forced_absent.action.target_index.item()
            ].item()
        ),
        "semantic_action_dimension": float(semantic.shape[-1]),
        "presence_active_count": float(sampled.action_mask.presence.sum().item()),
        "initial_position_active_count": float(sampled.action_mask.initial_position.sum().item()),
        "target_active_count": float(sampled.action_mask.target.sum().item()),
        "maneuver_active_count": float(sampled.action_mask.maneuver.sum().item()),
    }
    return rollout, diagnostics


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = HybridMAPPOConfig(
        observation_dim=args.observation_dim,
        target_slots=args.target_slots,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        initial_coordinate_log_std=args.initial_coordinate_log_std,
        seed=args.seed,
        device=args.device,
    )
    trainer = HybridMAPPOTrainer(config)
    before = parameter_vector(trainer)
    rollout, diagnostics = build_rollout(
        trainer,
        batch_size=args.batch_size,
        observation_dim=args.observation_dim,
        target_slots=args.target_slots,
        seed=args.seed + 1,
    )
    metrics = trainer.update(rollout)
    after = parameter_vector(trainer)
    update_norm = float(torch.linalg.vector_norm(after - before).item())
    checkpoint = args.output_dir / "unified_hybrid_mappo_sanity.pt"
    trainer.save(checkpoint)
    restored = HybridMAPPOTrainer.load(checkpoint, device=str(trainer.device))
    restore_error = float(
        torch.max(torch.abs(parameter_vector(restored) - parameter_vector(trainer))).item()
    )

    criteria = {
        "finite_outputs": diagnostics["finite_output_rate"] == 1.0,
        "legal_actions": diagnostics["legal_action_rate"] == 1.0,
        "mask_consistency": diagnostics["mask_consistency_rate"] == 1.0,
        "inactive_no_target": diagnostics["inactive_no_target_supported"] == 1.0,
        "conditional_absence_mask": diagnostics["forced_absent_conditional_mask"] == 1.0,
        "seven_semantic_dimensions": diagnostics["semantic_action_dimension"] == 7.0,
        "log_prob_recompute": diagnostics["log_prob_recompute_max_error"] < 1e-5,
        "parameter_update": math.isfinite(update_norm) and update_norm > 0.0,
        "finite_losses": all(math.isfinite(value) for value in metrics.values()),
        "checkpoint_restore": restore_error < 1e-6,
    }
    result = {
        "run_id": "UHM000",
        "status": "passed" if all(criteria.values()) else "failed",
        "device": str(trainer.device),
        "config": asdict(config),
        "diagnostics": diagnostics,
        "training_metrics": metrics,
        "parameter_update_norm": update_norm,
        "checkpoint_restore_max_error": restore_error,
        "criteria": criteria,
        "checkpoint": str(checkpoint.resolve()),
    }
    result_path = args.output_dir / "sanity_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
