#!/usr/bin/env python3
"""Measure post-update policy shift on the exact actions in saved rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.unified_mappo.model import HybridMAPPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("rollouts", nargs="+", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--anchor-replicas", type=int, default=0)
    return parser.parse_args()


def rollout_log_prob(trainer: HybridMAPPOTrainer, rollout) -> torch.Tensor:
    device = trainer.device
    with torch.no_grad():
        log_prob, _, _ = trainer.model.evaluate_actions(
            rollout.observations.to(device),
            rollout.target_coordinates.to(device),
            rollout.target_valid_mask.to(device),
            rollout.action_mask.to(device),
            rollout.actions,
            target_features=(
                rollout.target_features.to(device)
                if getattr(rollout, "target_features", None) is not None
                else None
            ),
        )
    return log_prob.cpu()


def rollout_critic_value(trainer: HybridMAPPOTrainer, rollout) -> torch.Tensor:
    device = trainer.device
    with torch.no_grad():
        value = trainer.model.critic_value(
            rollout.critic_states.to(device),
            rollout.agent_ids.to(device),
            rollout.observations.to(device),
        )
    return value.cpu()


def rollout_head_kl(
    before: HybridMAPPOTrainer,
    after: HybridMAPPOTrainer,
    rollout,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact old-to-new KL for the active C0 goal and position heads."""

    device = before.device
    observations = rollout.observations.to(device)
    valid = rollout.target_valid_mask.to(device, dtype=torch.bool)
    masks = rollout.action_mask.to(device)
    target_features = (
        rollout.target_features.to(device)
        if getattr(rollout, "target_features", None) is not None
        else None
    )
    with torch.no_grad():
        old_parameters = before.model.distribution_parameters(
            observations, target_features, valid
        )
        new_parameters = after.model.distribution_parameters(
            observations, target_features, valid
        )
        old_target_logits, _ = before.model._masked_logits(
            old_parameters["target_logits"], valid, masks.target
        )
        new_target_logits, _ = after.model._masked_logits(
            new_parameters["target_logits"], valid, masks.target
        )
        target_kl = torch.distributions.kl_divergence(
            torch.distributions.Categorical(logits=old_target_logits),
            torch.distributions.Categorical(logits=new_target_logits),
        ) * masks.target
        old_std = before.model.initial_log_std.exp().expand_as(
            old_parameters["initial_mean"]
        )
        new_std = after.model.initial_log_std.exp().expand_as(
            new_parameters["initial_mean"]
        )
        position_kl = torch.distributions.kl_divergence(
            torch.distributions.Normal(old_parameters["initial_mean"], old_std),
            torch.distributions.Normal(new_parameters["initial_mean"], new_std),
        ).sum(dim=-1) * masks.initial_position
    return target_kl.cpu(), position_kl.cpu()


@torch.no_grad()
def boundary_snapshot_shift(
    before: HybridMAPPOTrainer,
    after: HybridMAPPOTrainer,
    payloads: list[dict[str, torch.Tensor]],
) -> dict[str, float | int]:
    target_kls = []
    agreements = []
    multi_masks = []
    for payload in payloads:
        active = torch.ones(
            payload["observations"].shape[0], dtype=torch.bool
        )
        if "factor_mask" in payload:
            factor_mask = payload["factor_mask"]
            if factor_mask.ndim != 2 or factor_mask.shape[1] <= 4:
                raise ValueError("factor_mask 必须包含 target 动作头列")
            active = factor_mask[:, 4].to(dtype=torch.bool)
        if not bool(active.any()):
            continue
        observations = payload["observations"][active].to(before.device)
        features = payload["target_features"][active].to(before.device)
        valid = payload["target_valid_mask"][active].to(
            before.device, dtype=torch.bool
        )
        old_logits = before.model.distribution_parameters(
            observations, features, valid
        )["target_logits"].masked_fill(~valid, -1e9)
        new_logits = after.model.distribution_parameters(
            observations, features, valid
        )["target_logits"].masked_fill(~valid, -1e9)
        target_kls.append(torch.distributions.kl_divergence(
            torch.distributions.Categorical(logits=old_logits),
            torch.distributions.Categorical(logits=new_logits),
        ).cpu())
        agreements.append(old_logits.argmax(-1).eq(new_logits.argmax(-1)).cpu())
        multi_masks.append(valid.sum(-1).gt(1).cpu())
    if not target_kls:
        raise ValueError("输入数据没有目标动作头生效的边界")
    kl = torch.cat(target_kls)
    agreement = torch.cat(agreements)
    multi = torch.cat(multi_masks)
    return {
        "samples": int(kl.numel()),
        "multi_legal_samples": int(multi.sum().item()),
        "target_mean_kl": float(kl.mean().item()),
        "target_multi_legal_mean_kl": float(kl[multi].mean().item()),
        "target_p99_kl": float(torch.quantile(kl, 0.99).item()),
        "target_max_kl": float(kl.max().item()),
        "target_top1_agreement": float(agreement.float().mean().item()),
        "target_top1_changes": int((~agreement).sum().item()),
        "multi_target_top1_agreement": float(
            agreement[multi].float().mean().item()
        ),
    }


def main() -> None:
    args = parse_args()
    before = HybridMAPPOTrainer.load(args.before, device=args.device)
    after = HybridMAPPOTrainer.load(args.after, device=args.device)
    payloads = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.rollouts
    ]
    if all(
        isinstance(payload, dict)
        and "observations" in payload
        and "target_features" in payload
        for payload in payloads
    ):
        print(json.dumps(
            boundary_snapshot_shift(before, after, payloads),
            ensure_ascii=False,
            indent=2,
        ))
        return
    before_log_probs = []
    after_log_probs = []
    stored_log_probs = []
    before_values = []
    after_values = []
    episode_returns = []
    target_kls = []
    position_kls = []
    for payload in payloads:
        rollout = payload["rollout"] if isinstance(payload, dict) else payload
        before_log_probs.append(rollout_log_prob(before, rollout))
        after_log_probs.append(rollout_log_prob(after, rollout))
        stored_log_probs.append(rollout.old_log_probs.cpu())
        before_values.append(rollout_critic_value(before, rollout))
        after_values.append(rollout_critic_value(after, rollout))
        target_kl, position_kl = rollout_head_kl(before, after, rollout)
        target_kls.append(target_kl)
        position_kls.append(position_kl)
        episode_returns.append(float(rollout.rewards.sum().item()))
    old = torch.cat(before_log_probs)
    new = torch.cat(after_log_probs)
    stored = torch.cat(stored_log_probs)
    log_ratio = new - old
    ratio = log_ratio.exp()
    result = {
        "samples": int(old.numel()),
        "stored_old_log_prob_max_abs_error": float((stored - old).abs().max().item()),
        "sampled_forward_kl": float((old - new).mean().item()),
        "nonnegative_approx_kl": float(((ratio - 1.0) - log_ratio).mean().item()),
        "log_ratio_mean": float(log_ratio.mean().item()),
        "log_ratio_abs_mean": float(log_ratio.abs().mean().item()),
        "ratio_mean": float(ratio.mean().item()),
        "ratio_min": float(ratio.min().item()),
        "ratio_max": float(ratio.max().item()),
        "clip_fraction_at_0_2": float(((ratio - 1.0).abs() > 0.2).float().mean().item()),
        "target_head_exact_kl": float(torch.cat(target_kls).mean().item()),
        "position_head_exact_kl": float(torch.cat(position_kls).mean().item()),
    }
    if args.anchor_replicas:
        if len(args.rollouts) % args.anchor_replicas:
            raise ValueError("rollout 数量必须能被 anchor-replicas 整除")
        target_parts = []
        advantage_parts = []
        for start in range(0, len(args.rollouts), args.anchor_replicas):
            stop = start + args.anchor_replicas
            group_sum = sum(episode_returns[start:stop])
            group_mean = group_sum / args.anchor_replicas
            target_parts.extend(
                torch.full_like(before_values[index], group_mean)
                for index in range(start, stop)
            )
            advantage_parts.extend(
                torch.full_like(
                    before_values[index],
                    episode_returns[index]
                    - (group_sum - episode_returns[index])
                    / (args.anchor_replicas - 1),
                )
                for index in range(start, stop)
            )
        targets = torch.cat(target_parts)
        advantages = torch.cat(advantage_parts)
        raw_advantages = advantages.clone()
        advantage_scale = advantages.std(unbiased=False)
        if float(advantage_scale.item()) > 1e-8:
            advantages = advantages / (advantage_scale + 1e-8)
        pre_values = torch.cat(before_values)
        post_values = torch.cat(after_values)
        result.update({
            "critic_target_mean": float(targets.mean().item()),
            "critic_pre_mean": float(pre_values.mean().item()),
            "critic_pre_mae": float((pre_values - targets).abs().mean().item()),
            "critic_post_mean": float(post_values.mean().item()),
            "critic_post_mae": float((post_values - targets).abs().mean().item()),
            "post_update_surrogate_gain": float(
                ((ratio - 1.0) * advantages).mean().item()
            ),
            "post_update_raw_surrogate_gain": float(
                ((ratio - 1.0) * raw_advantages).mean().item()
            ),
        })
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
