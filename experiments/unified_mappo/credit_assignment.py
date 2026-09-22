"""反事实边际贡献的归一化信用分配。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CounterfactualCredit:
    agent_rewards: torch.Tensor
    coefficients: torch.Tensor
    marginal_contributions: torch.Tensor
    conservation_error: torch.Tensor


def normalized_counterfactual_credit(
    *,
    target_rewards: torch.Tensor,
    factual_q: torch.Tensor,
    counterfactual_q: torch.Tensor,
    candidate_mask: torch.Tensor,
    direct_damage: torch.Tensor,
    epsilon: float = 1e-12,
) -> CounterfactualCredit:
    """将每个目标奖励按非负反事实边际贡献归一化分配。"""

    if target_rewards.ndim != 2:
        raise ValueError("target_rewards 必须为 [batch, target]")
    batch_size, target_count = target_rewards.shape
    expected_shape = (
        batch_size,
        int(counterfactual_q.shape[1]),
        target_count,
    )
    if factual_q.shape != (batch_size, target_count):
        raise ValueError("factual_q 必须为 [batch, target]")
    if counterfactual_q.shape != expected_shape:
        raise ValueError("counterfactual_q 必须为 [batch, agent, target]")
    if candidate_mask.shape != expected_shape or direct_damage.shape != expected_shape:
        raise ValueError("candidate_mask 和 direct_damage 必须匹配 counterfactual_q")
    if torch.any(target_rewards < -epsilon) or torch.any(direct_damage < 0.0):
        raise ValueError("目标奖励和直接毁伤必须非负")

    eligible = candidate_mask.to(dtype=torch.bool) | (direct_damage > 0.0)
    marginal = torch.relu(factual_q.unsqueeze(1) - counterfactual_q)
    marginal = marginal * eligible.to(dtype=marginal.dtype)
    marginal_total = marginal.sum(dim=1, keepdim=True)

    direct = direct_damage * eligible.to(dtype=direct_damage.dtype)
    direct_total = direct.sum(dim=1, keepdim=True)
    needs_fallback = marginal_total <= epsilon
    missing = (
        (target_rewards > epsilon).unsqueeze(1)
        & needs_fallback
        & (direct_total <= epsilon)
    )
    if torch.any(missing):
        raise RuntimeError("非零目标奖励缺少反事实贡献和直接毁伤来源")

    counterfactual_coefficients = marginal / marginal_total.clamp_min(epsilon)
    direct_coefficients = direct / direct_total.clamp_min(epsilon)
    coefficients = torch.where(
        needs_fallback,
        direct_coefficients,
        counterfactual_coefficients,
    )
    zero_reward = (target_rewards <= epsilon).unsqueeze(1)
    coefficients = torch.where(zero_reward, torch.zeros_like(coefficients), coefficients)
    agent_rewards = torch.einsum("bnj,bj->bn", coefficients, target_rewards)
    error = (
        agent_rewards.sum(dim=1) - target_rewards.sum(dim=1)
    ).abs()
    return CounterfactualCredit(
        agent_rewards=agent_rewards,
        coefficients=coefficients,
        marginal_contributions=marginal,
        conservation_error=error,
    )
