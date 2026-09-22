"""目标级计分增量与智能体级直接信用分配。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


_TOLERANCE = 1e-12
TARGET_SCORE_DENOMINATOR = 21.0


def _lookup_entity(entities: Mapping, entity_id: int) -> Mapping | None:
    return entities.get(entity_id, entities.get(str(entity_id)))


def _clipped_damage(health: float, initial_health: float) -> float:
    if initial_health <= 0.0:
        raise ValueError("目标初始生命值必须为正数")
    return min(1.0, max(0.0, (initial_health - health) / initial_health))


def target_reward_vector(
    *,
    current_entities: Mapping,
    previous_entities: Mapping,
    initial_health: Mapping[int, float],
    objective_weights: Mapping[int, float],
) -> dict[int, float]:
    """计算与正式分数完全一致的逐目标奖励向量。"""

    total_weight = float(sum(objective_weights.values()))
    if total_weight <= 0.0:
        raise ValueError("目标权重总和必须为正数")
    rewards: dict[int, float] = {}
    for raw_target_id, raw_weight in objective_weights.items():
        target_id = int(raw_target_id)
        current = _lookup_entity(current_entities, target_id)
        previous = _lookup_entity(previous_entities, target_id)
        if current is None or previous is None:
            raise KeyError(f"目标 {target_id} 缺少相邻时刻观测")
        health_0 = float(initial_health[target_id])
        previous_damage = _clipped_damage(
            float(previous.get("health", health_0)),
            health_0,
        )
        current_damage = _clipped_damage(
            float(current.get("health", health_0)),
            health_0,
        )
        delta = current_damage - previous_damage
        if delta < -_TOLERANCE:
            raise ValueError(f"目标 {target_id} 的累计毁伤比例发生下降: {delta}")
        rewards[target_id] = (float(raw_weight) / total_weight) * max(0.0, delta)
    return rewards


def target_reward_vector_fixed_21(
    *,
    current_entities: Mapping,
    previous_entities: Mapping,
    initial_health: Mapping[int, float],
    objective_weights: Mapping[int, float],
) -> dict[int, float]:
    """按 ``w_j / 21 * (d_t - d_{t-1})`` 计算目标级新增毁伤奖励。

    该显式 API 使用固定分母 21，不改变 :func:`target_reward_vector`
    的历史“权重和归一化”语义。
    """

    rewards: dict[int, float] = {}
    for raw_target_id, raw_weight in objective_weights.items():
        target_id = int(raw_target_id)
        current = _lookup_entity(current_entities, target_id)
        previous = _lookup_entity(previous_entities, target_id)
        if current is None or previous is None:
            raise KeyError(f"目标 {target_id} 缺少相邻时刻观测")
        health_0 = float(initial_health[target_id])
        previous_damage = _clipped_damage(
            float(previous.get("health", health_0)),
            health_0,
        )
        current_damage = _clipped_damage(
            float(current.get("health", health_0)),
            health_0,
        )
        delta = current_damage - previous_damage
        if delta < -_TOLERANCE:
            raise ValueError(f"目标 {target_id} 的累计毁伤比例发生下降: {delta}")
        rewards[target_id] = (
            float(raw_weight) / TARGET_SCORE_DENOMINATOR
        ) * max(0.0, delta)
    return rewards


@dataclass(frozen=True)
class DirectCreditAllocation:
    agent_rewards: dict[int, float]
    target_agent_rewards: dict[int, dict[int, float]]
    conservation_error: float
    nonzero_agent_ids: tuple[int, ...]


def allocate_direct_credit(
    *,
    target_rewards: Mapping[int, float],
    hit_events: Mapping[int, Sequence[Mapping]],
    entity_to_agent: Mapping[int, int],
    agent_ids: Sequence[int],
    tolerance: float = _TOLERANCE,
) -> DirectCreditAllocation:
    """按本步实际毁伤来源分配目标奖励，并保持奖励守恒。"""

    rewards = {int(agent_id): 0.0 for agent_id in agent_ids}
    target_agent_rewards: dict[int, dict[int, float]] = {}
    for raw_target_id, raw_reward in target_rewards.items():
        target_id = int(raw_target_id)
        target_reward = float(raw_reward)
        if target_reward < -tolerance:
            raise ValueError(f"目标 {target_id} 奖励为负数: {target_reward}")
        if target_reward <= tolerance:
            target_agent_rewards[target_id] = {}
            continue
        source_weights: dict[int, float] = {}
        for event in hit_events.get(target_id, ()):
            source_entity_id = int(event.get("id", -1))
            agent_id = entity_to_agent.get(source_entity_id)
            damage_point = max(0.0, float(event.get("damage_point", 0.0)))
            if agent_id is None or damage_point <= tolerance:
                continue
            source_weights[int(agent_id)] = (
                source_weights.get(int(agent_id), 0.0) + damage_point
            )
        source_total = sum(source_weights.values())
        if source_total <= tolerance:
            raise RuntimeError(
                f"目标 {target_id} 产生奖励 {target_reward}，但没有可归因的直接毁伤来源"
            )
        allocations = {
            agent_id: target_reward * source_weight / source_total
            for agent_id, source_weight in source_weights.items()
        }
        target_agent_rewards[target_id] = allocations
        for agent_id, value in allocations.items():
            rewards.setdefault(agent_id, 0.0)
            rewards[agent_id] += value

    expected = float(sum(target_rewards.values()))
    actual = float(sum(rewards.values()))
    error = abs(expected - actual)
    if error > tolerance:
        raise RuntimeError(
            f"智能体奖励不守恒: expected={expected}, actual={actual}, error={error}"
        )
    nonzero = tuple(sorted(agent_id for agent_id, value in rewards.items() if value > tolerance))
    return DirectCreditAllocation(
        agent_rewards=rewards,
        target_agent_rewards=target_agent_rewards,
        conservation_error=error,
        nonzero_agent_ids=nonzero,
    )
