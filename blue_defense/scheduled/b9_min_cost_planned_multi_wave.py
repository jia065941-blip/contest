# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Mapping

from ..shared.controller import BlueDefenseController
from ..shared.config import DEFAULT_MAX_SHOTS_PER_TARGET, BlueDefenseConfig

from ..base import (
    BlueObservation,
    InterceptorAssignment,
    InterceptorState,
    PlannedInterceptorAssignment,
    ScheduledBluePolicy,
    ThreatTrack,
    distance_m,
    speed_mps,
)


class B9MinCostPlannedMultiWavePolicy(ScheduledBluePolicy):
    """按最小代价分配自爆虫，并按规划延迟和波次释放。"""

    name = "b9_min_cost_planned_multi_wave"

    def __init__(
        self,
        max_shots_per_target: int = DEFAULT_MAX_SHOTS_PER_TARGET,
        planning_delay: float | None = None,
        wave_gap: float | None = None,
        wave_size: int = 1,
    ):
        super().__init__(max_shots_per_target=max_shots_per_target)
        # 显式时间参数使用步数，默认按观测时钟保留一秒延迟与五秒波次间隔。
        self.planning_delay = planning_delay
        self.wave_gap = wave_gap
        self.wave_size = max(1, int(wave_size))

    def decide(
        self,
        observation: BlueObservation,
    ) -> list[InterceptorAssignment]:
        # 整理可规划自爆虫及扣除既有计划后的目标需求。
        available_by_id = {
            item.interceptor_id: item
            for item in self.available_interceptors_for_planning(observation)
        }
        planning_delay = (
            self.planning_delay
            if self.planning_delay is not None
            else observation.clock.seconds_to_steps(1.0)
        )
        wave_gap = (
            self.wave_gap
            if self.wave_gap is not None
            else observation.clock.seconds_to_steps(5.0)
        )
        remaining_need = {
            target.target_id: self.remaining_need_with_pending(
                observation,
                target.target_id,
            )
            for target in observation.targets
        }
        assigned_count = {
            target.target_id: 0
            for target in observation.targets
        }
        plans: list[PlannedInterceptorAssignment] = []

        # 逐条观测缓存固定代价，支持同一目标编号出现多条记录。
        # 缓存沿用观测顺序与原有浮点运算结合顺序。
        costs_by_target = []
        for target in observation.targets:
            if not available_by_id or remaining_need[target.target_id] <= 0:
                continue
            asset_cost = self._nearest_asset_distance(observation, target) * 0.05
            urgency_cost = self._target_urgency(observation, target) * 100_000.0
            pair_costs = {
                interceptor.interceptor_id: (
                    distance_m(interceptor.pos, target.pos) / 1200.0
                ) * 1000.0 + asset_cost
                for interceptor in available_by_id.values()
            }
            costs_by_target.append((target, pair_costs, urgency_cost))

        # 逐轮选择总代价最低的目标与自爆虫配对，重复分配逐步增加惩罚。
        while available_by_id and any(
            need > 0 for need in remaining_need.values()
        ):
            best_pair: tuple[
                float,
                ThreatTrack,
                InterceptorState,
            ] | None = None
            for target, pair_costs, urgency_cost in costs_by_target:
                if remaining_need.get(target.target_id, 0) <= 0:
                    continue
                overkill_penalty = assigned_count[target.target_id] * 50_000.0
                for interceptor in available_by_id.values():
                    cost = (
                        pair_costs[interceptor.interceptor_id]
                        + overkill_penalty
                        - urgency_cost
                    )
                    if best_pair is None or cost < best_pair[0]:
                        best_pair = (cost, target, interceptor)

            if best_pair is None:
                break

            # 扣减资源与需求，并按目标本轮分配顺序计算波次。
            cost, target, interceptor = best_pair
            available_by_id.pop(interceptor.interceptor_id, None)
            remaining_need[target.target_id] -= 1
            wave_index = (
                assigned_count[target.target_id] // self.wave_size
            )
            assigned_count[target.target_id] += 1
            # 释放步数由当前步号、规划延迟和波次间隔共同确定。
            plans.append(
                PlannedInterceptorAssignment(
                    assignment=InterceptorAssignment(
                        interceptor_id=interceptor.interceptor_id,
                        target_id=target.target_id,
                    ),
                    due_step=(
                        observation.step
                        + planning_delay
                        + wave_index * wave_gap
                    ),
                    wave_index=wave_index,
                )
            )

        # 合并新增计划，并输出本刻到期的追踪指派。
        self.schedule_assignments(observation, plans)
        return self.release_due_assignments(observation)

    @staticmethod
    def _nearest_asset_distance(
        observation: BlueObservation,
        target: ThreatTrack,
    ) -> float:
        if not observation.assets:
            return 0.0
        # 最近保护对象距离构成空间代价项。
        return min(
            distance_m(asset.pos, target.pos)
            for asset in observation.assets
        )

    @staticmethod
    def _target_urgency(
        observation: BlueObservation,
        target: ThreatTrack,
    ) -> float:
        if not observation.assets:
            return 1.0
        # 以最近保护对象的价值和预计到达时间计算紧迫度。
        nearest_asset = min(
            observation.assets,
            key=lambda asset: distance_m(
                asset.pos,
                target.pos,
            ),
        )
        eta = distance_m(
            nearest_asset.pos,
            target.pos,
        ) / speed_mps(target.vel, default=300.0)
        return nearest_asset.value / max(eta, 1.0)


class BlueDefense(BlueDefenseController):
    """当前蓝方算法的统一运行入口。"""

    def __init__(
        self,
        *,
        seed: int | None = None,
        asset_values: Mapping[int, float] | None = None,
    ):
        # 绑定本文件的策略名称，以共享种子和保护对象价值构造控制器。
        super().__init__(BlueDefenseConfig(
            policy_name="b9_min_cost_planned_multi_wave",
            seed=seed,
            asset_values=dict(asset_values) if asset_values is not None else {},
        ))
