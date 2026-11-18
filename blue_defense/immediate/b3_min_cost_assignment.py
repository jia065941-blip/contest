# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Mapping

from ..shared.config import BlueDefenseConfig
from ..shared.controller import BlueDefenseController
from ..base import (
    BlueObservation,
    BluePolicy,
    InterceptorAssignment,
    InterceptorState,
    ThreatTrack,
    distance_m,
    speed_mps,
)


class B3MinCostAssignmentPolicy(BluePolicy):
    name = "b3_min_cost_assignment"

    def decide(self, observation: BlueObservation) -> list[InterceptorAssignment]:
        # 按编号保存可用自爆虫，并统计各目标剩余需求和本轮分配数量。
        assignments: list[InterceptorAssignment] = []
        available_by_id = {
            item.interceptor_id: item
            for item in self.available_interceptors(observation)
        }
        remaining_need = {
            target.target_id: self.remaining_need(observation, target.target_id)
            for target in observation.targets
        }
        assigned_count = {target.target_id: 0 for target in observation.targets}

        # 每轮遍历目标与自爆虫配对，贪心选择当前总代价最低的一对。
        while available_by_id and any(need > 0 for need in remaining_need.values()):
            best_pair: tuple[float, ThreatTrack, InterceptorState] | None = None
            for target in observation.targets:
                if remaining_need.get(target.target_id, 0) <= 0:
                    continue
                for interceptor in available_by_id.values():
                    cost = self._assignment_cost(observation, target, interceptor, assigned_count[target.target_id])
                    if best_pair is None or cost < best_pair[0]:
                        best_pair = (cost, target, interceptor)

            if best_pair is None:
                break

            # 扣减资源和需求，将选择结果转换为即时追踪指派。
            cost, target, interceptor = best_pair
            available_by_id.pop(interceptor.interceptor_id, None)
            remaining_need[target.target_id] -= 1
            assigned_count[target.target_id] += 1
            assignments.append(
                InterceptorAssignment(
                    interceptor_id=interceptor.interceptor_id,
                    target_id=target.target_id,
                )
            )

        return assignments

    def _assignment_cost(
        self,
        observation: BlueObservation,
        target: ThreatTrack,
        interceptor: InterceptorState,
        already_assigned: int,
    ) -> float:
        # 组合接近目标耗时、保护对象距离、重复分配惩罚和紧迫度收益。
        intercept_distance = distance_m(interceptor.pos, target.pos)
        intercept_time = intercept_distance / 1200.0
        asset_distance = self._nearest_asset_distance(observation, target)
        target_urgency = self._target_urgency(observation, target)
        overkill_penalty = already_assigned * 50000.0
        return intercept_time * 1000.0 + asset_distance * 0.05 + overkill_penalty - target_urgency * 100000.0

    @staticmethod
    def _nearest_asset_distance(observation: BlueObservation, target: ThreatTrack) -> float:
        if not observation.assets:
            return 0.0
        # 保护对象距离作为配对代价中的空间项。
        return min(distance_m(asset.pos, target.pos) for asset in observation.assets)

    @staticmethod
    def _target_urgency(observation: BlueObservation, target: ThreatTrack) -> float:
        if not observation.assets:
            return 1.0
        # 由最近保护对象价值与目标预计到达时间计算紧迫度。
        nearest_asset = min(observation.assets, key=lambda asset: distance_m(asset.pos, target.pos))
        eta = distance_m(nearest_asset.pos, target.pos) / speed_mps(target.vel, default=300.0)
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
            policy_name="b3_min_cost_assignment",
            seed=seed,
            asset_values=dict(asset_values) if asset_values is not None else {},
        ))
