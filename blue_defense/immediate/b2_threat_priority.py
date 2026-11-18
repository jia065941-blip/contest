# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Mapping

from ..shared.config import BlueDefenseConfig
from ..shared.controller import BlueDefenseController
from ..base import (
    BlueObservation,
    BluePolicy,
    InterceptorAssignment,
    ThreatTrack,
    distance_m,
    speed_mps,
)


class B2ThreatPriorityPolicy(BluePolicy):
    name = "b2_threat_priority"

    def decide(self, observation: BlueObservation) -> list[InterceptorAssignment]:
        # 综合目标类型与保护对象紧迫度，将威胁从高到低排序。
        assignments: list[InterceptorAssignment] = []
        available = self.available_interceptors(observation)
        targets = sorted(
            observation.targets,
            key=lambda target: self._threat_score(observation, target),
            reverse=True,
        )

        for target in targets:
            if not available:
                break
            # 结合目标类型需求、已释放配额和可用自爆虫确定本次释放数量。
            need = min(self._desired_shots(target), self.remaining_need(observation, target.target_id), len(available))
            # 逐个选择最近自爆虫，按目标威胁顺序输出即时指派。
            for _ in range(need):
                interceptor = min(
                    available,
                    key=lambda item: distance_m(item.pos, target.pos),
                )
                available.remove(interceptor)
                assignments.append(
                    InterceptorAssignment(
                        interceptor_id=interceptor.interceptor_id,
                        target_id=target.target_id,
                    )
                )

        return assignments

    def _desired_shots(self, target: ThreatTrack) -> int:
        # 按目标类型设置期望自爆虫数，同时遵循统一的单目标自爆虫数上限。
        if target.type == 21000:
            return min(self.max_shots_per_target, 3)
        if target.type == 21001:
            return min(self.max_shots_per_target, 2)
        return 1

    def _threat_score(self, observation: BlueObservation, target: ThreatTrack) -> float:
        if not observation.assets:
            return self._type_weight(target)

        # 用最近保护对象的价值与预计到达时间衡量附加威胁。
        nearest_asset = min(
            observation.assets,
            key=lambda asset: distance_m(asset.pos, target.pos),
        )
        distance_to_asset = distance_m(nearest_asset.pos, target.pos)
        time_to_asset = distance_to_asset / speed_mps(target.vel, default=300.0)
        asset_term = nearest_asset.value / max(time_to_asset, 1.0)
        return self._type_weight(target) + asset_term * 1000.0

    @staticmethod
    def _type_weight(target: ThreatTrack) -> float:
        # 为三种红方目标提供基础威胁权重。
        if target.type == 21000:
            return 30.0
        if target.type == 21001:
            return 20.0
        if target.type == 21002:
            return 10.0
        return 1.0


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
            policy_name="b2_threat_priority",
            seed=seed,
            asset_values=dict(asset_values) if asset_values is not None else {},
        ))
