# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Mapping

from ..shared.config import BlueDefenseConfig
from ..shared.controller import BlueDefenseController
from ..base import BlueObservation, BluePolicy, InterceptorAssignment, distance_m


class B1NearestInterceptorPolicy(BluePolicy):
    name = "b1_nearest_interceptor"

    def decide(self, observation: BlueObservation) -> list[InterceptorAssignment]:
        # 整理可用自爆虫，按探测时间和目标编号确定处理顺序。
        assignments: list[InterceptorAssignment] = []
        available = self.available_interceptors(observation)
        targets = sorted(observation.targets, key=lambda item: (item.detect_step, item.target_id))

        for target in targets:
            if not available:
                break
            need = min(self.remaining_need(observation, target.target_id), len(available))
            # 为当前目标逐个选择距离最近的自爆虫，并同步扣减可用池。
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
            policy_name="b1_nearest_interceptor",
            seed=seed,
            asset_values=dict(asset_values) if asset_values is not None else {},
        ))
