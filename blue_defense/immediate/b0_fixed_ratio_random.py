# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Mapping

import random

from ..shared.controller import BlueDefenseController
from ..base import BlueObservation, BluePolicy, InterceptorAssignment
from ..shared.config import DEFAULT_MAX_SHOTS_PER_TARGET, BlueDefenseConfig


class B0FixedRatioRandomPolicy(BluePolicy):
    name = "b0_fixed_ratio_random"

    def __init__(self, max_shots_per_target: int = DEFAULT_MAX_SHOTS_PER_TARGET, seed: int | None = None):
        super().__init__(max_shots_per_target=max_shots_per_target)
        # 用独立随机流选择自爆虫，跨局状态由控制器管理。
        self._rng = random.Random(seed)

    def decide(self, observation: BlueObservation) -> list[InterceptorAssignment]:
        # 从当前观测整理可用自爆虫，按目标出现顺序进行分配。
        assignments: list[InterceptorAssignment] = []
        available = self.available_interceptors(observation)

        for target in observation.targets:
            if not available:
                break
            # 每个目标按剩余配额补足自爆虫数，选中的自爆虫立即移出可用池。
            need = min(self.remaining_need(observation, target.target_id), len(available))
            for _ in range(need):
                # 从剩余自爆虫中随机抽取一个，生成携带目标位置与速度的指派。
                index = self._rng.randrange(len(available))
                interceptor = available.pop(index)
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
            policy_name="b0_fixed_ratio_random",
            seed=seed,
            asset_values=dict(asset_values) if asset_values is not None else {},
        ))
