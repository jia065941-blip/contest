# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Mapping

from ..shared.controller import BlueDefenseController
from ..shared.config import DEFAULT_MAX_SHOTS_PER_TARGET, BlueDefenseConfig

from ..base import (
    BlueObservation,
    InterceptorAssignment,
    PlannedInterceptorAssignment,
    ScheduledBluePolicy,
    distance_m,
)


class B7PlannedNearestPolicy(ScheduledBluePolicy):
    """选择最近自爆虫，并在规划延迟后释放指派。"""

    name = "b7_planned_nearest"

    def __init__(
        self,
        max_shots_per_target: int = DEFAULT_MAX_SHOTS_PER_TARGET,
        planning_delay: float | None = None,
    ):
        super().__init__(max_shots_per_target=max_shots_per_target)
        # 显式间隔使用步数；未指定时按观测时钟保留原有一秒延迟。
        self.planning_delay = planning_delay

    def decide(
        self,
        observation: BlueObservation,
    ) -> list[InterceptorAssignment]:
        # 调度基类整理可规划自爆虫，目标按探测时间与编号依次处理。
        available = self.available_interceptors_for_planning(observation)
        planning_delay = (
            self.planning_delay
            if self.planning_delay is not None
            else observation.clock.seconds_to_steps(1.0)
        )
        plans: list[PlannedInterceptorAssignment] = []
        targets = sorted(
            observation.targets,
            key=lambda item: (item.detect_step, item.target_id),
        )

        for target in targets:
            if not available:
                break
            need = min(
                self.remaining_need_with_pending(
                    observation,
                    target.target_id,
                ),
                len(available),
            )
            # 为目标选择最近自爆虫，把指派放入统一延迟到期的计划。
            for _ in range(need):
                interceptor = min(
                    available,
                    key=lambda item: distance_m(
                        item.pos,
                        target.pos,
                    ),
                )
                available.remove(interceptor)
                plans.append(
                    PlannedInterceptorAssignment(
                        assignment=InterceptorAssignment(
                            interceptor_id=interceptor.interceptor_id,
                            target_id=target.target_id,
                        ),
                        due_step=(
                            observation.step + planning_delay
                        ),
                    )
                )

        # 合并新计划并释放本刻到期的指派，目标运动信息由调度基类更新。
        self.schedule_assignments(observation, plans)
        return self.release_due_assignments(observation)


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
            policy_name="b7_planned_nearest",
            seed=seed,
            asset_values=dict(asset_values) if asset_values is not None else {},
        ))
