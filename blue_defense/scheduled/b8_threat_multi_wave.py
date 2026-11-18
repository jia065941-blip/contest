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
    ThreatTrack,
    distance_m,
    speed_mps,
)


class B8ThreatMultiWavePolicy(ScheduledBluePolicy):
    """按目标威胁分配自爆虫，并分波次释放。"""

    name = "b8_threat_multi_wave"

    def __init__(
        self,
        max_shots_per_target: int = DEFAULT_MAX_SHOTS_PER_TARGET,
        wave_gap: float | None = None,
        wave_size: int = 1,
    ):
        super().__init__(max_shots_per_target=max_shots_per_target)
        # 显式波次间隔使用步数；未指定时按观测时钟保留原有十秒间隔。
        self.wave_gap = wave_gap
        self.wave_size = max(1, int(wave_size))

    def decide(
        self,
        observation: BlueObservation,
    ) -> list[InterceptorAssignment]:
        # 整理可规划自爆虫，并按威胁评分从高到低安排目标。
        available = self.available_interceptors_for_planning(observation)
        wave_gap = (
            self.wave_gap
            if self.wave_gap is not None
            else observation.clock.seconds_to_steps(10.0)
        )
        plans: list[PlannedInterceptorAssignment] = []
        targets = sorted(
            observation.targets,
            key=lambda target: self._threat_score(observation, target),
            reverse=True,
        )

        for target in targets:
            if not available:
                break
            # 目标类型、现有计划及自爆虫余量共同决定本轮新增计划数量。
            need = min(
                self._desired_shots(target),
                self.remaining_need_with_pending(
                    observation,
                    target.target_id,
                ),
                len(available),
            )
            # 按最近距离选择自爆虫，以自爆虫序号计算所属波次和释放时刻。
            for index in range(need):
                interceptor = min(
                    available,
                    key=lambda item: distance_m(
                        item.pos,
                        target.pos,
                    ),
                )
                available.remove(interceptor)
                wave_index = index // self.wave_size
                plans.append(
                    PlannedInterceptorAssignment(
                        assignment=InterceptorAssignment(
                            interceptor_id=interceptor.interceptor_id,
                            target_id=target.target_id,
                        ),
                        due_step=(
                            observation.step
                            + wave_index * wave_gap
                        ),
                        wave_index=wave_index,
                    )
                )

        # 写入计划队列，首波与历史计划中到期的指派在本刻释放。
        self.schedule_assignments(observation, plans)
        return self.release_due_assignments(observation)

    def _desired_shots(self, target: ThreatTrack) -> int:
        # 按目标类型设置期望自爆虫数，并应用统一的单目标自爆虫数上限。
        if target.type == 21000:
            return min(self.max_shots_per_target, 3)
        if target.type == 21001:
            return min(self.max_shots_per_target, 2)
        return 1

    def _threat_score(
        self,
        observation: BlueObservation,
        target: ThreatTrack,
    ) -> float:
        if not observation.assets:
            return self._type_weight(target)

        # 将目标类型权重与最近保护对象的价值、到达时间组合为威胁分数。
        nearest_asset = min(
            observation.assets,
            key=lambda asset: distance_m(
                asset.pos,
                target.pos,
            ),
        )
        distance_to_asset = distance_m(
            nearest_asset.pos,
            target.pos,
        )
        time_to_asset = distance_to_asset / speed_mps(
            target.vel,
            default=300.0,
        )
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
            policy_name="b8_threat_multi_wave",
            seed=seed,
            asset_values=dict(asset_values) if asset_values is not None else {},
        ))
