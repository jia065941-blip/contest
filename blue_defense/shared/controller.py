# -*- coding: utf-8 -*-
"""内置蓝方算法的配置装配、决策节流与回合重置。"""

from __future__ import annotations

import math
from numbers import Real

from ..base import BaseBlueDefense, BlueClock, BlueDecision, BlueObservation, DefendedAsset
from .config import BlueDefenseConfig
from .registry import build_blue_policy
from .information import BlueInformationStore


class BlueDefenseController(BaseBlueDefense):
    """管理策略配置、策略状态和信息库快照的决策间隔。"""

    def __init__(self, config: BlueDefenseConfig | None = None):
        # 控制器持有一个算法实例，并单独记录上次完成评估的仿真时刻。
        self.config = config or BlueDefenseConfig()
        super().__init__(seed=self.config.seed, asset_values=self.config.asset_values)
        self._policy = build_blue_policy(self.config)
        self._information = BlueInformationStore()
        self._last_decision_step = float("-inf")

    def bind_clock(self, clock: BlueClock) -> None:
        """绑定场景步长，不重建策略或重置其随机流。"""
        super().bind_clock(clock)
        self._information.bind_clock(clock)
        if self.name in ("b8_threat_multi_wave", "b9_min_cost_planned_multi_wave"):
            default_seconds = 10.0 if self.name == "b8_threat_multi_wave" else 5.0
            self._policy.wave_gap = (
                self.config.wave_gap_step if self.config.wave_gap_step is not None
                else clock.seconds_to_steps(default_seconds)
            )

    # 将算法声明的名称与计划模式提供给环境的控制执行逻辑。
    @property
    def name(self) -> str:
        return self._policy.name

    @property
    def plans_launch_time(self) -> bool:
        return bool(self._policy.plans_launch_time)

    @property
    def preserves_pending_launches(self) -> bool:
        return bool(self._policy.preserves_pending_launches)

    def decide(self, observation: BlueObservation) -> BlueDecision | None:
        """在决策间隔允许时评估当前信息库快照。"""

        self.bind_clock(observation.clock)
        self.receive_tracks(observation.targets, observation.step)
        # 保护对象和价值属于内置 B 的算法准备；保留原有类型分组顺序。
        assets = []
        for entity_type in (9400, 9500, 9600):
            for entity in observation.own_entities:
                if entity.type != entity_type or entity.health <= 0 or not entity.is_visible:
                    continue
                value = self.asset_value(entity.entity_id, entity.type)
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                    raise ValueError("blue policy asset_value must be finite")
                assets.append(DefendedAsset(
                    asset_id=entity.entity_id, name=entity.name,
                    pos=entity.pos, health=entity.health,
                    value=float(value), entity_type=entity.type,
                ))
        self._information.update_own_state(
            observation.interceptors, assets, observation.launched_map,
            own_entities=observation.own_entities,
        )
        prepared = self._information.snapshot(observation.step)
        # 未到评估时刻返回 None，环境据此继续保留现有待释放计划。
        interval_step = self._decision_interval_step()
        if observation.step - self._last_decision_step < interval_step:
            return None

        # 完成评估后统一封装动作和求解统计，空动作同样是一轮有效评估。
        self._last_decision_step = observation.step
        assignments = tuple(self._policy.decide(prepared))
        stats = dict(getattr(self._policy, "last_solve_stats", {}))
        return BlueDecision(assignments=assignments, solve_stats=stats)

    def receive_tracks(self, tracks, sim_step: int) -> None:
        self._information.receive_tracks(tracks, sim_step)

    def get_tracks(self, sim_step: int):
        return self._information.get_tracks(sim_step)

    def get_track(self, target_id: int, sim_step: int):
        return self._information.get_track(target_id, sim_step)

    def reset(self) -> None:
        """重置节流时钟与算法状态，按配置重新开始 B0 的随机序列。"""

        super().reset()
        self._information.reset()
        self._last_decision_step = float("-inf")
        # 重新构造 B0 才会重新设种；通常只重置算法的本局状态。
        if self.config.reset_rng_on_episode and self.name == "b0_fixed_ratio_random":
            self._policy = build_blue_policy(self.config)
            return
        self._policy.reset()

    def asset_value(self, entity_id: int, entity_type: int) -> float:
        """按实体配置查询价值，其余采用保护对象类型的默认值。"""
        if entity_id in self.config.asset_values:
            return float(self.config.asset_values[entity_id])
        return {
            9400: 5.0,
            9600: 2.0,
            9500: 1.0,
        }.get(entity_type, 1.0)

    def _decision_interval_step(self) -> float:
        """按算法选择评估间隔，内部延迟调度算法逐次推进。"""
        # B3 单独节流，交给环境执行计划的算法采用联合评估间隔。
        if self.name == "b3_min_cost_assignment":
            return self.config.min_cost_decision_interval_step
        if self.name == "b6_mixed_fire_coordination":
            # 在第二次拦截的5—15秒窗口内重新规划，按本局步长换算。
            return min(self.config.joint_decision_interval_step,
                       self._clock.seconds_to_steps(self._policy.MIN_INTERCEPT_GAP_SECONDS))
        if self.plans_launch_time:
            return self.config.joint_decision_interval_step
        return 0.0
