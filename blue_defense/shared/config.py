# -*- coding: utf-8 -*-
"""内置蓝方算法的默认参数与配置构造。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


# 每个目标的默认释放配额由各算法和控制器共用。
DEFAULT_MAX_SHOTS_PER_TARGET = 2


@dataclass(frozen=True)
class BlueDefenseConfig:
    """内置蓝方配置；决策间隔、延迟和显式波次间隔均使用步数。"""

    # 算法选择、单目标配额和随机种子。
    policy_name: str = "b0_fixed_ratio_random"
    max_shots_per_target: int = DEFAULT_MAX_SHOTS_PER_TARGET
    seed: int | None = None
    # B3 的最小费用评估间隔、联合计划评估间隔与延迟释放时长。
    min_cost_decision_interval_step: float = 0.1
    joint_decision_interval_step: float = 100.0
    planning_delay_step: float = 20.0
    # 显式波次间隔为步数，None在绑定场景时补默认值；每波至少一只。
    wave_gap_step: float | None = None
    wave_size: int = 1
    # B0 默认跨局延续随机序列；保护对象可按实体编号覆盖默认价值。
    reset_rng_on_episode: bool = False
    asset_values: Mapping[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """统一算法名并规范化数量与价值映射。"""
        normalized = self.policy_name.strip().lower().replace("-", "_")
        object.__setattr__(self, "policy_name", normalized)
        # 缺省波次间隔保留原5/10秒规则，绑定场景时钟后换算为步。
        # 数量参数转换为正整数，价值映射单独复制为整数键与浮点值。
        object.__setattr__(
            self,
            "max_shots_per_target",
            max(1, int(self.max_shots_per_target)),
        )
        object.__setattr__(self, "wave_size", max(1, int(self.wave_size)))
        object.__setattr__(
            self,
            "asset_values",
            {int(key): float(value) for key, value in self.asset_values.items()},
        )


def create_blue_config(
    *,
    policy_name: str | None = None,
    seed: int | None = None,
    asset_values: Mapping[int, float] | None = None,
) -> BlueDefenseConfig:
    """按蓝方默认参数构造指定算法的配置。"""
    options = {"seed": seed, "asset_values": {} if asset_values is None else asset_values}
    if policy_name is not None:
        options["policy_name"] = policy_name
    return BlueDefenseConfig(**options)
