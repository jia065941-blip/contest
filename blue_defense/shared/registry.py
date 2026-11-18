# -*- coding: utf-8 -*-
"""按策略名称加载并构造内置蓝方算法。"""

from __future__ import annotations

from importlib import import_module

from ..base import BluePolicy
from .config import DEFAULT_MAX_SHOTS_PER_TARGET, BlueDefenseConfig


# 名称映射到分组模块和算法类；注册表自身只保存定位信息。
_POLICIES: dict[str, tuple[str, str]] = {
    # B0—B3 每次评估直接给出释放分配。
    "b0_fixed_ratio_random": ("immediate.b0_fixed_ratio_random", "B0FixedRatioRandomPolicy"),
    "b1_nearest_interceptor": ("immediate.b1_nearest_interceptor", "B1NearestInterceptorPolicy"),
    "b2_threat_priority": ("immediate.b2_threat_priority", "B2ThreatPriorityPolicy"),
    "b3_min_cost_assignment": ("immediate.b3_min_cost_assignment", "B3MinCostAssignmentPolicy"),
    # B4—B6 将协同释放计划交给环境维护和执行。
    "b4_joint_timing_assignment": ("coordination.b4_joint_timing_assignment", "B4JointTimingAssignmentPolicy"),
    "b5_successive_depth_coordination": (
        "coordination.b5_successive_depth_coordination",
        "B5SuccessiveDepthCoordinationPolicy",
    ),
    "b6_mixed_fire_coordination": (
        "coordination.b6_mixed_fire_coordination",
        "B6MixedFireCoordinationPolicy",
    ),
    # B7—B9 在策略内部维护延迟或波次计划，到期后给出即时分配。
    "b7_planned_nearest": ("scheduled.b7_planned_nearest", "B7PlannedNearestPolicy"),
    "b8_threat_multi_wave": ("scheduled.b8_threat_multi_wave", "B8ThreatMultiWavePolicy"),
    "b9_min_cost_planned_multi_wave": (
        "scheduled.b9_min_cost_planned_multi_wave",
        "B9MinCostPlannedMultiWavePolicy",
    ),
}

# 选择列表保留注册顺序；名称查询使用排序结果，便于展示。
BLUE_POLICY_CHOICES = tuple(_POLICIES)


def blue_policy_names() -> tuple[str, ...]:
    """返回按名称排序的内置算法编号。"""
    return tuple(sorted(_POLICIES))


def build_blue_policy(
    config: BlueDefenseConfig | str | None = None,
    max_shots_per_target: int = DEFAULT_MAX_SHOTS_PER_TARGET,
    seed: int | None = None,
    *,
    name: str | None = None,
) -> BluePolicy:
    """接受配置或算法名称，按需加载并构造一个内置算法。"""

    # 将名称式调用统一为配置对象，后续构造共用同一组默认值。
    if not isinstance(config, BlueDefenseConfig):
        if name is not None and config is not None:
            raise TypeError("pass either config/policy name or name=, not both")
        config = BlueDefenseConfig(
            policy_name=name or config or "b0_fixed_ratio_random",
            max_shots_per_target=max_shots_per_target,
            seed=seed,
        )

    # 名称确定后才加载对应模块，求解器依赖随所选算法一同加载。
    policy_spec = _POLICIES.get(config.policy_name)
    if policy_spec is None:
        valid_names = ", ".join(blue_policy_names())
        raise ValueError(
            f"Unknown blue policy '{config.policy_name}'. Valid policies: {valid_names}"
        )

    module_name, class_name = policy_spec
    module = import_module(f"{__package__.rsplit('.', 1)[0]}.{module_name}")
    policy_cls = getattr(module, class_name)
    # 所有算法接收目标配额；随机和时间参数按各算法的构造接口传入。
    kwargs: dict[str, object] = {
        "max_shots_per_target": config.max_shots_per_target,
    }
    if config.policy_name == "b0_fixed_ratio_random":
        kwargs["seed"] = config.seed
    elif config.policy_name == "b7_planned_nearest":
        kwargs["planning_delay"] = config.planning_delay_step
    elif config.policy_name == "b8_threat_multi_wave":
        kwargs.update(
            wave_gap=config.wave_gap_step,
            wave_size=config.wave_size,
        )
    elif config.policy_name == "b9_min_cost_planned_multi_wave":
        kwargs.update(
            planning_delay=config.planning_delay_step,
            wave_gap=config.wave_gap_step,
            wave_size=config.wave_size,
        )
    return policy_cls(**kwargs)
