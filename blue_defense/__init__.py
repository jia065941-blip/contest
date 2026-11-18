# -*- coding: utf-8 -*-
"""蓝方公共接口、共用实现与内置策略导出。"""

from __future__ import annotations

from importlib import import_module

# 公共契约与数据类型：供环境和自定义蓝方实现统一使用。
from .base import (
    BaseBlueDefense,
    BlueClock,
    BlueDecision,
    BlueObservation,
    BluePolicy,
    DefendedAsset,
    InterceptorAssignment,
    InterceptorState,
    OwnEntityState,
    PlannedInterceptorAssignment,
    ScheduledBluePolicy,
    ScheduledInterceptorAssignment,
    ThreatTrack,
)
# 共用实现：集中导出配置构造、运行控制和信息库接口。
from .shared.config import BlueDefenseConfig, create_blue_config
from .shared.controller import BlueDefenseController
from .shared.information import BlueInformationStore
from .shared.registry import BLUE_POLICY_CHOICES, blue_policy_names, build_blue_policy
from envengine.common.BaseStruct import Vector3d


# 内置算法按名称延迟导出，在实际使用时加载对应分组模块。
_LAZY_POLICY_EXPORTS = {
    "B0FixedRatioRandomPolicy": "immediate.b0_fixed_ratio_random",
    "B1NearestInterceptorPolicy": "immediate.b1_nearest_interceptor",
    "B2ThreatPriorityPolicy": "immediate.b2_threat_priority",
    "B3MinCostAssignmentPolicy": "immediate.b3_min_cost_assignment",
    "B4JointTimingAssignmentPolicy": "coordination.b4_joint_timing_assignment",
    "B5SuccessiveDepthCoordinationPolicy": "coordination.b5_successive_depth_coordination",
    "B6MixedFireCoordinationPolicy": "coordination.b6_mixed_fire_coordination",
    "B7PlannedNearestPolicy": "scheduled.b7_planned_nearest",
    "B8ThreatMultiWavePolicy": "scheduled.b8_threat_multi_wave",
    "B9MinCostPlannedMultiWavePolicy": "scheduled.b9_min_cost_planned_multi_wave",
}


def __getattr__(name: str):
    """加载所请求的算法类，并缓存到包级命名空间。"""
    module_name = _LAZY_POLICY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


# 包级公开名称：保持环境、训练脚本和示例的统一导入方式。
__all__ = [
    "BaseBlueDefense",
    "BlueClock",
    "BlueDefenseConfig",
    "create_blue_config",
    "BlueDefenseController",
    "BlueDecision",
    "BlueInformationStore",
    "BlueObservation",
    "BLUE_POLICY_CHOICES",
    "BluePolicy",
    "DefendedAsset",
    "InterceptorAssignment",
    "InterceptorState",
    "OwnEntityState",
    "PlannedInterceptorAssignment",
    "ScheduledBluePolicy",
    "ScheduledInterceptorAssignment",
    "ThreatTrack",
    "Vector3d",
    "blue_policy_names",
    "build_blue_policy",
    *_LAZY_POLICY_EXPORTS,
]
