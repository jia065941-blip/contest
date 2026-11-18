"""统一导出蓝方公共基类、观测动作类型及内置算法辅助。"""

# 策略文件和环境从此处取得同一组类型，保持对象身份一致。
from .types import (
    BlueClock,
    BlueDecision,
    BlueObservation,
    DefendedAsset,
    InterceptorAssignment,
    InterceptorState,
    OwnEntityState,
    PlannedInterceptorAssignment,
    ScheduledInterceptorAssignment,
    ThreatTrack,
)
from .policy import BluePolicy, ScheduledBluePolicy, distance_m, speed_mps
from .base_defense import BaseBlueDefense
from envengine.common.BaseStruct import Vector3d


# 显式列出公共符号，供蓝方实现和现有训练入口使用。
__all__ = [
    "BaseBlueDefense",
    "BlueClock",
    "BlueDecision",
    "BlueObservation",
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
    "distance_m",
    "speed_mps",
]
