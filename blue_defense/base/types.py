"""蓝方观测与决策；位置轴为 x 向东、y 向北、z 向上，位置单位米、速度单位米/秒。"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Mapping, Sequence
from envengine.common.BaseStruct import Vector3d


@dataclass(frozen=True)
class ThreatTrack:
    """一条带探测来源和时间的目标记录，局部坐标位置单位为米。"""

    target_id: int
    name: str
    type: int
    pos: Vector3d
    vel: Vector3d
    detect_step: int
    source_id: int = 0


@dataclass(frozen=True)
class InterceptorState:
    """己方自爆虫的位置、速度、可用状态与平台释放方向角限制。"""

    interceptor_id: int
    pos: Vector3d
    vel: Vector3d
    available: bool
    max_launch_angle_deg: float = 180.0


@dataclass(frozen=True)
class DefendedAsset:
    """内置算法整理的保护对象；不是公共观测的必需结构。"""

    asset_id: int
    name: str
    pos: Vector3d
    health: float
    value: float
    entity_type: int


@dataclass(frozen=True)
class OwnEntityState:
    """己方实体当前状态；由策略自行筛选保护对象并确定价值。"""

    entity_id: int
    name: str
    type: int
    pos: Vector3d
    vel: Vector3d
    health: float
    is_visible: bool = True


@dataclass(frozen=True)
class BlueClock:
    """每局固定的步长；蓝方用步，物理计算通过这里换算秒。"""

    start_time_ms: float = 0
    step_duration_ms: float = 1

    def __post_init__(self) -> None:
        for name in ("start_time_ms", "step_duration_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"BlueClock.{name} must be a real number, excluding bool")
            if not math.isfinite(value):
                raise ValueError(f"BlueClock.{name} must be finite")
        if self.step_duration_ms <= 0:
            raise ValueError("BlueClock.step_duration_ms must be positive")

    def time_at(self, step: int) -> float:
        """将全局步号转换成毫秒时刻，不量化策略内部的计划延迟。"""
        if type(step) is not int:
            raise TypeError("blue observation step must be an int, excluding bool")
        if step < 0:
            raise ValueError("blue observation step must be non-negative")
        return self.start_time_ms + step * self.step_duration_ms

    def step_at(self, time_ms: float) -> int:
        """只接受时钟步边界上的时刻，避免丢失毫秒或浮点精度。"""
        if isinstance(time_ms, bool) or not isinstance(time_ms, Real):
            raise TypeError("blue observation time must be a real number, excluding bool")
        if not math.isfinite(time_ms):
            raise ValueError("blue observation time must be finite")
        step = round((time_ms - self.start_time_ms) / self.step_duration_ms)
        if step < 0 or self.time_at(step) != time_ms:
            raise ValueError("blue observation time must lie on a clock step boundary")
        return step

    @staticmethod
    def _finite(value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("clock conversion requires a real number, excluding bool")
        if not math.isfinite(value):
            raise ValueError("clock conversion requires a finite number")
        return value

    def steps_to_seconds(self, steps: float) -> float:
        """步数（可为小数）转换为物理计算使用的秒数。"""
        return self._finite(steps) * self.step_duration_ms / 1000.0

    def seconds_to_steps(self, seconds: float) -> float:
        """秒数转换为步数，保留计划和拦截计算的小数精度。"""
        return self._finite(seconds) * 1000.0 / self.step_duration_ms

    def continuous_step_at(self, time_ms: float) -> float:
        """引擎内部子步时刻转换为连续步号；不改变公开整数步号。"""
        step = (self._finite(time_ms) - self.start_time_ms) / self.step_duration_ms
        if step < 0:
            raise ValueError("blue execution step must be non-negative")
        return step


@dataclass(frozen=True)
class BlueObservation:
    """本批原始探测、己方状态与目标对应已释放自爆虫清单。"""

    step: int
    targets: Sequence[ThreatTrack]
    interceptors: Sequence[InterceptorState]
    own_entities: Sequence[OwnEntityState]
    launched_map: Mapping[int, Sequence[int]] = field(default_factory=dict)
    clock: BlueClock = field(default=BlueClock(), kw_only=True)

    def __post_init__(self) -> None:
        self.clock.time_at(self.step)

    @property
    def sim_step(self) -> int:
        """全局步号的兼容别名，单位为步。"""
        return self.step


# 即时指令由环境在当前决策后执行；计划指令交给环境按时释放。
@dataclass(frozen=True)
class InterceptorAssignment:
    """为一个自爆虫指定目标，由执行端查询蓝方维护的目标状态。"""

    interceptor_id: int
    target_id: int


@dataclass(frozen=True)
class ScheduledInterceptorAssignment:
    """附带全局计划步号与协同标记的释放指令；允许小数步。"""

    interceptor_id: int
    target_id: int
    decision_step: float
    planned_launch_step: float
    estimated_intercept_step: float
    coordination_id: str | None = None
    coordination_mode: str | None = None
    spatial_mode: str | None = None
    launch_pos: Vector3d | None = None
    estimated_intercept_pos: Vector3d | None = None


# 内部计划由延迟策略自行保存，到期后转换成即时指令。
@dataclass(frozen=True)
class PlannedInterceptorAssignment:
    """策略持有的待释放指令、到期步号和波次编号。"""

    assignment: InterceptorAssignment
    due_step: float
    wave_index: int = 0


# 一次完整评估将动作和求解诊断一起返回给环境。
@dataclass(frozen=True)
class BlueDecision:
    """一次策略评估的结果。

    decide 返回 None 表示等待决策时刻，环境继续保留原计划；
    assignments 为空则表示本次评估完成，并选择空的释放结果。
    """

    assignments: tuple[
        InterceptorAssignment | ScheduledInterceptorAssignment,
        ...,
    ] = ()
    solve_stats: dict[str, float | int | str | bool] = field(default_factory=dict)
