"""内置蓝方算法的共用接口、延迟计划管理和几何辅助。"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math

from ..shared.config import DEFAULT_MAX_SHOTS_PER_TARGET
from .types import (
    BlueObservation,
    InterceptorAssignment,
    InterceptorState,
    PlannedInterceptorAssignment,
    ScheduledInterceptorAssignment,
)
from envengine.common.BaseStruct import Vector3d


class BluePolicy(ABC):
    """根据一帧蓝方观测生成追踪指令。"""

    # 算法声明其释放模式，控制器将这些标志提供给环境。
    name = "base"
    plans_launch_time = False
    preserves_pending_launches = False

    def __init__(self, max_shots_per_target: int = DEFAULT_MAX_SHOTS_PER_TARGET):
        self.max_shots_per_target = max(1, int(max_shots_per_target))

    @abstractmethod
    def decide(
        self,
        observation: BlueObservation,
    ) -> list[InterceptorAssignment | ScheduledInterceptorAssignment]:
        """完成一次评估，返回即时或计划释放指令。"""
        pass

    # 基础资源查询只使用本次观测的可用自爆虫与已释放清单。
    def available_interceptors(self, observation: BlueObservation) -> list[InterceptorState]:
        """读取当前可用的自爆虫。"""
        return [item for item in observation.interceptors if item.available]

    def remaining_need(self, observation: BlueObservation, target_id: int) -> int:
        """计算目标距离配置释放配额还差多少个自爆虫。"""
        launched_count = len(observation.launched_map.get(target_id, []))
        return max(0, self.max_shots_per_target - launched_count)

    def reset(self) -> None:
        """重置本局算法状态，基础实现保持原状态。

        B0 默认跨局延续随机序列；控制器可按配置在重置时重新设种。
        """


class ScheduledBluePolicy(BluePolicy):
    """在策略内部保存延迟或分波计划，并在到期时输出即时指令。"""

    def __init__(self, max_shots_per_target: int = DEFAULT_MAX_SHOTS_PER_TARGET):
        super().__init__(max_shots_per_target=max_shots_per_target)
        # 待释放队列负责资源预留，上一帧步号用于识别仿真步号回退。
        self._pending: list[PlannedInterceptorAssignment] = []
        self._last_step = float("-inf")

    def available_interceptors_for_planning(
        self,
        observation: BlueObservation,
    ) -> list[InterceptorState]:
        """取出尚未被计划预留、也尚未释放的可用自爆虫。"""
        self._reset_if_new_round(observation)
        reserved = {item.assignment.interceptor_id for item in self._pending}
        launched = self._launched_interceptor_ids(observation)
        return [
            item
            for item in self.available_interceptors(observation)
            if item.interceptor_id not in reserved
            and item.interceptor_id not in launched
        ]

    def remaining_need_with_pending(
        self,
        observation: BlueObservation,
        target_id: int,
    ) -> int:
        """在目标配额中同时扣除已释放自爆虫和排队中的计划。"""
        pending_count = sum(
            item.assignment.target_id == target_id
            for item in self._pending
        )
        return max(
            0,
            self.remaining_need(observation, target_id) - pending_count,
        )

    def schedule_assignments(
        self,
        observation: BlueObservation,
        plans: list[PlannedInterceptorAssignment],
    ) -> None:
        """将新计划加入队列，同时占用自爆虫和目标释放配额。"""
        # 先清理失效计划，再统计当前预留，保证本批新计划共享同一份余额。
        self._drop_invalid_pending(observation)
        reserved = {item.assignment.interceptor_id for item in self._pending}
        pending_by_target: dict[int, int] = {}
        for item in self._pending:
            target_id = item.assignment.target_id
            pending_by_target[target_id] = (
                pending_by_target.get(target_id, 0) + 1
            )

        # 按传入顺序接受计划，优先保留先占用资源的方案。
        for plan in plans:
            interceptor_id = plan.assignment.interceptor_id
            target_id = plan.assignment.target_id
            if interceptor_id in reserved:
                continue
            remaining = (
                self.remaining_need(observation, target_id)
                - pending_by_target.get(target_id, 0)
            )
            if remaining <= 0:
                continue
            self._pending.append(plan)
            reserved.add(interceptor_id)
            pending_by_target[target_id] = (
                pending_by_target.get(target_id, 0) + 1
            )

    def release_due_assignments(
        self,
        observation: BlueObservation,
    ) -> list[InterceptorAssignment]:
        """按到期步数和波次顺序释放指令，保留未来计划。"""
        # 本次释放只使用当前观测仍然可用的自爆虫与目标。
        self._drop_invalid_pending(observation)
        available_ids = {
            item.interceptor_id
            for item in self.available_interceptors(observation)
        } - self._launched_interceptor_ids(observation)
        launched_by_target = {
            target_id: len(interceptors)
            for target_id, interceptors in observation.launched_map.items()
        }
        target_by_id = {
            target.target_id: target
            for target in observation.targets
        }
        released: list[InterceptorAssignment] = []
        future: list[PlannedInterceptorAssignment] = []
        used_interceptors: set[int] = set()

        # 相同时刻以波次、目标编号和单位编号确定稳定的处理顺序。
        for plan in sorted(
            self._pending,
            key=lambda item: (
                item.due_step,
                item.wave_index,
                item.assignment.target_id,
                item.assignment.interceptor_id,
            ),
        ):
            assignment = plan.assignment
            if plan.due_step > observation.step:
                future.append(plan)
                continue
            if (
                assignment.interceptor_id not in available_ids
                or assignment.interceptor_id in used_interceptors
            ):
                continue
            current_count = launched_by_target.get(assignment.target_id, 0)
            if current_count >= self.max_shots_per_target:
                continue
            current_target = target_by_id.get(assignment.target_id)
            if current_target is None:
                continue
            released.append(assignment)
            used_interceptors.add(assignment.interceptor_id)
            launched_by_target[assignment.target_id] = current_count + 1

        # 到期项在本次评估中消费，队列只留下尚未到期的项。
        self._pending = future
        return released

    def _drop_invalid_pending(self, observation: BlueObservation) -> None:
        """清理自爆虫不可用、目标信息消失或释放配额已满的计划。"""
        target_ids = {target.target_id for target in observation.targets}
        available_ids = {
            item.interceptor_id
            for item in self.available_interceptors(observation)
        } - self._launched_interceptor_ids(observation)
        self._pending = [
            item
            for item in self._pending
            if item.assignment.interceptor_id in available_ids
            and item.assignment.target_id in target_ids
            and self.remaining_need(observation, item.assignment.target_id) > 0
        ]

    def _reset_if_new_round(self, observation: BlueObservation) -> None:
        """发现仿真步号回退时清空计划，并记录当前步号。"""
        if observation.step < self._last_step:
            self._pending.clear()
        self._last_step = observation.step

    def reset(self) -> None:
        """在局边界清空延迟队列与步号记录。"""

        self._pending.clear()
        self._last_step = float("-inf")

    @staticmethod
    def _launched_interceptor_ids(observation: BlueObservation) -> set[int]:
        """将各目标的已释放清单展开为单位编号集合。"""
        return {
            interceptor_id
            for interceptors in observation.launched_map.values()
            for interceptor_id in interceptors
        }


# 几何量统一使用 局部 米制坐标，速度向量单位为米每秒。
def distance_m(a: Vector3d, b: Vector3d, ignore_z: bool = False) -> float:
    """计算两点距离，可按调用需求忽略第三维。"""
    dx = a.x - b.x
    dy = a.y - b.y
    dz = 0.0 if ignore_z else a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def speed_mps(v: Vector3d, default: float = 1.0) -> float:
    """计算速度模长，在接近静止时采用调用方提供的默认值。"""
    speed = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    return speed if speed > 1e-6 else default
