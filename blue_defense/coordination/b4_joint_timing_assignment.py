# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Mapping

from dataclasses import dataclass
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from ..shared.controller import BlueDefenseController
from ..base import (
    BlueClock,
    BlueObservation,
    BluePolicy,
    InterceptorAssignment,
    InterceptorState,
    ScheduledInterceptorAssignment,
    ThreatTrack,
    distance_m,
    speed_mps,
)
from ..immediate.b3_min_cost_assignment import B3MinCostAssignmentPolicy
from ..shared.config import DEFAULT_MAX_SHOTS_PER_TARGET, BlueDefenseConfig
from envengine.common.BaseStruct import Vector3d


@dataclass(frozen=True)
class _Candidate:
    assignment: ScheduledInterceptorAssignment
    cost: float


@dataclass(frozen=True)
class _ThreatAssessment:
    impact_step: float
    protection_value: float


class B4JointTimingAssignmentPolicy(BluePolicy):
    name = "b4_joint_timing_assignment"
    plans_launch_time = True
    preserves_pending_launches = True

    # 候选规划使用的行动范围、速度、时间网格和求解时限。
    MAX_LAUNCH_RANGE_M = 600_000.0
    INTERCEPTOR_SPEED_MPS = 1200.0
    PLANNING_HORIZON_SECONDS = 30.0
    PLANNING_STEP_SECONDS = 2.0
    SAFETY_MARGIN_SECONDS = 5.0
    SOLVE_TIME_LIMIT_S = 3.0
    # 优化目标中的自爆虫成本、接近目标耗时成本和追加释放收益。
    BASE_AMMO_COST = 0.01
    INTERCEPT_TIME_COST = 0.05
    SECOND_SHOT_VALUE = 0.6

    def __init__(self, max_shots_per_target: int = DEFAULT_MAX_SHOTS_PER_TARGET):
        super().__init__(max_shots_per_target=max_shots_per_target)
        # 保留即时最小代价算法，供联合求解降级时使用。
        self.immediate_policy = B3MinCostAssignmentPolicy(
            max_shots_per_target=max_shots_per_target,
        )
        self.last_solve_stats: dict[str, float | int | str | bool] = {}

    def reset(self) -> None:
        # 清理本局求解统计，并重置即时降级算法。
        self.last_solve_stats = {}
        self.immediate_policy.reset()

    def decide(
        self,
        observation: BlueObservation,
    ) -> list[InterceptorAssignment | ScheduledInterceptorAssignment]:
        # 遍历当前目标与可用自爆虫，构建各释放时刻的候选指派。
        started_at = time.perf_counter()
        candidates = []
        available = self.available_interceptors(observation)
        for target in observation.targets:
            for interceptor in available:
                candidates.extend(self._candidates(observation, target, interceptor))
        # 无候选时记录完整统计，结束本次评估。
        if not candidates:
            self.last_solve_stats = {
                "status": "no_candidates",
                "optimal": True,
                "fallback": False,
                "candidate_count": 0,
                "observed_target_count": len(observation.targets),
                "candidate_target_count": 0,
                "selected_target_count": 0,
                "selected_assignment_count": 0,
                "solve_time_s": time.perf_counter() - started_at,
                "mip_gap": 0.0,
            }
            return []

        candidate_target_ids = {
            item.assignment.target_id
            for item in candidates
        }
        # 为目标剩余配额建立逐次收益变量，与候选指派共同优化。
        shot_levels = [
            (target.target_id, level)
            for target in observation.targets
            if target.target_id in candidate_target_ids
            for level in range(
                1,
                self.remaining_need(observation, target.target_id) + 1,
            )
        ]
        objective = self._objective(
            observation,
            candidates,
            shot_levels,
        )
        # 在自爆虫占用、目标配额和收益档位约束下求解二元整数规划。
        result = milp(
            c=objective,
            integrality=np.ones(len(objective)),
            bounds=Bounds(0.0, 1.0),
            constraints=self._constraints(
                observation,
                candidates,
                shot_levels,
            ),
            options={
                "mip_rel_gap": 0.0,
                "time_limit": self.SOLVE_TIME_LIMIT_S,
            },
        )
        # 求解器未给出可用解时，执行即时分配并标记降级统计。
        if result.x is None:
            fallback_assignments = self.immediate_policy.decide(observation)
            self.last_solve_stats = {
                "status": "b3_immediate",
                "optimal": False,
                "fallback": True,
                "candidate_count": len(candidates),
                "observed_target_count": len(observation.targets),
                "candidate_target_count": len(candidate_target_ids),
                "selected_target_count": len(
                    {item.target_id for item in fallback_assignments}
                ),
                "selected_assignment_count": len(fallback_assignments),
                "solve_time_s": time.perf_counter() - started_at,
                "mip_gap": float("inf"),
            }
            return fallback_assignments
        # 提取选中候选，记录求解质量、规模与耗时。
        selected = [
            candidate
            for index, candidate in enumerate(candidates)
            if result.x[index] > 0.5
        ]
        self.last_solve_stats = {
            "status": "optimal" if result.status == 0 else "feasible",
            "optimal": result.status == 0,
            "fallback": False,
            "candidate_count": len(candidates),
            "observed_target_count": len(observation.targets),
            "candidate_target_count": len(candidate_target_ids),
            "selected_target_count": len(
                {item.assignment.target_id for item in selected}
            ),
            "selected_assignment_count": len(selected),
            "solve_time_s": time.perf_counter() - started_at,
            "mip_gap": float(result.mip_gap),
        }
        # 按释放时间、候选成本和实体编号确定输出顺序。
        return [
            item.assignment
            for item in sorted(
                selected,
                key=lambda item: (
                    item.assignment.planned_launch_step,
                    item.cost,
                    item.assignment.target_id,
                    item.assignment.interceptor_id,
                ),
            )
        ]

    def _candidates(
        self,
        observation: BlueObservation,
        target: ThreatTrack,
        interceptor: InterceptorState,
    ) -> list[_Candidate]:
        # 结合释放时间网格与目标威胁评估，准备单个目标—自爆虫对的候选。
        feasible_slots = self._feasible_slots(observation, target, interceptor)
        assessment = self._assess_threat(observation, target)
        if assessment is None:
            return []
        candidates = []
        # 用相对运动求接近目标耗时，并预留目标命中前的安全间隔。
        for planned_step, target_pos, _ in feasible_slots:
            intercept_duration = self._relative_intercept_duration_step(
                interceptor.pos,
                target_pos,
                target.vel,
                observation.clock,
            )
            if intercept_duration is None:
                continue
            estimated_intercept_step = planned_step + intercept_duration
            if (
                estimated_intercept_step
                + observation.clock.seconds_to_steps(self.SAFETY_MARGIN_SECONDS)
                >= assessment.impact_step
            ):
                continue
            # 为可行候选保存代价、释放位置和预计相遇位置。
            cost = self._assignment_cost(
                observation,
                target,
                planned_step,
                intercept_duration,
                assessment,
            )
            candidates.append(
                _Candidate(
                    assignment=ScheduledInterceptorAssignment(
                        interceptor_id=interceptor.interceptor_id,
                        target_id=target.target_id,
                        decision_step=observation.step,
                        planned_launch_step=planned_step,
                        estimated_intercept_step=estimated_intercept_step,
                        launch_pos=interceptor.pos,
                        estimated_intercept_pos=self._project_intercept_position(
                            target_pos,
                            target.vel,
                            intercept_duration,
                            observation.clock,
                        ),
                    ),
                    cost=cost,
                )
            )
        return candidates

    def _objective(
        self,
        observation: BlueObservation,
        candidates: list[_Candidate],
        shot_levels: list[tuple[int, int]],
    ) -> np.ndarray:
        # 保护价值转为收益，后续释放按递减比例追加收益。
        assessments = {
            target.target_id: self._assess_threat(observation, target)
            for target in observation.targets
        }
        shot_values = [
            -assessments[target_id].protection_value
            * (
                1.0
                if level == 1
                else self.SECOND_SHOT_VALUE ** (level - 1)
            )
            for target_id, level in shot_levels
        ]
        # 目标向量由候选成本和各目标逐次收益两部分组成。
        return np.array(
            [candidate.cost for candidate in candidates]
            + shot_values
        )

    def _constraints(
        self,
        observation: BlueObservation,
        candidates: list[_Candidate],
        shot_levels: list[tuple[int, int]],
    ) -> LinearConstraint:
        # 将候选按自爆虫和目标分组，约束单个自爆虫占用与单目标剩余配额。
        group_indices: dict[tuple, list[int]] = {}
        group_limits: dict[tuple, float] = {}
        for index, item in enumerate(candidates):
            assignment = item.assignment
            interceptor_key = ("interceptor", assignment.interceptor_id)
            target_key = ("target", assignment.target_id)
            for key in (interceptor_key, target_key):
                group_indices.setdefault(key, []).append(index)
            group_limits[interceptor_key] = 1.0
            group_limits[target_key] = float(
                self.remaining_need(observation, assignment.target_id)
            )

        groups = [
            (group_indices[key], group_limits[key])
            for key in sorted(group_indices)
        ]

        # 逐组组装稀疏约束矩阵的行、列、系数及上下界。
        rows = []
        columns = []
        values = []
        lower_bounds = []
        upper_bounds = []
        for row, (indices, _) in enumerate(groups):
            rows.extend([row] * len(indices))
            columns.extend(indices)
            values.extend([1.0] * len(indices))
            lower_bounds.append(0.0)
            upper_bounds.append(groups[row][1])

        # 将目标选中候选数与启用的收益档位数绑定为相等。
        target_ids = sorted({target_id for target_id, _ in shot_levels})
        for target_id in target_ids:
            row = len(lower_bounds)
            target_candidates = [
                index
                for index, item in enumerate(candidates)
                if item.assignment.target_id == target_id
            ]
            target_levels = [
                len(candidates) + index
                for index, (item_target_id, _) in enumerate(shot_levels)
                if item_target_id == target_id
            ]
            rows.extend([row] * (len(target_candidates) + len(target_levels)))
            columns.extend(target_candidates + target_levels)
            values.extend(
                [1.0] * len(target_candidates)
                + [-1.0] * len(target_levels)
            )
            lower_bounds.append(0.0)
            upper_bounds.append(0.0)

        # 追加释放档位依赖前一档位，形成连续的逐次收益序列。
        for index, (target_id, level) in enumerate(shot_levels):
            if level == 1:
                continue
            previous_index = shot_levels.index((target_id, level - 1))
            row = len(lower_bounds)
            rows.extend([row, row])
            columns.extend(
                [
                    len(candidates) + index,
                    len(candidates) + previous_index,
                ]
            )
            values.extend([1.0, -1.0])
            lower_bounds.append(float("-inf"))
            upper_bounds.append(0.0)

        # 转换为求解器使用的稀疏线性约束。
        matrix = coo_matrix(
            (values, (rows, columns)),
            shape=(len(lower_bounds), len(candidates) + len(shot_levels)),
        )
        return LinearConstraint(
            matrix,
            np.array(lower_bounds),
            np.array(upper_bounds),
        )

    def _feasible_slots(
        self,
        observation: BlueObservation,
        target: ThreatTrack,
        interceptor: InterceptorState,
    ) -> list[tuple[float, Vector3d, float]]:
        # 在规划窗口内离散预测目标位置，保留满足行动范围和角度的时刻。
        slots = []
        slot_count = math.floor(self.PLANNING_HORIZON_SECONDS / self.PLANNING_STEP_SECONDS)
        for slot_index in range(slot_count + 1):
            delay_step = observation.clock.seconds_to_steps(
                slot_index * self.PLANNING_STEP_SECONDS,
            )
            target_pos = self._project_position(target, delay_step, observation.clock)
            if not self._launch_eligible(interceptor, target_pos, target.vel):
                continue
            intercept_duration = observation.clock.seconds_to_steps(
                distance_m(interceptor.pos, target_pos)
                / self.INTERCEPTOR_SPEED_MPS
            )
            slots.append((observation.step + delay_step, target_pos, intercept_duration))
        return slots

    def _assignment_cost(
        self,
        observation: BlueObservation,
        target: ThreatTrack,
        planned_step: float,
        intercept_duration: float,
        assessment: _ThreatAssessment | None = None,
    ) -> float:
        # 接近目标耗时按目标预计命中窗口归一化，再叠加基础资源消耗成本。
        if assessment is None:
            assessment = self._assess_threat(observation, target)
        if assessment is None:
            return float("inf")
        impact_window = max(
            assessment.impact_step - observation.step,
            observation.clock.seconds_to_steps(0.001),
        )
        intercept_elapsed = (
            planned_step
            + intercept_duration
            - observation.step
        )
        return (
            self.BASE_AMMO_COST
            + self.INTERCEPT_TIME_COST
            * intercept_elapsed
            / impact_window
        )

    def _assess_threat(
        self,
        observation: BlueObservation,
        target: ThreatTrack,
    ) -> _ThreatAssessment | None:
        # 目标运动速度和保护对象信息用于估计命中时刻与保护收益。
        target_speed = self._length(target.vel)
        if target_speed <= 1e-6 or not observation.assets:
            return None

        # 航向与目标到保护对象的夹角余弦表示航向匹配程度。
        def alignment(asset) -> float:
            direction = Vector3d(
                asset.pos.x - target.pos.x,
                asset.pos.y - target.pos.y,
                asset.pos.z - target.pos.z,
            )
            distance = self._length(direction)
            if distance <= 1e-6:
                return 1.0
            return (
                target.vel.x * direction.x
                + target.vel.y * direction.y
                + target.vel.z * direction.z
            ) / (target_speed * distance)

        # 选取航向最接近的保护对象，估计命中时间与相对伤害。
        asset = max(observation.assets, key=alignment)
        impact_distance = distance_m(target.pos, asset.pos)
        impact_duration_step = observation.clock.seconds_to_steps(impact_distance / target_speed)
        damage = self._expected_damage(target.type, asset.entity_type)
        damage_ratio = min(damage / max(asset.health, 1e-6), 1.0)
        # 将保护对象价值、伤害比例和命中紧迫度合并为保护收益。
        urgency = 1.0 + min(
            2.0,
            observation.clock.seconds_to_steps(self.PLANNING_HORIZON_SECONDS)
            / max(impact_duration_step, observation.clock.seconds_to_steps(0.001)),
        )
        return _ThreatAssessment(
            impact_step=observation.step + impact_duration_step,
            protection_value=asset.value * damage_ratio * urgency,
        )

    @staticmethod
    def _expected_damage(target_type: int, asset_type: int) -> float:
        # 使用手册和当前引擎的基础伤害，供B4—B6的保护收益估计共用。
        is_ship = asset_type == 9500
        if target_type == 21002:
            return 1.0 if is_ship else 0.0
        if target_type == 21000:
            return 0.0 if is_ship else 15.0
        if target_type == 21001:
            return 0.0 if is_ship else 5.0
        return 0.0

    def _relative_intercept_duration_step(
        self,
        interceptor_pos: Vector3d,
        target_pos: Vector3d,
        target_vel: Vector3d,
        clock: BlueClock,
    ) -> float | None:
        # 匀速目标与定速自爆虫的相对运动形成距离—时间二次方程。
        relative_x = target_pos.x - interceptor_pos.x
        relative_y = target_pos.y - interceptor_pos.y
        relative_z = target_pos.z - interceptor_pos.z
        a = (
            target_vel.x * target_vel.x
            + target_vel.y * target_vel.y
            + target_vel.z * target_vel.z
            - self.INTERCEPTOR_SPEED_MPS * self.INTERCEPTOR_SPEED_MPS
        )
        b = 2.0 * (
            relative_x * target_vel.x
            + relative_y * target_vel.y
            + relative_z * target_vel.z
        )
        c = (
            relative_x * relative_x
            + relative_y * relative_y
            + relative_z * relative_z
        )
        # 分别求解退化的一次方程和标准二次方程。
        if abs(a) < 1e-9:
            if abs(b) < 1e-9:
                return None
            roots = [-c / b]
        else:
            discriminant = b * b - 4.0 * a * c
            if discriminant < 0:
                return None
            root = math.sqrt(discriminant)
            roots = [
                (-b - root) / (2.0 * a),
                (-b + root) / (2.0 * a),
            ]
        # 取最早的正时间解，并将物理秒转换为本局的浮点步数。
        positive_roots = [value for value in roots if value > 0]
        if not positive_roots:
            return None
        return clock.seconds_to_steps(min(positive_roots))

    @staticmethod
    def _project_intercept_position(
        target_pos: Vector3d,
        target_vel: Vector3d,
        intercept_duration_step: float,
        clock: BlueClock,
    ) -> Vector3d:
        # 按接近目标耗时外推目标的预计相遇位置。
        duration_s = clock.steps_to_seconds(intercept_duration_step)
        return Vector3d(
            target_pos.x + target_vel.x * duration_s,
            target_pos.y + target_vel.y * duration_s,
            target_pos.z + target_vel.z * duration_s,
        )

    def _asset_arrival_step(
        self,
        observation: BlueObservation,
        target: ThreatTrack,
    ) -> float | None:
        if not observation.assets:
            return None
        nearest_asset = min(
            observation.assets,
            key=lambda asset: distance_m(asset.pos, target.pos),
        )
        # 用最近保护对象距离和目标速度估算绝对到达时刻。
        travel_duration_step = observation.clock.seconds_to_steps(
            distance_m(nearest_asset.pos, target.pos)
            / speed_mps(target.vel, default=300.0)
        )
        return observation.step + travel_duration_step

    def _threat_score(
        self,
        observation: BlueObservation,
        target: ThreatTrack,
    ) -> float:
        # 基础类型权重叠加最近保护对象价值与到达紧迫度。
        type_score = {
            21000: 30.0,
            21001: 20.0,
            21002: 10.0,
        }.get(target.type, 1.0)
        if not observation.assets:
            return type_score
        nearest_asset = min(
            observation.assets,
            key=lambda asset: distance_m(asset.pos, target.pos),
        )
        eta = (
            distance_m(nearest_asset.pos, target.pos)
            / speed_mps(target.vel, default=300.0)
        )
        return type_score + nearest_asset.value / max(eta, 1.0) * 1000.0

    @staticmethod
    def _project_position(target: ThreatTrack, delay_step: float, clock: BlueClock) -> Vector3d:
        # 将规划延迟转换为秒，按目标速度外推释放时刻的位置。
        delay_s = clock.steps_to_seconds(delay_step)
        return Vector3d(
            target.pos.x + target.vel.x * delay_s,
            target.pos.y + target.vel.y * delay_s,
            target.pos.z + target.vel.z * delay_s,
        )

    def _launch_eligible(
        self,
        interceptor: InterceptorState,
        target_pos: Vector3d,
        target_vel: Vector3d,
    ) -> bool:
        # 根据释放点到目标的方向检查行动范围与平台释放方向角。
        relative = Vector3d(
            interceptor.pos.x - target_pos.x,
            interceptor.pos.y - target_pos.y,
            interceptor.pos.z - target_pos.z,
        )
        if self._length(relative) >= self.MAX_LAUNCH_RANGE_M:
            return False
        return self._angle_deg(relative, target_vel) < interceptor.max_launch_angle_deg

    @staticmethod
    def _length(vector: Vector3d) -> float:
        # 三维向量模长供航向、距离和相对运动计算共用。
        return math.sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z)

    @classmethod
    def _angle_deg(cls, left: Vector3d, right: Vector3d) -> float:
        # 由夹角余弦计算角度，并限制反余弦输入以处理浮点误差。
        denominator = cls._length(left) * cls._length(right)
        if denominator == 0:
            return 0.0
        cosine = (left.x * right.x + left.y * right.y + left.z * right.z) / denominator
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


class BlueDefense(BlueDefenseController):
    """联合时序追踪算法的统一运行入口。"""

    def __init__(
        self,
        *,
        seed: int | None = None,
        asset_values: Mapping[int, float] | None = None,
    ):
        # 绑定本文件的联合时序策略，以共享种子和保护对象价值构造控制器。
        super().__init__(BlueDefenseConfig(
            policy_name="b4_joint_timing_assignment",
            seed=seed,
            asset_values=dict(asset_values) if asset_values is not None else {},
        ))
