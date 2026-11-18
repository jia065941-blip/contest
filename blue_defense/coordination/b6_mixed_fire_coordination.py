# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Mapping

from dataclasses import dataclass, replace
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from ..shared.controller import BlueDefenseController
from ..shared.config import DEFAULT_MAX_SHOTS_PER_TARGET, BlueDefenseConfig

from ..base import (
    BlueObservation,
    BluePolicy,
    ScheduledInterceptorAssignment,
)
from .b4_joint_timing_assignment import (
    B4JointTimingAssignmentPolicy,
    _Candidate,
)
from envengine.common.BaseStruct import Vector3d


@dataclass
class _TargetPlan:
    mode: str
    first: ScheduledInterceptorAssignment
    second: ScheduledInterceptorAssignment | None = None


@dataclass(frozen=True)
class _MixedCandidate:
    candidate: _Candidate
    role: str
    objective_cost: float
    coordination_id: str | None = None
    intercept_gap_step: float = 0.0
    depth_separation_m: float = 0.0


class B6MixedFireCoordinationPolicy(B4JointTimingAssignmentPolicy):
    name = "b6_mixed_fire_coordination"
    preserves_pending_launches = True

    MIN_INTERCEPT_GAP_SECONDS = 5.0
    MAX_INTERCEPT_GAP_SECONDS = 15.0
    HEADING_COSINE_MIN = 0.5
    CANDIDATES_PER_LAUNCH_SLOT = 4

    def __init__(self, max_shots_per_target: int = DEFAULT_MAX_SHOTS_PER_TARGET):
        # 混合策略保留成对配额，并记录每个目标采用单个释放还是连续成对释放。
        if max_shots_per_target < 2:
            raise ValueError(
                "B6 mixed-fire coordination requires max_shots_per_target >= 2"
            )
        BluePolicy.__init__(self, max_shots_per_target=max_shots_per_target)
        self.last_solve_stats: dict[str, float | int | str | bool] = {}
        self._plans: dict[int, _TargetPlan] = {}
        self._last_step = float("-inf")

    def reset(self) -> None:
        # 新回合清空目标计划、求解统计和时间基准。
        self.last_solve_stats = {}
        self._plans.clear()
        self._last_step = float("-inf")

    def decide(
        self,
        observation: BlueObservation,
    ) -> list[ScheduledInterceptorAssignment]:
        # 同步已有计划与系统释放回执，再为当前已知目标生成候选。
        started_at = time.perf_counter()
        self._synchronize_plans(observation)
        available = self.available_interceptors(observation)
        candidates: list[_MixedCandidate] = []

        for target in observation.targets:
            plan = self._plans.get(target.target_id)
            assessment = self._assess_threat(observation, target)
            if assessment is None:
                continue
            if plan is None:
                # 按资产威胁与可用自爆虫数量，决定采用单个释放或连续成对释放。
                target_candidates = self._target_candidates(
                    observation,
                    target,
                    available,
                )
                if not target_candidates:
                    continue
                successive = (
                    self._needs_dual_shot(observation, target)
                    and len({
                        item.assignment.interceptor_id
                        for item in target_candidates
                    }) >= 2
                )
                role = "successive_first" if successive else "direct"
                coordination_id = (
                    f"{observation.step}:{target.target_id}"
                    if successive
                    else None
                )
                candidates.extend(
                    _MixedCandidate(
                        candidate=item,
                        role=role,
                        coordination_id=coordination_id,
                        objective_cost=item.cost - assessment.protection_value,
                    )
                    for item in target_candidates
                )
                continue
            # 已确认首个指派的成对计划，进入后续指派的候选构建阶段。
            if (
                plan.mode == "successive"
                and plan.second is None
                and self._first_was_launched(observation, plan)
            ):
                candidates.extend(
                    self._second_candidates(
                        observation,
                        target,
                        available,
                        plan,
                        assessment,
                    )
                )

        if not candidates:
            self._set_stats("no_candidates", started_at, observation, [], [])
            return []

        # 联合求解单个释放、成对首个指派和后续指派的候选，约束共享目标与自爆虫。
        result = milp(
            c=np.array([item.objective_cost for item in candidates]),
            integrality=np.ones(len(candidates)),
            bounds=Bounds(0.0, 1.0),
            constraints=self._constraints(candidates),
            options={
                "mip_rel_gap": 0.0,
                "time_limit": self.SOLVE_TIME_LIMIT_S,
            },
        )
        if result.x is None:
            self._set_stats(
                "infeasible",
                started_at,
                observation,
                candidates,
                [],
                optimal=False,
                mip_gap=float("inf"),
            )
            return []

        # 将选中动作写回各目标计划，协同动作保留同一追踪标识。
        selected = [
            item
            for index, item in enumerate(candidates)
            if result.x[index] > 0.5
        ]
        assignments = []
        for item in selected:
            coordinated = item.role != "direct"
            assignment = replace(
                item.candidate.assignment,
                coordination_id=item.coordination_id if coordinated else None,
                coordination_mode="staggered" if coordinated else None,
                spatial_mode="layered" if coordinated else None,
            )
            target_id = assignment.target_id
            if item.role == "direct":
                self._plans[target_id] = _TargetPlan("direct", assignment)
            elif item.role == "successive_first":
                self._plans[target_id] = _TargetPlan("successive", assignment)
            else:
                self._plans[target_id].second = assignment
            assignments.append(assignment)

        # 汇总单个释放和成对释放的效果，按释放时间稳定输出计划。
        self._set_stats(
            "optimal" if result.status == 0 else "feasible",
            started_at,
            observation,
            candidates,
            selected,
            optimal=result.status == 0,
            mip_gap=float(result.mip_gap),
        )
        return sorted(
            assignments,
            key=lambda item: (
                item.planned_launch_step,
                item.target_id,
                item.interceptor_id,
            ),
        )

    def _second_candidates(
        self,
        observation,
        target,
        available,
        plan: _TargetPlan,
        assessment,
    ) -> list[_MixedCandidate]:
        # 后续指派使用不同自爆虫，并满足接近目标耗时窗口和正向前后间距。
        candidates = []
        for item in self._target_candidates(observation, target, available):
            if item.assignment.interceptor_id == plan.first.interceptor_id:
                continue
            intercept_gap = (
                item.assignment.estimated_intercept_step
                - plan.first.estimated_intercept_step
            )
            if not (
                observation.clock.seconds_to_steps(self.MIN_INTERCEPT_GAP_SECONDS)
                <= intercept_gap
                <= observation.clock.seconds_to_steps(self.MAX_INTERCEPT_GAP_SECONDS)
            ):
                continue
            depth_separation = self._depth_separation(
                plan.first,
                item.assignment,
                target.vel,
            )
            if depth_separation <= 0.0:
                continue
            candidates.append(
                _MixedCandidate(
                    candidate=item,
                    role="successive_second",
                    coordination_id=plan.first.coordination_id,
                    objective_cost=(
                        item.cost
                        - assessment.protection_value
                        * self.SECOND_SHOT_VALUE
                    ),
                    intercept_gap_step=intercept_gap,
                    depth_separation_m=depth_separation,
                )
            )
        return candidates

    def _synchronize_plans(self, observation: BlueObservation) -> None:
        # 时间回退时清空旧局计划，失去信息的目标也释放其计划。
        if observation.step < self._last_step:
            self._plans.clear()
        self._last_step = observation.step
        known_target_ids = {target.target_id for target in observation.targets}
        for target_id in list(self._plans):
            if target_id not in known_target_ids:
                del self._plans[target_id]
                continue
            plan = self._plans[target_id]
            # 到达计划时刻后核对回执，清除未执行的首个或后续指派。
            launched_ids = set(observation.launched_map.get(target_id, []))
            if (
                observation.step >= plan.first.planned_launch_step
                and plan.first.interceptor_id not in launched_ids
            ):
                del self._plans[target_id]
                continue
            if (
                plan.second is not None
                and observation.step >= plan.second.planned_launch_step
                and plan.second.interceptor_id not in launched_ids
            ):
                plan.second = None

    @staticmethod
    def _first_was_launched(
        observation: BlueObservation,
        plan: _TargetPlan,
    ) -> bool:
        # 同时满足计划时刻和系统已释放记录，才确认首个指派完成。
        return (
            observation.step >= plan.first.planned_launch_step
            and plan.first.interceptor_id
            in observation.launched_map.get(plan.first.target_id, [])
        )

    def _target_candidates(self, observation, target, available) -> list[_Candidate]:
        # 枚举目标与可用自爆虫的可行时隙，再压缩候选规模。
        candidates = [
            candidate
            for interceptor in available
            for candidate in self._candidates(observation, target, interceptor)
        ]
        return self._reduce_candidates(candidates)

    def _needs_dual_shot(self, observation, target) -> bool:
        # 会损伤最高价值资产且方向满足阈值的接近目标采用成对释放。
        highest_value = max(asset.value for asset in observation.assets)
        return any(
            asset.value >= highest_value
            and self._expected_damage(target.type, asset.entity_type) > 0.0
            and self._heading_cosine(target, asset) >= self.HEADING_COSINE_MIN
            for asset in observation.assets
        )

    def _heading_cosine(self, target, asset) -> float:
        # 用速度与目标指向资产方向的夹角余弦衡量目标接近趋势。
        speed = self._length(target.vel)
        direction = Vector3d(
            asset.pos.x - target.pos.x,
            asset.pos.y - target.pos.y,
            asset.pos.z - target.pos.z,
        )
        distance = self._length(direction)
        if speed <= 1e-6:
            return 0.0
        if distance <= 1e-6:
            return 1.0
        return (
            target.vel.x * direction.x
            + target.vel.y * direction.y
            + target.vel.z * direction.z
        ) / (speed * distance)

    def _reduce_candidates(self, candidates: list[_Candidate]) -> list[_Candidate]:
        # 各释放时隙保留成本最低的若干个不同自爆虫。
        by_launch_step: dict[float, list[_Candidate]] = {}
        for candidate in candidates:
            by_launch_step.setdefault(
                candidate.assignment.planned_launch_step,
                [],
            ).append(candidate)
        reduced = []
        for launch_step in sorted(by_launch_step):
            used_interceptors: set[int] = set()
            for candidate in sorted(
                by_launch_step[launch_step],
                key=lambda item: (item.cost, item.assignment.interceptor_id),
            ):
                interceptor_id = candidate.assignment.interceptor_id
                if interceptor_id in used_interceptors:
                    continue
                reduced.append(candidate)
                used_interceptors.add(interceptor_id)
                if len(used_interceptors) >= self.CANDIDATES_PER_LAUNCH_SLOT:
                    break
        return reduced

    @classmethod
    def _depth_separation(
        cls,
        first: ScheduledInterceptorAssignment,
        second: ScheduledInterceptorAssignment,
        target_velocity: Vector3d,
    ) -> float:
        # 沿目标速度方向投影两次预测相遇点，计算前后间距。
        first_point = first.estimated_intercept_pos
        second_point = second.estimated_intercept_pos
        speed = cls._length(target_velocity)
        if first_point is None or second_point is None or speed <= 1e-9:
            return 0.0
        return (
            (second_point.x - first_point.x) * target_velocity.x
            + (second_point.y - first_point.y) * target_velocity.y
            + (second_point.z - first_point.z) * target_velocity.z
        ) / speed

    @staticmethod
    def _constraints(candidates: list[_MixedCandidate]) -> LinearConstraint:
        # 每个目标和每个自爆虫在本轮最多选择一个候选。
        groups: dict[tuple[str, int], list[int]] = {}
        for index, item in enumerate(candidates):
            assignment = item.candidate.assignment
            groups.setdefault(("target", assignment.target_id), []).append(index)
            groups.setdefault(
                ("interceptor", assignment.interceptor_id),
                [],
            ).append(index)
        # 将共享资源的候选索引组成稀疏约束矩阵。
        rows = []
        columns = []
        for row, key in enumerate(sorted(groups)):
            indices = groups[key]
            rows.extend([row] * len(indices))
            columns.extend(indices)
        matrix = coo_matrix(
            (np.ones(len(rows)), (rows, columns)),
            shape=(len(groups), len(candidates)),
        )
        return LinearConstraint(matrix, 0.0, 1.0)

    def _set_stats(
        self,
        status: str,
        started_at: float,
        observation: BlueObservation,
        candidates: list[_MixedCandidate],
        selected: list[_MixedCandidate],
        *,
        optimal: bool = True,
        mip_gap: float = 0.0,
    ) -> None:
        # 分别统计单个释放、成对首个指派及后续指派，并汇总成对协同效果。
        selected_seconds = [
            item for item in selected if item.role == "successive_second"
        ]
        pair_count = len(selected_seconds)
        self.last_solve_stats = {
            "status": status,
            "optimal": optimal,
            "fallback": False,
            "candidate_count": len(candidates),
            "observed_target_count": len(observation.targets),
            "candidate_target_count": len({
                item.candidate.assignment.target_id
                for item in candidates
            }),
            "selected_target_count": len({
                item.candidate.assignment.target_id
                for item in selected
            }),
            "selected_assignment_count": len(selected),
            "selected_direct_fire_count": sum(
                item.role == "direct" for item in selected
            ),
            "selected_successive_first_count": sum(
                item.role == "successive_first" for item in selected
            ),
            "selected_second_shot_count": pair_count,
            "solve_time_s": time.perf_counter() - started_at,
            "mip_gap": mip_gap,
            "selected_pair_count": pair_count,
            "selected_temporal_pair_count": pair_count,
            "selected_staggered_pair_count": pair_count,
            "selected_synchronized_pair_count": 0,
            "selected_spatial_pair_count": pair_count,
            "selected_concentrated_pair_count": 0,
            "selected_layered_pair_count": pair_count,
            "selected_crossfire_pair_count": 0,
            "selected_spatiotemporal_pair_count": pair_count,
            "intercept_gap_step_total": sum(
                item.intercept_gap_step for item in selected_seconds
            ),
            "intercept_separation_m_total": sum(
                item.depth_separation_m for item in selected_seconds
            ),
            "approach_angle_deg_total": 0.0,
            "asset_clearance_m_total": 0.0,
            "coordination_mode": "mixed_single_successive",
        }


class BlueDefense(BlueDefenseController):
    """B6 单个与连续成对释放混合策略的公共入口。"""

    def __init__(
        self,
        *,
        seed: int | None = None,
        asset_values: Mapping[int, float] | None = None,
    ):
        # 统一入口参数交给 B6 配置，其余参数沿用该策略默认值。
        super().__init__(BlueDefenseConfig(
            policy_name="b6_mixed_fire_coordination",
            seed=seed,
            asset_values=dict(asset_values) if asset_values is not None else {},
        ))
