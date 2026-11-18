# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Mapping

from dataclasses import dataclass, replace
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from ..shared.controller import BlueDefenseController
from ..base import BlueObservation, BluePolicy, ScheduledInterceptorAssignment
from .b4_joint_timing_assignment import B4JointTimingAssignmentPolicy, _Candidate
from ..shared.config import DEFAULT_MAX_SHOTS_PER_TARGET, BlueDefenseConfig
from envengine.common.BaseStruct import Vector3d


@dataclass
class _Engagement:
    coordination_id: str
    first: ScheduledInterceptorAssignment
    second: ScheduledInterceptorAssignment | None = None


@dataclass(frozen=True)
class _StageCandidate:
    candidate: _Candidate
    role: str
    coordination_id: str
    objective_cost: float
    intercept_gap_step: float = 0.0
    depth_separation_m: float = 0.0


class B5SuccessiveDepthCoordinationPolicy(B4JointTimingAssignmentPolicy):
    name = "b5_successive_depth_coordination"
    preserves_pending_launches = True

    MIN_INTERCEPT_GAP_SECONDS = 5.0
    CANDIDATES_PER_LAUNCH_SLOT = 4

    def __init__(self, max_shots_per_target: int = DEFAULT_MAX_SHOTS_PER_TARGET):
        # 前后协同需要成对配额，并维护每个目标的跨帧追踪计划。
        if max_shots_per_target < 2:
            raise ValueError(
                "B5 successive-depth coordination requires "
                "max_shots_per_target >= 2"
            )
        BluePolicy.__init__(self, max_shots_per_target=max_shots_per_target)
        self.last_solve_stats: dict[str, float | int | str | bool] = {}
        self._engagements: dict[int, _Engagement] = {}
        self._last_step = float("-inf")

    def reset(self) -> None:
        # 新回合清空追踪计划、求解统计和时间基准。
        self.last_solve_stats = {}
        self._engagements.clear()
        self._last_step = float("-inf")

    def decide(
        self,
        observation: BlueObservation,
    ) -> list[ScheduledInterceptorAssignment]:
        # 先按当前信息和释放回执同步计划，再读取本轮可用自爆虫。
        started_at = time.perf_counter()
        self._synchronize_engagements(observation)
        available = self.available_interceptors(observation)
        stage_candidates: list[_StageCandidate] = []

        for target in observation.targets:
            engagement = self._engagements.get(target.target_id)
            if engagement is None:
                # 新目标至少需要两个不同自爆虫，首个指派的收益计入目标保护价值。
                candidates = self._target_candidates(
                    observation,
                    target,
                    available,
                )
                if len({
                    item.assignment.interceptor_id
                    for item in candidates
                }) < 2:
                    continue
                assessment = self._assess_threat(observation, target)
                if assessment is None:
                    continue
                coordination_id = (
                    f"{observation.step}:{target.target_id}"
                )
                stage_candidates.extend(
                    _StageCandidate(
                        candidate=item,
                        role="first",
                        coordination_id=coordination_id,
                        objective_cost=(
                            item.cost - assessment.protection_value
                        ),
                    )
                    for item in candidates
                )
                continue

            # 首个自爆虫确认出巢后，才为尚未安排后续自爆虫的目标续接前后协同。
            if not self._first_was_launched(observation, engagement):
                continue
            if engagement.second is not None:
                continue

            assessment = self._assess_threat(observation, target)
            if assessment is None:
                continue
            # 后续自爆虫需与首个自爆虫形成足够的接近目标耗时差和正向前后间距。
            for item in self._target_candidates(
                observation,
                target,
                available,
            ):
                if (
                    item.assignment.interceptor_id
                    == engagement.first.interceptor_id
                ):
                    continue
                intercept_gap = (
                    item.assignment.estimated_intercept_step
                    - engagement.first.estimated_intercept_step
                )
                if intercept_gap < observation.clock.seconds_to_steps(self.MIN_INTERCEPT_GAP_SECONDS):
                    continue
                depth_separation = self._depth_separation(
                    engagement.first,
                    item.assignment,
                    target.vel,
                )
                if depth_separation <= 0.0:
                    continue
                stage_candidates.append(
                    _StageCandidate(
                        candidate=item,
                        role="second",
                        coordination_id=engagement.coordination_id,
                        objective_cost=(
                            item.cost
                            - assessment.protection_value
                            * self.SECOND_SHOT_VALUE
                        ),
                        intercept_gap_step=intercept_gap,
                        depth_separation_m=depth_separation,
                    )
                )

        if not stage_candidates:
            self._set_stats(
                "no_candidates",
                started_at,
                observation,
                [],
                [],
            )
            return []

        # 以二元变量选择候选，联合最小化成本并满足目标与自爆虫互斥约束。
        result = milp(
            c=np.array([
                item.objective_cost
                for item in stage_candidates
            ]),
            integrality=np.ones(len(stage_candidates)),
            bounds=Bounds(0.0, 1.0),
            constraints=self._constraints(stage_candidates),
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
                stage_candidates,
                [],
                optimal=False,
                mip_gap=float("inf"),
            )
            return []

        # 将求解选中的首个或后续指派写回追踪状态，并附加协同标识。
        selected = [
            item
            for index, item in enumerate(stage_candidates)
            if result.x[index] > 0.5
        ]
        assignments = []
        for item in selected:
            assignment = replace(
                item.candidate.assignment,
                coordination_id=item.coordination_id,
                coordination_mode="staggered",
                spatial_mode="layered",
            )
            target_id = assignment.target_id
            if item.role == "first":
                self._engagements[target_id] = _Engagement(
                    coordination_id=item.coordination_id,
                    first=assignment,
                )
            else:
                self._engagements[target_id].second = assignment
            assignments.append(assignment)

        # 汇总本轮求解与协同指标，按释放时间稳定输出计划。
        self._set_stats(
            "optimal" if result.status == 0 else "feasible",
            started_at,
            observation,
            stage_candidates,
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

    def _synchronize_engagements(
        self,
        observation: BlueObservation,
    ) -> None:
        # 时间回退表示进入新回合；当前信息决定哪些目标计划继续保留。
        if observation.step < self._last_step:
            self._engagements.clear()
        self._last_step = observation.step
        known_target_ids = {
            target.target_id
            for target in observation.targets
        }
        for target_id in list(self._engagements):
            if target_id not in known_target_ids:
                del self._engagements[target_id]
                continue
            engagement = self._engagements[target_id]
            # 到达计划时刻后核对释放回执，清除未执行的首个或后续指派。
            launched_ids = set(
                observation.launched_map.get(target_id, [])
            )
            if (
                observation.step
                >= engagement.first.planned_launch_step
                and engagement.first.interceptor_id not in launched_ids
            ):
                del self._engagements[target_id]
                continue
            if (
                engagement.second is not None
                and observation.step
                >= engagement.second.planned_launch_step
                and engagement.second.interceptor_id not in launched_ids
            ):
                engagement.second = None

    @staticmethod
    def _first_was_launched(
        observation: BlueObservation,
        engagement: _Engagement,
    ) -> bool:
        # 首个计划时刻已到且系统确认对应自爆虫出巢，才允许安排后续指派。
        return (
            observation.step
            >= engagement.first.planned_launch_step
            and engagement.first.interceptor_id
            in observation.launched_map.get(
                engagement.first.target_id,
                [],
            )
        )

    def _target_candidates(
        self,
        observation,
        target,
        available,
    ) -> list[_Candidate]:
        # 组合目标与可用自爆虫的可行时隙，再按释放时隙压缩候选。
        candidates = [
            candidate
            for interceptor in available
            for candidate in self._candidates(
                observation,
                target,
                interceptor,
            )
        ]
        return self._reduce_candidates(candidates)

    def _reduce_candidates(
        self,
        candidates: list[_Candidate],
    ) -> list[_Candidate]:
        # 按释放时刻分组，每个时隙保留成本最低的若干个不同自爆虫。
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
                key=lambda item: (
                    item.cost,
                    item.assignment.interceptor_id,
                ),
            ):
                interceptor_id = candidate.assignment.interceptor_id
                if interceptor_id in used_interceptors:
                    continue
                reduced.append(candidate)
                used_interceptors.add(interceptor_id)
                if (
                    len(used_interceptors)
                    >= self.CANDIDATES_PER_LAUNCH_SLOT
                ):
                    break
        return reduced

    @classmethod
    def _depth_separation(
        cls,
        first: ScheduledInterceptorAssignment,
        second: ScheduledInterceptorAssignment,
        target_velocity: Vector3d,
    ) -> float:
        # 将两次预测相遇点的位移投影到目标速度方向，得到前后间距。
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
    def _constraints(
        candidates: list[_StageCandidate],
    ) -> LinearConstraint:
        # 每个目标和每个自爆虫在本轮最多选择一个候选。
        groups: dict[tuple[str, int], list[int]] = {}
        for index, item in enumerate(candidates):
            assignment = item.candidate.assignment
            groups.setdefault(
                ("target", assignment.target_id),
                [],
            ).append(index)
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
        candidates: list[_StageCandidate],
        selected: list[_StageCandidate],
        *,
        optimal: bool = True,
        mip_gap: float = 0.0,
    ) -> None:
        # 用本轮选中的后续指派统计成对协同及时间、间距收益。
        selected_seconds = [
            item
            for item in selected
            if item.role == "second"
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
            "selected_first_shot_count": sum(
                item.role == "first"
                for item in selected
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
                item.intercept_gap_step
                for item in selected_seconds
            ),
            "intercept_separation_m_total": sum(
                item.depth_separation_m
                for item in selected_seconds
            ),
            "approach_angle_deg_total": 0.0,
            "asset_clearance_m_total": 0.0,
            "coordination_mode": "successive_depth",
        }


class BlueDefense(BlueDefenseController):
    """B5 前后协同策略的公共入口。"""

    def __init__(
        self,
        *,
        seed: int | None = None,
        asset_values: Mapping[int, float] | None = None,
    ):
        # 统一入口参数交给 B5 配置，其余参数沿用该策略默认值。
        super().__init__(BlueDefenseConfig(
            policy_name="b5_successive_depth_coordination",
            seed=seed,
            asset_values=dict(asset_values) if asset_values is not None else {},
        ))
