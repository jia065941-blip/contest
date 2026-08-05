from __future__ import annotations

from ..domain import Assignment, Decision, GlobalRules, Observation, Platform, Target
from ..geometry import distance_km
from ..optimization import AssignmentModel
from .base import BaselinePolicy


class B2StaticAssignmentPolicy(BaselinePolicy):
    def __init__(self, rules: GlobalRules, candidate_limit: int = 6, exact_platform_limit: int = 12) -> None:
        self.rules = rules
        self.model = AssignmentModel(rules.target_capacity, candidate_limit, exact_platform_limit)
        self._plan: Decision | None = None

    def pair_score(self, platform: Platform, target: Target) -> float:
        normalized_distance = distance_km(platform.position, target.position) / 100.0
        return self.rules.value_weight * target.value - self.rules.distance_weight * normalized_distance - self.rules.cost_by_kind.get(platform.kind, 0.0)

    def decide(self, observation: Observation) -> Decision:
        if self._plan is not None:
            active_ids = {platform.entity_id for platform in observation.active_platforms()}
            target_ids = {target.entity_id for target in observation.active_targets()}
            assignments = tuple(item for item in self._plan.assignments if item.platform_id in active_ids and item.target_id in target_ids)
            return Decision(assignments=assignments)
        platforms = observation.active_platforms()
        targets = observation.active_targets()
        # Allocate the complete available force across targets in proportion to
        # declared target value.  This is an optimization capacity, not a
        # global fire-budget cap.
        capacities = self._target_capacities(platforms, targets)
        self.model = AssignmentModel(capacities, self.model.candidate_limit, self.model.exact_platform_limit)
        solution = self.model.solve(platforms, targets, self.pair_score)
        platform_kinds = {platform.entity_id: platform.kind for platform in platforms}
        pairs = sorted(solution.pairs, key=lambda item: (self.rules.wave_by_kind.get(platform_kinds[item[0]], 0), item[0]))
        assignments = tuple(
            Assignment(platform_id, target_id, observation.step + self.rules.wave_by_kind.get(platform_kinds[platform_id], 0) + index * self.rules.min_launch_interval, score)
            for index, (platform_id, target_id, score) in enumerate(pairs)
        )
        self._plan = Decision(assignments=assignments)
        return self._plan

    @staticmethod
    def _target_capacities(platforms: tuple[Platform, ...], targets: tuple[Target, ...]) -> dict[int, int]:
        if not platforms or not targets:
            return {}
        total_value = sum(max(0.0, target.value) for target in targets)
        if total_value <= 0:
            total_value = float(len(targets))
        raw = {target.entity_id: len(platforms) * max(0.0, target.value) / total_value for target in targets}
        capacities = {target_id: int(value) for target_id, value in raw.items()}
        remainder = len(platforms) - sum(capacities.values())
        for target_id, _ in sorted(raw.items(), key=lambda item: (-(item[1] - int(item[1])), item[0]))[:remainder]:
            capacities[target_id] += 1
        return capacities
