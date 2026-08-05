from __future__ import annotations

from ..domain import Assignment, Decision, GlobalRules, Observation, Platform, Target
from ..geometry import distance_km
from .base import BaselinePolicy


class B1PriorityPolicy(BaselinePolicy):
    def __init__(self, rules: GlobalRules) -> None:
        self.rules = rules

    def score(self, platform: Platform, target: Target, assigned_count: int) -> float:
        distance = distance_km(platform.position, target.position) / 100.0
        cost = self.rules.cost_by_kind.get(platform.kind, 0.0)
        return self.rules.value_weight * target.value - self.rules.distance_weight * distance - self.rules.coverage_weight * assigned_count - cost

    def decide(self, observation: Observation) -> Decision:
        targets = sorted(observation.active_targets(), key=lambda item: item.entity_id)
        counts: dict[int, int] = {}
        assignments: list[Assignment] = []
        platforms = sorted(observation.active_platforms(), key=lambda item: (self.rules.wave_by_kind.get(item.kind, 0), item.entity_id))
        for index, platform in enumerate(platforms):
            options = [(self.score(platform, target, counts.get(target.entity_id, 0)), target) for target in targets if counts.get(target.entity_id, 0) < self.rules.target_capacity]
            if not options:
                break
            score, target = max(options, key=lambda item: (item[0], -item[1].entity_id))
            launch_step = observation.step + self.rules.wave_by_kind.get(platform.kind, 0) + index * self.rules.min_launch_interval
            assignments.append(Assignment(platform.entity_id, target.entity_id, launch_step, score))
            counts[target.entity_id] = counts.get(target.entity_id, 0) + 1
        return Decision(assignments=tuple(assignments))
