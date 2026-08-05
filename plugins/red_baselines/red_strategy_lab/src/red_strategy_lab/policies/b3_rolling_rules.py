from __future__ import annotations

from ..domain import Decision, GlobalRules, Observation
from .b1_priority import B1PriorityPolicy
from .base import BaselinePolicy


class B3RollingRulePolicy(BaselinePolicy):
    def __init__(self, rules: GlobalRules) -> None:
        self.rules = rules
        self._last_replan_step: int | None = None
        self._decision = Decision()

    def decide(self, observation: Observation) -> Decision:
        if self._last_replan_step is None or observation.step - self._last_replan_step >= self.rules.replan_interval:
            self._decision = B1PriorityPolicy(self.rules).decide(observation)
            self._last_replan_step = observation.step
        active_platforms = {platform.entity_id for platform in observation.active_platforms()}
        active_targets = {target.entity_id for target in observation.active_targets()}
        return Decision(assignments=tuple(item for item in self._decision.assignments if item.platform_id in active_platforms and item.target_id in active_targets))

