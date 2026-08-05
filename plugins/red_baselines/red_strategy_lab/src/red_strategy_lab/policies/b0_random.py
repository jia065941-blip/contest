from __future__ import annotations

import random

from ..domain import Assignment, Decision, GlobalRules, Observation
from .base import BaselinePolicy


class B0RandomPolicy(BaselinePolicy):
    def __init__(self, rules: GlobalRules, seed: int) -> None:
        self.rules = rules
        self.seed = seed

    def decide(self, observation: Observation) -> Decision:
        targets = sorted(observation.active_targets(), key=lambda item: item.entity_id)
        assignments: list[Assignment] = []
        counts: dict[int, int] = {}
        for offset, platform in enumerate(sorted(observation.active_platforms(), key=lambda item: item.entity_id)):
            candidates = [target for target in targets if counts.get(target.entity_id, 0) < self.rules.target_capacity]
            if not candidates:
                break
            generator = random.Random(self.seed + observation.step * 1009 + platform.entity_id)
            target = candidates[generator.randrange(len(candidates))]
            launch_step = observation.step + self.rules.wave_by_kind.get(platform.kind, 0) + offset * self.rules.min_launch_interval
            assignments.append(Assignment(platform.entity_id, target.entity_id, launch_step, 0.0))
            counts[target.entity_id] = counts.get(target.entity_id, 0) + 1
        return Decision(assignments=tuple(assignments))

