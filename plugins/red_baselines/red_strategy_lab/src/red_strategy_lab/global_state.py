"""Small deterministic state board shared by global-rule policies."""

from __future__ import annotations

from collections import Counter

from .domain import Assignment, GlobalRules


class GlobalStateBoard:
    def __init__(self, rules: GlobalRules) -> None:
        self.rules = rules
        self.assignments: dict[int, Assignment] = {}
        self.target_counts: Counter[int] = Counter()
        self.satellite_used = 0
        self.last_launch_step: int | None = None

    def reset(self) -> None:
        self.assignments.clear()
        self.target_counts.clear()
        self.satellite_used = 0
        self.last_launch_step = None

    def can_assign(self, platform_id: int, target_id: int, step: int) -> bool:
        if platform_id in self.assignments or self.target_counts[target_id] >= self.rules.target_capacity:
            return False
        return self.last_launch_step is None or step - self.last_launch_step >= self.rules.min_launch_interval

    def assign(self, assignment: Assignment) -> None:
        if not self.can_assign(assignment.platform_id, assignment.target_id, assignment.launch_step):
            raise ValueError("assignment violates global rules")
        self.assignments[assignment.platform_id] = assignment
        self.target_counts[assignment.target_id] += 1
        self.last_launch_step = assignment.launch_step

    def can_use_satellite(self) -> bool:
        return self.satellite_used < self.rules.satellite_budget

    def use_satellite(self) -> None:
        if not self.can_use_satellite():
            raise ValueError("satellite budget exhausted")
        self.satellite_used += 1

