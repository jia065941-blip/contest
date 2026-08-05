"""Metrics shared by all baseline runners."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..domain import Decision


@dataclass
class EpisodeMetrics:
    initial_target_value: float
    destroyed_target_value: float = 0.0
    breakout_count: int = 0
    loss_count: int = 0
    satellite_count: int = 0
    assigned_targets: set[int] = field(default_factory=set)
    actions: int = 0
    constraint_violations: list[str] = field(default_factory=list)

    def record_decision(self, decision: Decision, target_capacity: int) -> None:
        target_counts: dict[int, int] = {}
        for assignment in decision.assignments:
            target_counts[assignment.target_id] = target_counts.get(assignment.target_id, 0) + 1
            self.assigned_targets.add(assignment.target_id)
        self.actions += len(decision.assignments)
        self.satellite_count += len(decision.satellite_users)
        for target_id, count in target_counts.items():
            if count > target_capacity:
                self.constraint_violations.append(f"target {target_id} has {count} assignments")

    def summary(self) -> dict[str, float | int]:
        completion = self.destroyed_target_value / self.initial_target_value if self.initial_target_value else 0.0
        return {
            "weighted_damage": self.destroyed_target_value,
            "completion_rate": completion,
            "breakout_count": self.breakout_count,
            "loss_count": self.loss_count,
            "satellite_count": self.satellite_count,
            "target_coverage": len(self.assigned_targets),
            "action_count": self.actions,
            "constraint_violations": len(self.constraint_violations),
        }

