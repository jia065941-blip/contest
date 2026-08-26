"""Core-facing adapter for R0--R3 without blue-side global observations."""

from __future__ import annotations

import math

from .baselines import (
    BaselineObservation,
    BaselineRules,
    PlatformState,
    RED_POLICY_CHOICES,
    TargetPrior,
    build_red_baseline,
)
from .contracts import Position


_KIND_BY_TYPE = {21000: "H", 21001: "M", 21002: "L"}


class RedBaselineCommander:
    """Coordinate launch allocation from own-platform reports and core priors."""

    def __init__(self, targets: tuple[TargetPrior, ...], policy_name: str, seed: int = 1) -> None:
        if policy_name not in RED_POLICY_CHOICES:
            raise ValueError(f"Unsupported red baseline '{policy_name}'")
        self.targets = targets
        self.policy_name = policy_name
        self.seed = seed
        self.rules = BaselineRules()
        self.policy = build_red_baseline(policy_name, self.rules, seed)
        self.reports: dict[int, PlatformState] = {}
        self.expected_platform_ids: set[int] = set()
        self.launched_ids: set[int] = set()
        self.assigned_by_target: dict[int, int] = {}
        self.pending: dict[int, list[tuple[int, float, float]]] = {}
        self.last_plan_step = -1

    def register_platform(self, entity_id: int) -> None:
        self.expected_platform_ids.add(int(entity_id))

    def report(self, observation: dict) -> None:
        own = observation.get("self", {})
        entity_id = int(observation.get("entity_id", -1))
        kind = _KIND_BY_TYPE.get(int(own.get("type", 0)))
        position = own.get("position", {})
        if entity_id < 0 or kind is None or not isinstance(position, dict):
            return
        self.reports[entity_id] = PlatformState(
            entity_id=entity_id,
            kind=kind,
            position=Position(
                lon=float(position.get("lon", 0.0)),
                lat=float(position.get("lat", 0.0)),
                alt=float(position.get("alt", 0.0)),
            ),
            alive=float(own.get("health", 0)) > 0 and bool(own.get("isVisible", True)),
            launched=entity_id in self.launched_ids,
        )

    def action_for(self, entity_id: int, step: int) -> list[float] | None:
        if self._should_plan(step):
            self._plan(step)
        rows = self.pending.get(int(entity_id), [])
        due = [row for row in rows if row[0] <= step]
        future = [row for row in rows if row[0] > step]
        if future:
            self.pending[int(entity_id)] = future
        else:
            self.pending.pop(int(entity_id), None)
        if not due:
            return None
        _, lon, lat = due[0]
        self.launched_ids.add(int(entity_id))
        return [1.0, float(entity_id), lon, lat]

    def _should_plan(self, step: int) -> bool:
        if not self.targets or len(self.reports) < len(self.expected_platform_ids):
            return False
        return self.last_plan_step < 0 or step - self.last_plan_step >= self.rules.replan_interval

    def _plan(self, step: int) -> None:
        platforms = tuple(
            PlatformState(item.entity_id, item.kind, item.position, item.alive, item.entity_id in self.launched_ids)
            for item in self.reports.values()
        )
        decision = self.policy.decide(BaselineObservation(step=step, platforms=platforms, targets=self.targets))
        target_by_id = {item.entity_id: item for item in self.targets}
        rolling_limit = math.ceil(sum(item.alive and not item.launched for item in platforms) * 0.25)
        accepted = 0
        for assignment in decision:
            if self.policy_name == "r3_rolling_rules" and accepted >= rolling_limit:
                break
            if assignment.platform_id in self.launched_ids or assignment.platform_id in self.pending:
                continue
            if self.assigned_by_target.get(assignment.target_id, 0) >= self.rules.target_capacity:
                continue
            target = target_by_id.get(assignment.target_id)
            if target is None:
                continue
            self.pending.setdefault(assignment.platform_id, []).append(
                (assignment.launch_step, target.position.lon, target.position.lat)
            )
            self.assigned_by_target[assignment.target_id] = self.assigned_by_target.get(assignment.target_id, 0) + 1
            accepted += 1
        self.last_plan_step = step

    def reset(self) -> None:
        self.reports.clear()
        self.launched_ids.clear()
        self.assigned_by_target.clear()
        self.pending.clear()
        self.last_plan_step = -1
        self.policy = build_red_baseline(self.policy_name, self.rules, self.seed)
