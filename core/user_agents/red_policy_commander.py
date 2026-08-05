"""Red-side command post built from local platform reports.

The commander is deliberately not given ``TrainingEnv._get_observation``.  Each
platform reports only its own state and its received detection/communication
information.  A static target catalogue is treated as pre-mission intelligence,
not as a live blue-side observation.
"""
from __future__ import annotations

import os
import math
from dataclasses import dataclass

from red_strategy_lab.domain import GlobalRules, Observation, Platform, Position, Target
from red_strategy_lab.policies import B0RandomPolicy, B1PriorityPolicy, B2StaticAssignmentPolicy, B3RollingRulePolicy


_KIND_BY_TYPE = {21000: "H", 21001: "M", 21002: "L"}


@dataclass
class _Report:
    step: int
    entity_id: int
    kind: str
    position: Position
    alive: bool
    detections: dict


class RedPolicyCommander:
    """Persist planned launches and fuse reports received from red platforms."""

    def __init__(self, target_catalogue: tuple[Target, ...]) -> None:
        self.targets = target_catalogue
        self.detected_positions: dict[int, Position] = {}
        self.policy_name = os.getenv("RED_POLICY", "b0_random")
        self.rules = self._rules_for(self.policy_name)
        self.policy = self._build_policy()
        self.reports: dict[int, _Report] = {}
        self.expected_platform_ids: set[int] = set()
        self.launched_ids: set[int] = set()
        self.assigned_by_target: dict[int, int] = {}
        self.pending: dict[int, list[tuple[int, float, float]]] = {}
        self.last_plan_step = -1
        self.dispatched_count = 0

    def register_platform(self, entity_id: int) -> None:
        self.expected_platform_ids.add(int(entity_id))

    def _rules_for(self, policy_name: str) -> GlobalRules:
        """Keep each baseline's budget and timing assumptions explicit."""
        common = dict(
            wave_by_kind={"H": 0, "M": 10, "L": 20},
            value_weight=1.0,
            distance_weight=0.25,
            coverage_weight=0.75,
        )
        configs = {
            # B0--B2 schedule every available platform.  Their distinction is
            # target allocation, not an artificial total-fire budget.
            "b0_random": dict(target_capacity=10_000, min_launch_interval=0, replan_interval=20),
            "b1_priority": dict(target_capacity=10_000, min_launch_interval=0, replan_interval=20, cost_by_kind={"H": 0.8, "M": 0.3, "L": 0.1}),
            "b2_static_assignment": dict(target_capacity=10_000, min_launch_interval=0, replan_interval=120, cost_by_kind={"H": 0.8, "M": 0.3, "L": 0.1}),
            # B3 releases every platform eventually, but divides them into
            # locally informed waves rather than firing the whole force at once.
            "b3_rolling_rules": dict(target_capacity=10_000, min_launch_interval=0, replan_interval=20, cost_by_kind={"H": 0.8, "M": 0.3, "L": 0.1}),
        }
        return GlobalRules(**common, **configs[policy_name])

    def _build_policy(self):
        if self.policy_name == "b0_random":
            return B0RandomPolicy(self.rules, int(os.getenv("RED_POLICY_SEED", "1")))
        if self.policy_name == "b1_priority":
            return B1PriorityPolicy(self.rules)
        if self.policy_name == "b2_static_assignment":
            return B2StaticAssignmentPolicy(self.rules, candidate_limit=max(1, len(self.targets)))
        return B3RollingRulePolicy(self.rules)

    def report(self, observation: dict) -> None:
        """Accept exactly one platform's isolated observation."""
        own = observation.get("self", {})
        entity_id = int(observation.get("entity_id", -1))
        entity_type = int(own.get("type", 0))
        kind = _KIND_BY_TYPE.get(entity_type)
        position = own.get("position", {})
        if entity_id < 0 or kind is None or not isinstance(position, dict):
            return
        self.reports[entity_id] = _Report(
            step=int(observation.get("step", 0)),
            entity_id=entity_id,
            kind=kind,
            position=Position(float(position.get("lon", 0)), float(position.get("lat", 0)), float(position.get("alt", 0))),
            alive=float(own.get("health", 0)) > 0 and bool(own.get("isVisible", True)),
            detections=dict(own.get("detectInfo", {}) or {}),
        )
        # Only tracks that a platform has actually received may refine the
        # pre-mission target catalogue.  No blue-side live entity lookup occurs.
        target_ids = {target.entity_id for target in self.targets}
        for track in self.reports[entity_id].detections.values():
            target_id = self._field(track, "entity_id")
            lla = self._field(track, "lla")
            if target_id is None or int(target_id) not in target_ids or lla is None:
                continue
            self.detected_positions[int(target_id)] = Position(
                float(self._field(lla, "x", 0.0)),
                float(self._field(lla, "y", 0.0)),
                float(self._field(lla, "z", 0.0)),
            )

    def action_for(self, entity_id: int, step: int) -> list[float] | None:
        # Replan at the policy's cadence.  Reports from the preceding step are
        # retained, so the first callback of a frame never needs global state.
        if self._should_plan(step):
            self._plan(step)
        planned = self.pending.get(int(entity_id), [])
        due = [item for item in planned if item[0] <= step]
        future = [item for item in planned if item[0] > step]
        if future:
            self.pending[int(entity_id)] = future
        else:
            self.pending.pop(int(entity_id), None)
        if not due:
            return None
        _, lon, lat = due[0]
        self.launched_ids.add(int(entity_id))
        self.dispatched_count += 1
        return [1.0, float(entity_id), lon, lat]

    def _should_plan(self, step: int) -> bool:
        if not self.reports or len(self.reports) < len(self.expected_platform_ids):
            return False
        if self.last_plan_step < 0:
            return True
        return step - self.last_plan_step >= self.rules.replan_interval

    def _plan(self, step: int) -> None:
        platforms = tuple(
            Platform(item.entity_id, item.kind, item.position, alive=item.alive, launched=item.entity_id in self.launched_ids)
            for item in self.reports.values()
        )
        targets = tuple(
            Target(item.entity_id, self.detected_positions.get(item.entity_id, item.position), item.value, item.health, item.alive)
            for item in self.targets
        )
        decision = self.policy.decide(Observation(step=step, platforms=platforms, targets=targets))
        target_map = {item.entity_id: item for item in targets}
        rolling_limit = math.ceil(sum(item.alive and not item.launched for item in platforms) * 0.25)
        accepted = 0
        for assignment in decision.assignments:
            if self.policy_name == "b3_rolling_rules" and accepted >= rolling_limit:
                break
            if assignment.platform_id in self.launched_ids or assignment.platform_id in self.pending:
                continue
            if self.assigned_by_target.get(assignment.target_id, 0) >= self.rules.target_capacity:
                continue
            target = target_map.get(assignment.target_id)
            if target is not None:
                self.pending.setdefault(assignment.platform_id, []).append(
                    (assignment.launch_step, target.position.lon, target.position.lat)
                )
                self.assigned_by_target[assignment.target_id] = self.assigned_by_target.get(assignment.target_id, 0) + 1
                accepted += 1
        self.last_plan_step = step

    def reset(self) -> None:
        self.reports.clear()
        self.detected_positions.clear()
        self.launched_ids.clear()
        self.assigned_by_target.clear()
        self.pending.clear()
        self.last_plan_step = -1
        self.dispatched_count = 0
        self.policy = self._build_policy()

    @staticmethod
    def _field(value, name: str, default=None):
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)
