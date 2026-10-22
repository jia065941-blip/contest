"""Pure three-missile coordination rules with V13 actions as the baseline."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
import math

from skill_generation.skills import ManeuverAction
from stop_turning_skill import StopTurningSkill
from turn_left_skill import TurnLeftSkill
from turn_right_skill import TurnRightSkill


@dataclass(frozen=True)
class Vector2:
    east: float
    north: float


@dataclass(frozen=True)
class JointThreat:
    threat_id: int
    position: Vector2
    velocity: Vector2
    age_seconds: float = 0.0


@dataclass(frozen=True)
class JointMemberSnapshot:
    entity_id: int
    missile_type: int
    target_id: int
    position: Vector2
    velocity: Vector2
    goal_position: Vector2
    threats: tuple[JointThreat, ...]
    baseline_action: int
    launch_step: int = 0
    alive: bool = True


class JointRole(str, Enum):
    DECOY = "decoy"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    BACKUP = "backup"
    LOST = "lost"


@dataclass(frozen=True)
class JointMissionContext:
    target_expected_hp: float | None = None
    target_confidence: float = 0.0
    target_confirmed_destroyed: bool = False
    authorized_fallback_target_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class RetargetRequest:
    from_target_id: int
    to_target_id: int
    member_ids: tuple[int, ...]
    reason: str
    confidence: float


@dataclass(frozen=True)
class StrikePackageStatus:
    target_id: int
    roles: tuple[tuple[int, JointRole], ...]
    live_member_ids: tuple[int, ...]
    threat_owner_count: int
    immediate_danger_count: int
    eta_spread_seconds: float | None
    preferred_actions: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ThreeMissileJointConfig:
    min_members: int = 2
    max_members: int = 6
    ownership_t_cpa_min_seconds: float = 1.0
    ownership_t_cpa_max_seconds: float = 25.0
    ownership_d_cpa_metres: float = 750.0
    ownership_range_metres: float = 9000.0
    ownership_closing_mps: float = 100.0
    ownership_ratio: float = 0.90
    ownership_margin: float = 250.0
    ownership_confirm_steps: int = 2
    ownership_hold_steps: int = 3
    immediate_t_cpa_seconds: float = 12.0
    immediate_d_cpa_metres: float = 500.0
    immediate_range_metres: float = 5000.0
    peer_horizon_seconds: float = 3.0
    peer_trigger_metres: float = 150.0
    peer_release_metres: float = 200.0
    peer_closing_mps: float = 20.0
    minimum_separation_improvement_metres: float = 150.0
    override_cooldown_steps: int = 3
    override_window_steps: int = 10
    maximum_overrides_per_window: int = 2
    lateral_acceleration_mps2: float = 196.0
    eta_shadow_enabled: bool = True
    eta_actuation_enabled: bool = False
    approach_angle_actuation_enabled: bool = False
    retarget_confidence_threshold: float = 0.75
    require_peer_closing: bool = False
    protect_primary: bool = False
    allow_direct_reversal: bool = True
    stop_to_turn_trigger_metres: float = 150.0
    maximum_added_goal_error_degrees: float | None = None
    maximum_eta_penalty_seconds: float | None = None
    action_hold_seconds: float = 1.0

    @classmethod
    def experimental(cls) -> "ThreeMissileJointConfig":
        return cls(
            peer_horizon_seconds=6.0,
            peer_trigger_metres=1000.0,
            peer_release_metres=400.0,
            minimum_separation_improvement_metres=100.0,
            override_cooldown_steps=3,
            override_window_steps=10,
            maximum_overrides_per_window=2,
        )

    @classmethod
    def mission_aware(cls) -> "ThreeMissileJointConfig":
        return cls(
            peer_horizon_seconds=4.0,
            peer_trigger_metres=750.0,
            peer_release_metres=400.0,
            minimum_separation_improvement_metres=100.0,
            override_cooldown_steps=3,
            override_window_steps=10,
            maximum_overrides_per_window=2,
            require_peer_closing=True,
            protect_primary=True,
            allow_direct_reversal=False,
            stop_to_turn_trigger_metres=400.0,
            maximum_added_goal_error_degrees=3.0,
            maximum_eta_penalty_seconds=1.5,
        )


@dataclass(frozen=True)
class JointDecision:
    actions: tuple[int, ...]
    baseline_actions: tuple[int, ...]
    overridden_entity_id: int | None
    reason: str
    threat_owners: tuple[tuple[int, int], ...]
    baseline_min_separation_metres: float
    selected_min_separation_metres: float
    eta_spread_seconds: float | None
    status: StrikePackageStatus
    retarget_request: RetargetRequest | None


class HStrikePackageSkill:
    """Coordinate a fixed package of 2-6 same-type, same-target H missiles.

    V13 remains authoritative. The initial version changes at most one action
    only for a hard predicted peer collision; ownership and ETA are advisory.
    """

    def __init__(self, config: ThreeMissileJointConfig | None = None) -> None:
        self.config = config or ThreeMissileJointConfig()
        self._owner_candidates: dict[int, tuple[int, int]] = {}
        self._owners: dict[int, tuple[int, int]] = {}
        self._override_steps: deque[int] = deque()
        self._last_override_step = -10_000
        self._children = {
            0: TurnLeftSkill(),
            1: StopTurningSkill(),
            2: TurnRightSkill(),
        }

    def reset(self) -> None:
        self._owner_candidates.clear()
        self._owners.clear()
        self._override_steps.clear()
        self._last_override_step = -10_000
        for child in self._children.values():
            child.reset()

    def decide(
        self,
        step: int,
        members: tuple[JointMemberSnapshot, ...],
        mission_context: JointMissionContext | None = None,
    ) -> JointDecision:
        ordered = self._validate(members)
        roles = self._assign_roles(ordered)
        baselines = tuple(item.baseline_action for item in ordered)
        owners, immediate = self._threat_ownership(step, ordered)
        baseline_separation = self._group_minimum_separation(ordered, baselines)
        selected = list(baselines)
        overridden: int | None = None
        reason = "v13_passthrough"

        self._trim_budget(step)
        pair = self._closest_peer_pair(ordered, baselines)
        peer_closing = self._peer_closing_speed(ordered, pair)
        can_override = (
            baseline_separation < self.config.peer_trigger_metres
            and (
                not self.config.require_peer_closing
                or peer_closing >= self.config.peer_closing_mps
            )
            and step - self._last_override_step >= self.config.override_cooldown_steps
            and len(self._override_steps) < self.config.maximum_overrides_per_window
        )
        if can_override:
            candidate = self._collision_candidate(
                ordered, pair, owners, immediate, roles
            )
            if candidate is not None:
                index = next(i for i, item in enumerate(ordered) if item.entity_id == candidate)
                best_action = baselines[index]
                best_separation = baseline_separation
                for action in self._candidate_actions(
                    baselines[index], baseline_separation
                ):
                    trial = list(baselines)
                    trial[index] = action
                    separation = self._group_minimum_separation(ordered, tuple(trial))
                    if (
                        separation > best_separation
                        and self._preserves_mission(
                            ordered[index], baselines[index], action
                        )
                    ):
                        best_action = action
                        best_separation = separation
                if (
                    best_action != baselines[index]
                    and best_separation - baseline_separation
                    >= self.config.minimum_separation_improvement_metres
                    and best_separation >= self.config.peer_release_metres
                ):
                    selected[index] = best_action
                    overridden = candidate
                    reason = "hard_peer_deconfliction"
                    self._last_override_step = step
                    self._override_steps.append(step)

        materialized = tuple(self._materialize(action) for action in selected)
        selected_separation = self._group_minimum_separation(ordered, materialized)
        eta_spread = self._eta_spread(ordered)
        retarget_request = self._retarget_request(
            ordered, mission_context
        )
        preferred_actions = tuple(
            (item.entity_id, self._preferred_role_action(roles[item.entity_id]))
            for item in ordered
            if item.alive
        )
        status = StrikePackageStatus(
            target_id=ordered[0].target_id,
            roles=tuple((item.entity_id, roles[item.entity_id]) for item in ordered),
            live_member_ids=tuple(item.entity_id for item in ordered if item.alive),
            threat_owner_count=len(owners),
            immediate_danger_count=len(immediate),
            eta_spread_seconds=eta_spread,
            preferred_actions=preferred_actions,
        )
        return JointDecision(
            actions=materialized,
            baseline_actions=baselines,
            overridden_entity_id=overridden,
            reason=reason,
            threat_owners=tuple(sorted((threat, owner) for threat, (owner, _until) in owners.items())),
            baseline_min_separation_metres=baseline_separation,
            selected_min_separation_metres=selected_separation,
            eta_spread_seconds=eta_spread,
            status=status,
            retarget_request=retarget_request,
        )

    @staticmethod
    def _assign_roles(
        members: tuple[JointMemberSnapshot, ...]
    ) -> dict[int, JointRole]:
        roles = {item.entity_id: JointRole.LOST for item in members}
        live = sorted(
            (item for item in members if item.alive),
            key=lambda item: (item.launch_step, item.entity_id),
        )
        if len(live) == 1:
            roles[live[0].entity_id] = JointRole.PRIMARY
        elif len(live) == 2:
            roles[live[0].entity_id] = JointRole.PRIMARY
            roles[live[1].entity_id] = JointRole.BACKUP
        elif len(live) == 3:
            roles[live[0].entity_id] = JointRole.DECOY
            roles[live[1].entity_id] = JointRole.PRIMARY
            roles[live[2].entity_id] = JointRole.BACKUP
        elif len(live) >= 4:
            roles[live[0].entity_id] = JointRole.DECOY
            roles[live[1].entity_id] = JointRole.PRIMARY
            for item in live[2:-1]:
                roles[item.entity_id] = JointRole.SECONDARY
            roles[live[-1].entity_id] = JointRole.BACKUP
        return roles

    def _retarget_request(
        self,
        members: tuple[JointMemberSnapshot, ...],
        context: JointMissionContext | None,
    ) -> RetargetRequest | None:
        if context is None or not context.authorized_fallback_target_ids:
            return None
        reason = None
        confidence = context.target_confidence
        if context.target_confirmed_destroyed:
            reason = "target_confirmed_destroyed"
            confidence = 1.0
        elif (
            context.target_expected_hp is not None
            and context.target_expected_hp <= 0.0
            and context.target_confidence >= self.config.retarget_confidence_threshold
        ):
            reason = "target_expected_destroyed"
        if reason is None:
            return None
        live = tuple(item.entity_id for item in members if item.alive)
        if not live:
            return None
        return RetargetRequest(
            from_target_id=members[0].target_id,
            to_target_id=int(context.authorized_fallback_target_ids[0]),
            member_ids=live,
            reason=reason,
            confidence=float(confidence),
        )

    @staticmethod
    def _preferred_role_action(role: JointRole) -> int:
        if role == JointRole.DECOY:
            return int(ManeuverAction.LEFT)
        if role == JointRole.BACKUP:
            return int(ManeuverAction.RIGHT)
        return int(ManeuverAction.STOP)

    def _validate(
        self, members: tuple[JointMemberSnapshot, ...]
    ) -> tuple[JointMemberSnapshot, ...]:
        if not self.config.min_members <= len(members) <= self.config.max_members:
            raise ValueError(
                f"strike package requires {self.config.min_members}-"
                f"{self.config.max_members} missiles"
            )
        ordered = tuple(sorted(members, key=lambda item: item.entity_id))
        if len({item.entity_id for item in ordered}) != len(ordered):
            raise ValueError("joint missile IDs must be unique")
        if len({item.missile_type for item in ordered}) != 1:
            raise ValueError("joint missiles must have the same type")
        if len({item.target_id for item in ordered}) != 1:
            raise ValueError("joint missiles must have the same target ID")
        if any(item.baseline_action not in (0, 1, 2) for item in ordered):
            raise ValueError("V13 baseline actions must be 0, 1, or 2")
        return ordered

    def _threat_ownership(
        self,
        step: int,
        members: tuple[JointMemberSnapshot, ...],
    ) -> tuple[dict[int, tuple[int, int]], set[int]]:
        by_threat: dict[int, dict[int, JointThreat]] = {}
        for member in members:
            for threat in member.threats:
                if threat.age_seconds <= 1.0:
                    by_threat.setdefault(threat.threat_id, {})[member.entity_id] = threat
        immediate: set[int] = set()
        for threat_id, observations in by_threat.items():
            costs = []
            for member in members:
                threat = observations.get(member.entity_id)
                if threat is None:
                    continue
                metrics = self._threat_metrics(member, threat)
                if metrics is None:
                    continue
                range_m, closing, t_cpa, d_cpa = metrics
                if (
                    0.0 < t_cpa <= self.config.immediate_t_cpa_seconds
                    and d_cpa <= self.config.immediate_d_cpa_metres
                    and range_m <= self.config.immediate_range_metres
                ):
                    immediate.add(member.entity_id)
                if (
                    closing >= self.config.ownership_closing_mps
                    and self.config.ownership_t_cpa_min_seconds
                    <= t_cpa
                    <= self.config.ownership_t_cpa_max_seconds
                    and d_cpa <= self.config.ownership_d_cpa_metres
                    and range_m <= self.config.ownership_range_metres
                ):
                    cost = d_cpa + 50.0 * t_cpa + 0.10 * range_m
                    costs.append((cost, member.entity_id))
            costs.sort()
            candidate = None
            if len(costs) >= 2:
                best, second = costs[0], costs[1]
                if (
                    best[0] / max(second[0], 1e-9) <= self.config.ownership_ratio
                    and second[0] - best[0] >= self.config.ownership_margin
                ):
                    candidate = best[1]
            prior = self._owner_candidates.get(threat_id)
            count = prior[1] + 1 if prior and prior[0] == candidate and candidate is not None else 1
            if candidate is None:
                self._owner_candidates.pop(threat_id, None)
                continue
            self._owner_candidates[threat_id] = (candidate, count)
            if count >= self.config.ownership_confirm_steps:
                self._owners[threat_id] = (
                    candidate, step + self.config.ownership_hold_steps
                )
        self._owners = {
            threat: value for threat, value in self._owners.items() if value[1] >= step
        }
        return self._owners, immediate

    @staticmethod
    def _threat_metrics(
        member: JointMemberSnapshot, threat: JointThreat
    ) -> tuple[float, float, float, float] | None:
        rx = threat.position.east - member.position.east
        ry = threat.position.north - member.position.north
        ux = threat.velocity.east - member.velocity.east
        uy = threat.velocity.north - member.velocity.north
        range_m = math.hypot(rx, ry)
        speed_sq = ux * ux + uy * uy
        if range_m <= 1e-9 or speed_sq <= 1e-9:
            return None
        dot = rx * ux + ry * uy
        closing = -dot / range_m
        t_cpa = -dot / speed_sq
        d_cpa = math.hypot(rx + ux * max(t_cpa, 0.0), ry + uy * max(t_cpa, 0.0))
        return range_m, closing, t_cpa, d_cpa

    def _collision_candidate(
        self,
        members: tuple[JointMemberSnapshot, ...],
        pair: tuple[int, int],
        owners: dict[int, tuple[int, int]],
        immediate: set[int],
        roles: dict[int, JointRole],
    ) -> int | None:
        protected = immediate | {owner for owner, _until in owners.values()}
        candidates = [
            members[index]
            for index in pair
            if members[index].entity_id not in protected
            and (
                not self.config.protect_primary
                or roles[members[index].entity_id] != JointRole.PRIMARY
            )
        ]
        priority = {
            JointRole.DECOY: 0,
            JointRole.BACKUP: 1,
            JointRole.SECONDARY: 2,
            JointRole.PRIMARY: 3,
            JointRole.LOST: 4,
        }
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda item: (priority[roles[item.entity_id]], item.entity_id),
        ).entity_id

    def _candidate_actions(
        self, baseline_action: int, baseline_separation: float
    ) -> tuple[int, ...]:
        if baseline_action == int(ManeuverAction.STOP):
            if baseline_separation > self.config.stop_to_turn_trigger_metres:
                return ()
            return (int(ManeuverAction.LEFT), int(ManeuverAction.RIGHT))
        if self.config.allow_direct_reversal:
            return tuple(
                action for action in (0, 1, 2) if action != baseline_action
            )
        return (int(ManeuverAction.STOP),)

    def _peer_closing_speed(
        self,
        members: tuple[JointMemberSnapshot, ...],
        pair: tuple[int, int],
    ) -> float:
        first, second = members[pair[0]], members[pair[1]]
        rx = second.position.east - first.position.east
        ry = second.position.north - first.position.north
        vx = second.velocity.east - first.velocity.east
        vy = second.velocity.north - first.velocity.north
        distance = math.hypot(rx, ry)
        if distance <= 1e-9:
            return float("inf")
        return -(rx * vx + ry * vy) / distance

    def _preserves_mission(
        self,
        member: JointMemberSnapshot,
        baseline_action: int,
        candidate_action: int,
    ) -> bool:
        horizon = self.config.peer_horizon_seconds
        if self.config.maximum_added_goal_error_degrees is not None:
            baseline_error = self._goal_error(member, baseline_action, horizon)
            candidate_error = self._goal_error(member, candidate_action, horizon)
            if (
                candidate_error - baseline_error
                > self.config.maximum_added_goal_error_degrees
            ):
                return False
        if self.config.maximum_eta_penalty_seconds is not None:
            baseline_eta = self._action_eta(member, baseline_action, horizon)
            candidate_eta = self._action_eta(member, candidate_action, horizon)
            if candidate_eta - baseline_eta > self.config.maximum_eta_penalty_seconds:
                return False
        return True

    def _goal_error(
        self, member: JointMemberSnapshot, action: int, time_s: float
    ) -> float:
        position = self._predict_position(member, action, time_s)
        velocity = self._predict_velocity(member, action, time_s)
        gx = member.goal_position.east - position.east
        gy = member.goal_position.north - position.north
        return abs(math.degrees(math.atan2(
            velocity.east * gy - velocity.north * gx,
            velocity.east * gx + velocity.north * gy,
        )))

    def _action_eta(
        self, member: JointMemberSnapshot, action: int, time_s: float
    ) -> float:
        position = self._predict_position(member, action, time_s)
        velocity = self._predict_velocity(member, action, time_s)
        gx = member.goal_position.east - position.east
        gy = member.goal_position.north - position.north
        distance = math.hypot(gx, gy)
        radial = 0.0 if distance <= 1e-9 else (
            velocity.east * gx + velocity.north * gy
        ) / distance
        return time_s + distance / max(50.0, radial)

    def _closest_peer_pair(
        self,
        members: tuple[JointMemberSnapshot, ...],
        actions: tuple[int, ...],
    ) -> tuple[int, int]:
        best_pair = (0, 1)
        minimum = float("inf")
        for sample in range(13):
            time_s = self.config.peer_horizon_seconds * sample / 12.0
            positions = [
                self._predict_position(member, action, time_s)
                for member, action in zip(members, actions, strict=True)
            ]
            for first in range(len(members)):
                for second in range(first + 1, len(members)):
                    separation = math.hypot(
                        positions[first].east - positions[second].east,
                        positions[first].north - positions[second].north,
                    )
                    if separation < minimum:
                        minimum = separation
                        best_pair = (first, second)
        return best_pair

    def _group_minimum_separation(
        self,
        members: tuple[JointMemberSnapshot, ...],
        actions: tuple[int, ...],
    ) -> float:
        minimum = float("inf")
        samples = 12
        for sample in range(samples + 1):
            time_s = self.config.peer_horizon_seconds * sample / samples
            positions = [
                self._predict_position(member, action, time_s)
                for member, action in zip(members, actions, strict=True)
            ]
            for first in range(len(members)):
                for second in range(first + 1, len(members)):
                    minimum = min(
                        minimum,
                        math.hypot(
                            positions[first].east - positions[second].east,
                            positions[first].north - positions[second].north,
                        ),
                    )
        return minimum

    def _predict_position(
        self, member: JointMemberSnapshot, action: int, time_s: float
    ) -> Vector2:
        speed = math.hypot(member.velocity.east, member.velocity.north)
        side = 1.0 if action == 0 else (-1.0 if action == 2 else 0.0)
        if speed <= 1e-9:
            lateral_east = lateral_north = 0.0
        else:
            lateral_east = -member.velocity.north / speed
            lateral_north = member.velocity.east / speed
        acceleration_time = min(time_s, self.config.action_hold_seconds)
        displacement = self.config.lateral_acceleration_mps2 * (
            acceleration_time * time_s - 0.5 * acceleration_time * acceleration_time
        )
        return Vector2(
            member.position.east
            + member.velocity.east * time_s
            + side * displacement * lateral_east,
            member.position.north
            + member.velocity.north * time_s
            + side * displacement * lateral_north,
        )

    def _predict_velocity(
        self, member: JointMemberSnapshot, action: int, time_s: float
    ) -> Vector2:
        speed = math.hypot(member.velocity.east, member.velocity.north)
        side = 1.0 if action == 0 else (-1.0 if action == 2 else 0.0)
        if speed <= 1e-9:
            return member.velocity
        acceleration_time = min(time_s, self.config.action_hold_seconds)
        lateral_east = -member.velocity.north / speed
        lateral_north = member.velocity.east / speed
        velocity_delta = self.config.lateral_acceleration_mps2 * acceleration_time
        return Vector2(
            member.velocity.east + side * velocity_delta * lateral_east,
            member.velocity.north + side * velocity_delta * lateral_north,
        )

    @staticmethod
    def _eta_spread(
        members: tuple[JointMemberSnapshot, ...]
    ) -> float | None:
        values = []
        for member in members:
            gx = member.goal_position.east - member.position.east
            gy = member.goal_position.north - member.position.north
            distance = math.hypot(gx, gy)
            if distance <= 1e-9:
                values.append(0.0)
                continue
            radial = (
                member.velocity.east * gx + member.velocity.north * gy
            ) / distance
            values.append(distance / max(50.0, radial))
        return max(values) - min(values) if values else None

    def _trim_budget(self, step: int) -> None:
        while self._override_steps and (
            step - self._override_steps[0] >= self.config.override_window_steps
        ):
            self._override_steps.popleft()

    def _materialize(self, action: int) -> int:
        child = self._children[action]
        child.start(1)
        command = child.step()
        if int(command.action) != action or not command.completed:
            raise RuntimeError("child maneuver skill returned an invalid action")
        return int(command.action)


class ThreeMissileJointSkill(HStrikePackageSkill):
    """Backward-compatible exact-three wrapper over HStrikePackageSkill."""

    def __init__(self, config: ThreeMissileJointConfig | None = None) -> None:
        selected = config or ThreeMissileJointConfig()
        super().__init__(replace(selected, min_members=3, max_members=3))


__all__ = [
    "HStrikePackageSkill",
    "JointDecision",
    "JointMemberSnapshot",
    "JointThreat",
    "ThreeMissileJointConfig",
    "ThreeMissileJointSkill",
    "Vector2",
]
