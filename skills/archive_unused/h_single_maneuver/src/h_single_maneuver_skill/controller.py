"""Threat-aware H-missile controller composed from three maneuver Skills."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import math
from typing import Any, Mapping, Sequence

from skill_generation.errors import StateError, ValidationError
from skill_generation.skills import ManeuverAction, ManeuverCommand, ManeuverState
from stop_turning_skill import StopTurningSkill
from turn_left_skill import TurnLeftSkill
from turn_right_skill import TurnRightSkill


H_MISSILE_TYPE = 21000


def _finite(value: float, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError("non_finite_value", field, "finite float", result)
    return result


def _positive(value: float, field: str, *, allow_zero: bool = False) -> float:
    result = _finite(value, field)
    if result < 0.0 or (not allow_zero and result == 0.0):
        expected = "non-negative float" if allow_zero else "positive float"
        raise ValidationError("invalid_config_value", field, expected, result)
    return result


def _steps(value: int, field: str, *, allow_zero: bool = False) -> int:
    if type(value) is not int or value < 0 or (not allow_zero and value == 0):
        expected = "non-negative integer" if allow_zero else "positive integer"
        raise ValidationError("invalid_config_value", field, expected, value)
    return value


@dataclass(frozen=True)
class Vector2:
    """A local tangent-plane vector in metres or metres per second."""

    east: float
    north: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "east", _finite(self.east, "vector.east"))
        object.__setattr__(self, "north", _finite(self.north, "vector.north"))


@dataclass(frozen=True)
class ThreatTrack:
    """Kinematics for one detected interceptor."""

    entity_id: int
    position: Vector2
    velocity: Vector2

    def __post_init__(self) -> None:
        if type(self.entity_id) is not int:
            raise ValidationError(
                "invalid_entity_id", "threat.entity_id", "integer", type(self.entity_id).__name__
            )


@dataclass(frozen=True)
class HMissileSnapshot:
    """One simulator-step observation for exactly one H missile."""

    step: int
    missile_id: int
    position: Vector2
    velocity: Vector2
    goal_position: Vector2
    threats: tuple[ThreatTrack, ...] = ()

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 0:
            raise ValidationError("invalid_step", "step", "non-negative integer", self.step)
        if type(self.missile_id) is not int:
            raise ValidationError(
                "invalid_entity_id", "missile_id", "integer", type(self.missile_id).__name__
            )
        normalized = tuple(self.threats)
        if any(not isinstance(track, ThreatTrack) for track in normalized):
            raise ValidationError(
                "invalid_threat", "threats", "ThreatTrack sequence", "invalid item"
            )
        ids = [track.entity_id for track in normalized]
        if len(ids) != len(set(ids)):
            raise ValidationError("duplicate_threat", "threats", "unique entity ids", ids)
        object.__setattr__(self, "threats", normalized)


@dataclass(frozen=True)
class HSingleManeuverConfig:
    """Conservative defaults inferred from H-missile trajectory evidence."""

    prediction_horizon_seconds: float = 45.0
    cpa_distance_metres: float = 3000.0
    emergency_range_metres: float = 750.0
    terminal_goal_distance_metres: float = 12000.0
    terminal_risk_scale: float = 0.65
    minimum_closing_speed_mps: float = 50.0
    minimum_own_speed_mps: float = 10.0
    goal_deadband_degrees: float = 4.0
    break_steps: int = 2
    settle_steps: int = 1
    recovery_steps: int = 1
    cooldown_steps: int = 4
    side_lock_steps: int = 8
    maximum_breaks: int = 2
    escape_direction: str = "geometry"

    def __post_init__(self) -> None:
        for name in (
            "prediction_horizon_seconds",
            "cpa_distance_metres",
            "emergency_range_metres",
            "terminal_goal_distance_metres",
            "minimum_own_speed_mps",
            "goal_deadband_degrees",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        object.__setattr__(
            self,
            "minimum_closing_speed_mps",
            _positive(
                self.minimum_closing_speed_mps,
                "minimum_closing_speed_mps",
                allow_zero=True,
            ),
        )
        risk_scale = _positive(self.terminal_risk_scale, "terminal_risk_scale")
        if risk_scale > 1.0:
            raise ValidationError(
                "invalid_config_value", "terminal_risk_scale", "value in (0, 1]", risk_scale
            )
        object.__setattr__(self, "terminal_risk_scale", risk_scale)
        for name in ("break_steps", "settle_steps", "recovery_steps"):
            object.__setattr__(self, name, _steps(getattr(self, name), name))
        for name in ("cooldown_steps", "side_lock_steps", "maximum_breaks"):
            object.__setattr__(
                self, name, _steps(getattr(self, name), name, allow_zero=True)
            )
        if self.escape_direction not in {"geometry", "left", "right"}:
            raise ValidationError(
                "invalid_config_value",
                "escape_direction",
                "geometry, left, or right",
                self.escape_direction,
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "HSingleManeuverConfig":
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValidationError("unknown_config_fields", "config", sorted(allowed), unknown)
        return cls(**dict(values))


@dataclass(frozen=True)
class ThreatAssessment:
    threat_id: int
    range_metres: float
    time_to_cpa_seconds: float
    cpa_distance_metres: float
    closing_speed_mps: float
    emergency: bool


class ManeuverPhase(str, Enum):
    CRUISE = "CRUISE"
    EVADE = "EVADE"
    SETTLE = "SETTLE"
    RECOVER = "RECOVER"


@dataclass(frozen=True)
class HSingleManeuverDecision:
    """Composite decision and the exact child-skill command used for this step."""

    command: ManeuverCommand
    phase: ManeuverPhase
    reason: str
    goal_error_degrees: float
    threat: ThreatAssessment | None

    @property
    def action(self) -> ManeuverAction:
        return self.command.action

    @property
    def engine_acc_z(self) -> float:
        return self.command.engine_acc_z


class HSingleManeuverSkill:
    """One-missile state machine that delegates every command to a child Skill.

    The rule is deliberately STOP-dominant. It emits a short, side-locked break
    only for an imminent closest-point-of-approach conflict, settles at zero
    lateral acceleration, and then performs bounded one-step goal recovery.
    """

    def __init__(self, config: HSingleManeuverConfig | None = None) -> None:
        self.config = config or HSingleManeuverConfig()
        self._children = {
            ManeuverAction.LEFT: TurnLeftSkill(),
            ManeuverAction.STOP: StopTurningSkill(),
            ManeuverAction.RIGHT: TurnRightSkill(),
        }
        self.reset()

    def reset(self) -> None:
        for child in self._children.values():
            child.reset()
        self.phase = ManeuverPhase.CRUISE
        self._active_action: ManeuverAction | None = None
        self._active_reason = "cruise"
        self._missile_id: int | None = None
        self._last_step: int | None = None
        self._last_escape_action: ManeuverAction | None = None
        self._side_lock_until = -1
        self._cooldown_until = -1
        self._last_recovery_action: ManeuverAction | None = None
        self._breaks_started = 0

    def step(self, snapshot: HMissileSnapshot) -> HSingleManeuverDecision:
        self._validate_sequence(snapshot)
        goal_error = self._goal_error_degrees(snapshot)
        threat = self._select_threat(snapshot)

        if self._active_action is None:
            self._schedule(snapshot, goal_error, threat)
        phase = self.phase
        reason = self._active_reason
        child = self._children[self._active_action]
        command = child.step()
        if command.completed:
            self._finish_phase(snapshot.step, phase)

        self._last_step = snapshot.step
        return HSingleManeuverDecision(
            command=command,
            phase=phase,
            reason=reason,
            goal_error_degrees=goal_error,
            threat=threat,
        )

    def _validate_sequence(self, snapshot: HMissileSnapshot) -> None:
        if self._missile_id is None:
            self._missile_id = snapshot.missile_id
        elif snapshot.missile_id != self._missile_id:
            raise StateError(
                "missile_changed", "missile_id", self._missile_id, snapshot.missile_id
            )
        if self._last_step is not None and snapshot.step <= self._last_step:
            raise StateError(
                "non_monotonic_step", "step", f"> {self._last_step}", snapshot.step
            )

    def _schedule(
        self,
        snapshot: HMissileSnapshot,
        goal_error: float,
        threat: ThreatAssessment | None,
    ) -> None:
        emergency = threat is not None and threat.emergency
        if self.phase is ManeuverPhase.SETTLE:
            self._start(ManeuverAction.STOP, self.config.settle_steps, "post-break settle")
            return

        break_available = self._breaks_started < self.config.maximum_breaks
        if (
            threat is not None
            and break_available
            and (emergency or snapshot.step >= self._cooldown_until)
        ):
            action = self._escape_action(snapshot, threat)
            self.phase = ManeuverPhase.EVADE
            self._last_escape_action = action
            self._side_lock_until = snapshot.step + self.config.side_lock_steps
            self._breaks_started += 1
            self._start(action, self.config.break_steps, "predicted interceptor conflict")
            return

        if self.phase is ManeuverPhase.RECOVER:
            if abs(goal_error) <= self.config.goal_deadband_degrees:
                self.phase = ManeuverPhase.CRUISE
                self._last_recovery_action = None
                self._start(ManeuverAction.STOP, 1, "goal corridor reacquired")
                return
            desired = ManeuverAction.LEFT if goal_error > 0.0 else ManeuverAction.RIGHT
            if (
                self._last_recovery_action is not None
                and desired is not self._last_recovery_action
            ):
                self.phase = ManeuverPhase.CRUISE
                self._last_recovery_action = None
                self._start(ManeuverAction.STOP, 1, "recovery crossing guard")
                return
            self._last_recovery_action = desired
            self._start(desired, self.config.recovery_steps, "bounded goal recovery")
            return

        self.phase = ManeuverPhase.CRUISE
        self._start(ManeuverAction.STOP, 1, "no imminent conflict")

    def _start(self, action: ManeuverAction, steps: int, reason: str) -> None:
        child = self._children[action]
        if child.state is ManeuverState.ACTIVE:
            raise StateError(
                "child_already_active", "child.state", "IDLE or COMPLETE", child.state.value
            )
        child.start(steps)
        self._active_action = action
        self._active_reason = reason

    def _finish_phase(self, step: int, phase: ManeuverPhase) -> None:
        self._active_action = None
        if phase is ManeuverPhase.EVADE:
            self.phase = ManeuverPhase.SETTLE
            self._cooldown_until = step + self.config.cooldown_steps
        elif phase is ManeuverPhase.SETTLE:
            self.phase = ManeuverPhase.RECOVER

    def _select_threat(self, snapshot: HMissileSnapshot) -> ThreatAssessment | None:
        goal_range = self._distance(snapshot.position, snapshot.goal_position)
        terminal = goal_range <= self.config.terminal_goal_distance_metres
        cpa_limit = self.config.cpa_distance_metres * (
            self.config.terminal_risk_scale if terminal else 1.0
        )
        candidates: list[tuple[tuple[float, ...], ThreatAssessment]] = []
        for track in snapshot.threats:
            relative_x = track.position.east - snapshot.position.east
            relative_y = track.position.north - snapshot.position.north
            range_metres = math.hypot(relative_x, relative_y)
            relative_vx = track.velocity.east - snapshot.velocity.east
            relative_vy = track.velocity.north - snapshot.velocity.north
            relative_speed_sq = relative_vx * relative_vx + relative_vy * relative_vy
            if range_metres > 1e-9:
                closing = -(
                    relative_x * relative_vx + relative_y * relative_vy
                ) / range_metres
            else:
                closing = float("inf")
            if relative_speed_sq > 1e-9:
                time_to_cpa = -(
                    relative_x * relative_vx + relative_y * relative_vy
                ) / relative_speed_sq
                cpa_x = relative_x + relative_vx * max(time_to_cpa, 0.0)
                cpa_y = relative_y + relative_vy * max(time_to_cpa, 0.0)
                cpa_distance = math.hypot(cpa_x, cpa_y)
            else:
                time_to_cpa = float("inf")
                cpa_distance = range_metres
            emergency = range_metres <= self.config.emergency_range_metres
            predicted_conflict = (
                0.0 <= time_to_cpa <= self.config.prediction_horizon_seconds
                and cpa_distance <= cpa_limit
                and closing >= self.config.minimum_closing_speed_mps
            )
            if not emergency and not predicted_conflict:
                continue
            assessment = ThreatAssessment(
                threat_id=track.entity_id,
                range_metres=range_metres,
                time_to_cpa_seconds=time_to_cpa,
                cpa_distance_metres=cpa_distance,
                closing_speed_mps=closing,
                emergency=emergency,
            )
            normalized_cpa = cpa_distance / max(cpa_limit, 1e-9)
            normalized_time = max(time_to_cpa, 0.0) / self.config.prediction_horizon_seconds
            key = (0.0 if emergency else 1.0, normalized_cpa + normalized_time, range_metres)
            candidates.append((key, assessment))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def _escape_action(
        self, snapshot: HMissileSnapshot, threat: ThreatAssessment
    ) -> ManeuverAction:
        if self.config.escape_direction == "left":
            return ManeuverAction.LEFT
        if self.config.escape_direction == "right":
            return ManeuverAction.RIGHT
        if (
            self._last_escape_action is not None
            and snapshot.step <= self._side_lock_until
        ):
            return self._last_escape_action
        speed = math.hypot(snapshot.velocity.east, snapshot.velocity.north)
        if speed >= self.config.minimum_own_speed_mps:
            left_east = -snapshot.velocity.north / speed
            left_north = snapshot.velocity.east / speed

            def minimum_separation(side: float) -> float:
                result = float("inf")
                for track in snapshot.threats:
                    relative_x = track.position.east - snapshot.position.east
                    relative_y = track.position.north - snapshot.position.north
                    relative_vx = track.velocity.east - snapshot.velocity.east
                    relative_vy = track.velocity.north - snapshot.velocity.north
                    relative_speed_sq = (
                        relative_vx * relative_vx + relative_vy * relative_vy
                    )
                    if relative_speed_sq <= 1e-9:
                        time_to_cpa = 0.0
                    else:
                        time_to_cpa = max(
                            0.0,
                            min(
                                self.config.prediction_horizon_seconds,
                                -(
                                    relative_x * relative_vx
                                    + relative_y * relative_vy
                                )
                                / relative_speed_sq,
                            ),
                        )
                    acceleration_time = min(
                        time_to_cpa, float(self.config.break_steps)
                    )
                    displacement = 0.5 * 196.0 * acceleration_time * acceleration_time
                    predicted_x = (
                        relative_x
                        + relative_vx * time_to_cpa
                        - side * displacement * left_east
                    )
                    predicted_y = (
                        relative_y
                        + relative_vy * time_to_cpa
                        - side * displacement * left_north
                    )
                    result = min(result, math.hypot(predicted_x, predicted_y))
                return result

            left_separation = minimum_separation(1.0)
            right_separation = minimum_separation(-1.0)
            if abs(left_separation - right_separation) >= 50.0:
                return (
                    ManeuverAction.LEFT
                    if left_separation > right_separation
                    else ManeuverAction.RIGHT
                )
        goal_error = self._goal_error_degrees(snapshot)
        if abs(goal_error) > self.config.goal_deadband_degrees:
            return ManeuverAction.LEFT if goal_error > 0.0 else ManeuverAction.RIGHT
        parity = (snapshot.missile_id ^ threat.threat_id) & 1
        return ManeuverAction.LEFT if parity == 0 else ManeuverAction.RIGHT

    def _goal_error_degrees(self, snapshot: HMissileSnapshot) -> float:
        velocity_norm = math.hypot(snapshot.velocity.east, snapshot.velocity.north)
        if velocity_norm < self.config.minimum_own_speed_mps:
            return 0.0
        goal_x = snapshot.goal_position.east - snapshot.position.east
        goal_y = snapshot.goal_position.north - snapshot.position.north
        if math.hypot(goal_x, goal_y) <= 1e-9:
            return 0.0
        cross = snapshot.velocity.east * goal_y - snapshot.velocity.north * goal_x
        dot = snapshot.velocity.east * goal_x + snapshot.velocity.north * goal_y
        return math.degrees(math.atan2(cross, dot))

    @staticmethod
    def _distance(first: Vector2, second: Vector2) -> float:
        return math.hypot(second.east - first.east, second.north - first.north)


__all__ = [
    "H_MISSILE_TYPE",
    "HMissileSnapshot",
    "HSingleManeuverConfig",
    "HSingleManeuverDecision",
    "HSingleManeuverSkill",
    "ManeuverPhase",
    "ThreatAssessment",
    "ThreatTrack",
    "Vector2",
]
