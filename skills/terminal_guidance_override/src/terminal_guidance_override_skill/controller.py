"""Pure pulse terminal guidance over caller-owned observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import math


M_MISSILE_TYPE = 21001
FRESH_DETECT_SOURCE = "detect_fresh"

LEFT = 0
STRAIGHT = 1
RIGHT = 2

_EARTH_MEAN_RADIUS_M = 6_371_008.8
_WGS84_SEMI_MAJOR_AXIS_M = 6_378_137.0
_WGS84_ECCENTRICITY_SQUARED = 6.69437999014e-3
_DEFAULT_CAPTURE_DISTANCE = 2000.0
_DEFAULT_FINAL_STRAIGHT_DISTANCE = 300.0
_DEFAULT_MAX_TRACK_AGE = 5
_DEFAULT_HEADING_ENTER_STRAIGHT_DEG = 6.0
_DEFAULT_HEADING_EXIT_STRAIGHT_DEG = 10.0

GuidanceMode = Literal["capture", "straight", "passthrough"]
ControlState = Literal[
    "PASSTHROUGH",
    "CAPTURE",
    "TURN_PULSE",
    "STRAIGHT_RECOVERY",
    "STRAIGHT",
    "STALE_GRACE",
    "FINAL_STRAIGHT",
]
TurnDirection = Literal["left", "right"]


def _nonnegative_int(value: object, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _positive_int(value: object, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _nonnegative_finite(value: object, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _coordinate(value: object, name: str, lower: float, upper: float) -> float:
    result = _finite(value, name)
    if not lower <= result <= upper:
        raise ValueError(f"{name} must be in [{lower:g}, {upper:g}]")
    return result


def _triple(value: object, name: str) -> tuple[float, float, float]:
    try:
        raw = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{name} must contain three finite numbers") from exc
    if len(raw) != 3:
        raise ValueError(f"{name} must contain three finite numbers")
    return tuple(_finite(component, f"{name} component") for component in raw)  # type: ignore[return-value]


@dataclass(frozen=True)
class TerminalGuidanceSnapshot:
    """One owner's baseline action and caller-supplied terminal observations.

    Longitude/latitude positions use degrees. A local velocity is east/north in
    metres per second. ECF is the same Earth-centred, Earth-fixed frame commonly
    called ECEF.
    """

    entity_id: int
    missile_type: int
    stage: int
    baseline_action: int
    aimpoint_lon: float | None = None
    aimpoint_lat: float | None = None
    track_source: str = FRESH_DETECT_SOURCE
    track_age: float = 0.0
    nearest_interceptor_distance_m: float | None = None
    step: int = 0
    position_lon: float | None = None
    position_lat: float | None = None
    position_ecf: tuple[float, float, float] | None = None
    velocity_east_mps: float | None = None
    velocity_north_mps: float | None = None
    velocity_ecf: tuple[float, float, float] | None = None
    aimpoint_ecf: tuple[float, float, float] | None = None
    velocity_source: str = "provided"

    def __post_init__(self) -> None:
        _nonnegative_int(self.entity_id, "entity_id")
        _nonnegative_int(self.missile_type, "missile_type")
        _nonnegative_int(self.stage, "stage")
        _nonnegative_int(self.step, "step")
        if not isinstance(self.track_source, str) or not self.track_source:
            raise ValueError("track_source must be a nonempty string")
        if not isinstance(self.velocity_source, str) or not self.velocity_source:
            raise ValueError("velocity_source must be a nonempty string")
        object.__setattr__(
            self,
            "track_age",
            _nonnegative_finite(self.track_age, "track_age"),
        )
        if self.nearest_interceptor_distance_m is not None:
            object.__setattr__(
                self,
                "nearest_interceptor_distance_m",
                _nonnegative_finite(
                    self.nearest_interceptor_distance_m,
                    "nearest_interceptor_distance_m",
                ),
            )

        has_position_lla = self._validate_lon_lat_pair(
            "position_lon", "position_lat"
        )
        has_position_ecf = self.position_ecf is not None
        if has_position_lla == has_position_ecf:
            raise ValueError("provide exactly one of lon/lat position or position_ecf")
        if has_position_ecf:
            object.__setattr__(
                self, "position_ecf", _triple(self.position_ecf, "position_ecf")
            )

        has_velocity_local = (self.velocity_east_mps is not None) or (
            self.velocity_north_mps is not None
        )
        if (self.velocity_east_mps is None) != (self.velocity_north_mps is None):
            raise ValueError(
                "velocity_east_mps and velocity_north_mps must be provided together"
            )
        has_velocity_ecf = self.velocity_ecf is not None
        if has_velocity_local == has_velocity_ecf:
            raise ValueError(
                "provide exactly one of local east/north velocity or velocity_ecf"
            )
        if has_velocity_local:
            object.__setattr__(
                self,
                "velocity_east_mps",
                _finite(self.velocity_east_mps, "velocity_east_mps"),
            )
            object.__setattr__(
                self,
                "velocity_north_mps",
                _finite(self.velocity_north_mps, "velocity_north_mps"),
            )
        else:
            object.__setattr__(
                self, "velocity_ecf", _triple(self.velocity_ecf, "velocity_ecf")
            )

        has_aimpoint_lla = self._validate_lon_lat_pair(
            "aimpoint_lon", "aimpoint_lat"
        )
        has_aimpoint_ecf = self.aimpoint_ecf is not None
        if has_aimpoint_lla and has_aimpoint_ecf:
            raise ValueError("provide only one of lon/lat aimpoint or aimpoint_ecf")
        if has_aimpoint_ecf:
            object.__setattr__(
                self, "aimpoint_ecf", _triple(self.aimpoint_ecf, "aimpoint_ecf")
            )

    def _validate_lon_lat_pair(self, lon_name: str, lat_name: str) -> bool:
        longitude = getattr(self, lon_name)
        latitude = getattr(self, lat_name)
        if (longitude is None) != (latitude is None):
            raise ValueError(f"{lon_name} and {lat_name} must be provided together")
        if longitude is None:
            return False
        object.__setattr__(
            self, lon_name, _coordinate(longitude, lon_name, -180.0, 180.0)
        )
        object.__setattr__(
            self, lat_name, _coordinate(latitude, lat_name, -90.0, 90.0)
        )
        return True

    @property
    def has_aimpoint(self) -> bool:
        return self.aimpoint_ecf is not None or self.aimpoint_lon is not None


@dataclass(frozen=True)
class TerminalGuidanceConfig:
    """Thresholds and counters for pulse terminal guidance."""

    enabled_types: tuple[int, ...] = (M_MISSILE_TYPE,)
    min_stage: int = 4
    capture_distance: float = _DEFAULT_CAPTURE_DISTANCE
    final_straight_distance: float = _DEFAULT_FINAL_STRAIGHT_DISTANCE
    max_track_age: int = _DEFAULT_MAX_TRACK_AGE
    stale_grace_age: int = 8
    heading_enter_straight_deg: float = _DEFAULT_HEADING_ENTER_STRAIGHT_DEG
    heading_exit_straight_deg: float = _DEFAULT_HEADING_EXIT_STRAIGHT_DEG
    predicted_miss_threshold_m: float = 80.0
    turn_pulse_steps: int = 1
    straight_recovery_steps: int = 2
    no_improvement_limit: int = 3
    latch: bool = True
    # Deprecated v0.1/v0.2 aliases. They remain constructor-compatible.
    capture_distance_m: float | None = None
    straight_distance_m: float | None = None
    exit_distance_m: float | None = None
    max_track_age_steps: int | None = None
    heading_deadband_degrees: float | None = None
    ignore_interceptors_after_latch: bool = True
    enter_distance_m: float | None = None
    threat_distance_m: float = 3000.0
    straight_action: int = STRAIGHT

    def __post_init__(self) -> None:
        if self.enabled_types != (M_MISSILE_TYPE,):
            raise ValueError("enabled_types must contain only the M missile type 21001")
        _nonnegative_int(self.min_stage, "min_stage")

        capture = _nonnegative_finite(self.capture_distance, "capture_distance")
        legacy_capture = (
            None
            if self.capture_distance_m is None
            else _nonnegative_finite(
                self.capture_distance_m, "capture_distance_m"
            )
        )
        if self.enter_distance_m is not None:
            legacy_enter = _nonnegative_finite(
                self.enter_distance_m, "enter_distance_m"
            )
            if (
                legacy_capture is not None
                and legacy_capture != 3000.0
                and not math.isclose(legacy_capture, legacy_enter)
            ):
                raise ValueError(
                    "enter_distance_m conflicts with capture_distance_m"
                )
            legacy_capture = legacy_enter
        if legacy_capture is not None:
            if (
                capture != _DEFAULT_CAPTURE_DISTANCE
                and not math.isclose(capture, legacy_capture)
            ):
                raise ValueError(
                    "deprecated capture distance conflicts with capture_distance"
                )
            capture = legacy_capture

        final_straight = _nonnegative_finite(
            self.final_straight_distance, "final_straight_distance"
        )
        if self.straight_distance_m is not None:
            legacy_straight = _nonnegative_finite(
                self.straight_distance_m, "straight_distance_m"
            )
            if (
                final_straight != _DEFAULT_FINAL_STRAIGHT_DISTANCE
                and not math.isclose(final_straight, legacy_straight)
            ):
                raise ValueError(
                    "straight_distance_m conflicts with final_straight_distance"
                )
            final_straight = legacy_straight
        if final_straight > capture:
            raise ValueError(
                "final_straight_distance must not exceed capture_distance"
            )

        exit_ = (
            max(3500.0, capture)
            if self.exit_distance_m is None
            else _nonnegative_finite(self.exit_distance_m, "exit_distance_m")
        )
        if self.exit_distance_m is not None and exit_ < capture:
            raise ValueError("exit_distance_m must be at least capture_distance")
        object.__setattr__(self, "capture_distance", capture)
        object.__setattr__(self, "final_straight_distance", final_straight)
        object.__setattr__(self, "capture_distance_m", capture)
        object.__setattr__(self, "enter_distance_m", capture)
        object.__setattr__(self, "straight_distance_m", final_straight)
        object.__setattr__(self, "exit_distance_m", exit_)

        _nonnegative_int(self.max_track_age, "max_track_age")
        max_track_age = self.max_track_age
        if self.max_track_age_steps is not None:
            _nonnegative_int(self.max_track_age_steps, "max_track_age_steps")
            if (
                max_track_age != _DEFAULT_MAX_TRACK_AGE
                and max_track_age != self.max_track_age_steps
            ):
                raise ValueError(
                    "max_track_age_steps conflicts with max_track_age"
                )
            max_track_age = self.max_track_age_steps
        _nonnegative_int(self.stale_grace_age, "stale_grace_age")
        if self.stale_grace_age < max_track_age:
            raise ValueError("stale_grace_age must be at least max_track_age")
        object.__setattr__(self, "max_track_age", max_track_age)
        object.__setattr__(self, "max_track_age_steps", max_track_age)

        heading_enter = _nonnegative_finite(
            self.heading_enter_straight_deg, "heading_enter_straight_deg"
        )
        heading_exit = _nonnegative_finite(
            self.heading_exit_straight_deg, "heading_exit_straight_deg"
        )
        if self.heading_deadband_degrees is not None:
            legacy_deadband = _nonnegative_finite(
                self.heading_deadband_degrees, "heading_deadband_degrees"
            )
            if (
                heading_enter != _DEFAULT_HEADING_ENTER_STRAIGHT_DEG
                and not math.isclose(heading_enter, legacy_deadband)
            ) or (
                heading_exit != _DEFAULT_HEADING_EXIT_STRAIGHT_DEG
                and not math.isclose(heading_exit, legacy_deadband)
            ):
                raise ValueError(
                    "heading_deadband_degrees conflicts with v0.3 heading thresholds"
                )
            heading_enter = legacy_deadband
            heading_exit = legacy_deadband
        if heading_exit > 180.0:
            raise ValueError("heading thresholds must not exceed 180")
        if heading_enter > heading_exit:
            raise ValueError(
                "heading_enter_straight_deg must not exceed heading_exit_straight_deg"
            )
        object.__setattr__(self, "heading_enter_straight_deg", heading_enter)
        object.__setattr__(self, "heading_exit_straight_deg", heading_exit)
        object.__setattr__(self, "heading_deadband_degrees", heading_enter)

        object.__setattr__(
            self,
            "predicted_miss_threshold_m",
            _nonnegative_finite(
                self.predicted_miss_threshold_m, "predicted_miss_threshold_m"
            ),
        )
        _positive_int(self.turn_pulse_steps, "turn_pulse_steps")
        _positive_int(self.straight_recovery_steps, "straight_recovery_steps")
        _positive_int(self.no_improvement_limit, "no_improvement_limit")
        if not isinstance(self.ignore_interceptors_after_latch, bool):
            raise ValueError("ignore_interceptors_after_latch must be a boolean")
        if not isinstance(self.latch, bool):
            raise ValueError("latch must be a boolean")
        object.__setattr__(
            self,
            "threat_distance_m",
            _nonnegative_finite(self.threat_distance_m, "threat_distance_m"),
        )
        if self.straight_action != STRAIGHT or type(self.straight_action) is not int:
            raise ValueError("straight_action must be STRAIGHT=1")


@dataclass(frozen=True)
class TerminalGuidanceDecision:
    """Final maneuver category and auditable terminal-guidance diagnostics."""

    baseline: int
    final: int
    overridden: bool
    latched: bool
    reason: str
    distance: float | None
    threat: bool
    mode: GuidanceMode
    heading_error: float | None
    turn_direction: TurnDirection | None
    predicted_miss_distance: float | None = None
    control_state: ControlState = "PASSTHROUGH"
    no_improvement_count: int = 0
    velocity_source: str = "provided"


@dataclass
class _OwnerState:
    mode: ControlState = "CAPTURE"
    previous_error: float | None = None
    error_sign: int = 0
    no_improvement: int = 0
    pulse_remaining: int = 0
    recovery_remaining: int = 0
    straight_hysteresis: bool = False
    turn_direction: TurnDirection | None = None


class TerminalGuidanceSkill:
    """Capture an M missile and guide it with turn pulses and re-estimation."""

    def __init__(self, config: TerminalGuidanceConfig | None = None) -> None:
        if config is not None and not isinstance(config, TerminalGuidanceConfig):
            raise ValueError("config must be a TerminalGuidanceConfig")
        self.config = config or TerminalGuidanceConfig()
        self._owner_states: dict[int, _OwnerState] = {}

    def decide(self, snapshot: TerminalGuidanceSnapshot) -> TerminalGuidanceDecision:
        if not isinstance(snapshot, TerminalGuidanceSnapshot):
            raise ValueError("snapshot must be a TerminalGuidanceSnapshot")
        baseline = snapshot.baseline_action
        if type(baseline) is not int or baseline not in (LEFT, STRAIGHT, RIGHT):
            raise ValueError(
                "baseline_action must be LEFT=0, STRAIGHT=1, or RIGHT=2"
            )

        threat = (
            snapshot.nearest_interceptor_distance_m is not None
            and snapshot.nearest_interceptor_distance_m
            <= self.config.threat_distance_m
        )
        owner = snapshot.entity_id
        state = self._owner_states.get(owner)
        was_latched = state is not None and self.config.latch

        if snapshot.missile_type != M_MISSILE_TYPE:
            return self._passthrough(
                snapshot, None, threat, "missile_type_passthrough"
            )
        if not snapshot.has_aimpoint:
            return self._passthrough(snapshot, None, threat, "aimpoint_missing")
        if snapshot.track_age > self.config.stale_grace_age:
            return self._passthrough(snapshot, None, threat, "track_stale")

        stale_grace = snapshot.track_age > self.config.max_track_age
        if stale_grace and not was_latched:
            return self._passthrough(snapshot, None, threat, "track_stale")
        if not stale_grace and snapshot.track_source != FRESH_DETECT_SOURCE:
            return self._passthrough(snapshot, None, threat, "track_stale")

        distance, los_east, los_north, velocity_east, velocity_north = (
            _horizontal_geometry(snapshot)
        )
        heading_error = _heading_error_degrees(
            los_east, los_north, velocity_east, velocity_north
        )
        predicted_miss = _predicted_closest_approach_miss(
            los_east, los_north, velocity_east, velocity_north
        )

        if stale_grace:
            assert state is not None
            self._clear_turn_cycle(state)
            state.mode = "STALE_GRACE"
            return self._decision(
                snapshot,
                distance,
                threat,
                heading_error,
                predicted_miss,
                state,
                STRAIGHT,
                "stale_grace",
                "STALE_GRACE",
            )

        entered = state is None
        if not was_latched:
            if snapshot.stage < self.config.min_stage:
                return self._passthrough(
                    snapshot,
                    distance,
                    threat,
                    "stage_below_minimum",
                    heading_error,
                    predicted_miss,
                )
            if distance > self.config.capture_distance:
                return self._passthrough(
                    snapshot,
                    distance,
                    threat,
                    "outside_capture_distance",
                    heading_error,
                    predicted_miss,
                )

        if state is None:
            state = _OwnerState()
            self._owner_states[owner] = state
        return self._guidance(
            snapshot,
            distance,
            threat,
            heading_error,
            predicted_miss,
            state,
            entered,
        )

    def reset(self, entity_id: int | None = None) -> None:
        """Clear one owner's capture state, or all state when no owner is given."""
        if entity_id is None:
            self._owner_states.clear()
            return
        _nonnegative_int(entity_id, "entity_id")
        self._owner_states.pop(entity_id, None)

    def _passthrough(
        self,
        snapshot: TerminalGuidanceSnapshot,
        distance: float | None,
        threat: bool,
        reason: str,
        heading_error: float | None = None,
        predicted_miss: float | None = None,
    ) -> TerminalGuidanceDecision:
        self._owner_states.pop(snapshot.entity_id, None)
        return TerminalGuidanceDecision(
            baseline=snapshot.baseline_action,
            final=snapshot.baseline_action,
            overridden=False,
            latched=False,
            reason=reason,
            distance=distance,
            threat=threat,
            mode="passthrough",
            heading_error=heading_error,
            turn_direction=None,
            predicted_miss_distance=predicted_miss,
            control_state="PASSTHROUGH",
            no_improvement_count=0,
            velocity_source=snapshot.velocity_source,
        )

    def _guidance(
        self,
        snapshot: TerminalGuidanceSnapshot,
        distance: float,
        threat: bool,
        heading_error: float,
        predicted_miss: float,
        state: _OwnerState,
        entered: bool,
    ) -> TerminalGuidanceDecision:
        crossed, observed_no_improvement = self._observe_error(
            state, heading_error
        )
        absolute_error = abs(heading_error)
        if state.straight_hysteresis:
            if absolute_error >= self.config.heading_exit_straight_deg or math.isclose(
                absolute_error,
                self.config.heading_exit_straight_deg,
                abs_tol=1e-9,
            ):
                state.straight_hysteresis = False
        elif absolute_error <= self.config.heading_enter_straight_deg or math.isclose(
            absolute_error,
            self.config.heading_enter_straight_deg,
            abs_tol=1e-9,
        ):
            state.straight_hysteresis = True

        if distance <= self.config.final_straight_distance or math.isclose(
            distance, self.config.final_straight_distance, abs_tol=1e-6
        ):
            self._clear_turn_cycle(state)
            state.no_improvement = 0
            state.mode = "FINAL_STRAIGHT"
            return self._decision(
                snapshot,
                distance,
                threat,
                heading_error,
                predicted_miss,
                state,
                STRAIGHT,
                "final_straight",
                "FINAL_STRAIGHT",
            )

        if crossed:
            state.turn_direction = None
            return self._start_recovery(
                snapshot,
                distance,
                threat,
                heading_error,
                predicted_miss,
                state,
                "heading_sign_cross",
                observed_no_improvement,
            )

        if predicted_miss <= self.config.predicted_miss_threshold_m or math.isclose(
            predicted_miss,
            self.config.predicted_miss_threshold_m,
            abs_tol=1e-6,
        ):
            self._clear_turn_cycle(state)
            state.no_improvement = 0
            state.mode = "STRAIGHT"
            return self._decision(
                snapshot,
                distance,
                threat,
                heading_error,
                predicted_miss,
                state,
                STRAIGHT,
                "predicted_miss_straight",
                "STRAIGHT",
            )

        if state.straight_hysteresis:
            self._clear_turn_cycle(state)
            state.no_improvement = 0
            state.mode = "STRAIGHT"
            return self._decision(
                snapshot,
                distance,
                threat,
                heading_error,
                predicted_miss,
                state,
                STRAIGHT,
                "heading_hysteresis_straight",
                "STRAIGHT",
            )

        if observed_no_improvement >= self.config.no_improvement_limit:
            return self._start_recovery(
                snapshot,
                distance,
                threat,
                heading_error,
                predicted_miss,
                state,
                "no_improvement",
                observed_no_improvement,
            )

        if state.recovery_remaining > 0:
            return self._emit_recovery(
                snapshot,
                distance,
                threat,
                heading_error,
                predicted_miss,
                state,
                "straight_recovery",
                observed_no_improvement,
            )

        if state.pulse_remaining > 0:
            return self._emit_turn(
                snapshot,
                distance,
                threat,
                heading_error,
                predicted_miss,
                state,
                observed_no_improvement,
            )

        desired_direction: TurnDirection = (
            "right" if heading_error > 0.0 else "left"
        )
        if (
            state.turn_direction is not None
            and state.turn_direction != desired_direction
        ):
            state.turn_direction = None
            return self._start_recovery(
                snapshot,
                distance,
                threat,
                heading_error,
                predicted_miss,
                state,
                "turn_reversal_guard",
                observed_no_improvement,
            )

        state.turn_direction = desired_direction
        state.pulse_remaining = self.config.turn_pulse_steps
        return self._emit_turn(
            snapshot,
            distance,
            threat,
            heading_error,
            predicted_miss,
            state,
            observed_no_improvement,
            entered=entered,
        )

    def _observe_error(
        self, state: _OwnerState, heading_error: float
    ) -> tuple[bool, int]:
        sign = 1 if heading_error > 1e-9 else -1 if heading_error < -1e-9 else 0
        crossed = sign != 0 and state.error_sign != 0 and sign != state.error_sign
        if crossed:
            state.no_improvement = 0
        elif state.previous_error is None:
            state.no_improvement = 0
        elif abs(heading_error) >= abs(state.previous_error) - 1e-9:
            state.no_improvement += 1
        else:
            state.no_improvement = 0
        state.previous_error = heading_error
        if sign != 0:
            state.error_sign = sign
        return crossed, state.no_improvement

    def _start_recovery(
        self,
        snapshot: TerminalGuidanceSnapshot,
        distance: float,
        threat: bool,
        heading_error: float,
        predicted_miss: float,
        state: _OwnerState,
        reason: str,
        reported_no_improvement: int,
    ) -> TerminalGuidanceDecision:
        state.pulse_remaining = 0
        state.recovery_remaining = self.config.straight_recovery_steps
        state.no_improvement = 0
        return self._emit_recovery(
            snapshot,
            distance,
            threat,
            heading_error,
            predicted_miss,
            state,
            reason,
            reported_no_improvement,
        )

    def _emit_recovery(
        self,
        snapshot: TerminalGuidanceSnapshot,
        distance: float,
        threat: bool,
        heading_error: float,
        predicted_miss: float,
        state: _OwnerState,
        reason: str,
        no_improvement_count: int,
    ) -> TerminalGuidanceDecision:
        state.recovery_remaining -= 1
        state.mode = "STRAIGHT_RECOVERY"
        return self._decision(
            snapshot,
            distance,
            threat,
            heading_error,
            predicted_miss,
            state,
            STRAIGHT,
            reason,
            "STRAIGHT_RECOVERY",
            no_improvement_count,
        )

    def _emit_turn(
        self,
        snapshot: TerminalGuidanceSnapshot,
        distance: float,
        threat: bool,
        heading_error: float,
        predicted_miss: float,
        state: _OwnerState,
        no_improvement_count: int,
        *,
        entered: bool = False,
    ) -> TerminalGuidanceDecision:
        assert state.turn_direction is not None
        state.pulse_remaining -= 1
        if state.pulse_remaining == 0:
            state.recovery_remaining = self.config.straight_recovery_steps
        state.mode = "TURN_PULSE"
        prefix = "entered_" if entered else ""
        return self._decision(
            snapshot,
            distance,
            threat,
            heading_error,
            predicted_miss,
            state,
            RIGHT if state.turn_direction == "right" else LEFT,
            f"{prefix}turn_pulse_{state.turn_direction}",
            "TURN_PULSE",
            no_improvement_count,
            state.turn_direction,
        )

    @staticmethod
    def _clear_turn_cycle(state: _OwnerState) -> None:
        state.pulse_remaining = 0
        state.recovery_remaining = 0
        state.turn_direction = None

    def _decision(
        self,
        snapshot: TerminalGuidanceSnapshot,
        distance: float,
        threat: bool,
        heading_error: float,
        predicted_miss: float,
        state: _OwnerState,
        final: int,
        reason: str,
        control_state: ControlState,
        no_improvement_count: int = 0,
        turn_direction: TurnDirection | None = None,
    ) -> TerminalGuidanceDecision:
        mode: GuidanceMode = (
            "capture" if control_state == "TURN_PULSE" else "straight"
        )
        return TerminalGuidanceDecision(
            baseline=snapshot.baseline_action,
            final=final,
            overridden=final != snapshot.baseline_action,
            latched=self.config.latch and snapshot.entity_id in self._owner_states,
            reason=reason,
            distance=distance,
            threat=threat,
            mode=mode,
            heading_error=heading_error,
            turn_direction=turn_direction,
            predicted_miss_distance=predicted_miss,
            control_state=control_state,
            no_improvement_count=no_improvement_count,
            velocity_source=snapshot.velocity_source,
        )


def _horizontal_geometry(
    snapshot: TerminalGuidanceSnapshot,
) -> tuple[float, float, float, float, float]:
    if snapshot.position_ecf is not None:
        position_ecf = snapshot.position_ecf
        position_lon, position_lat = _ecf_to_lon_lat(position_ecf)
    else:
        assert snapshot.position_lon is not None
        assert snapshot.position_lat is not None
        position_lon = snapshot.position_lon
        position_lat = snapshot.position_lat
        position_ecf = _lla_to_ecf(position_lon, position_lat)

    if snapshot.aimpoint_ecf is not None:
        aimpoint_ecf = snapshot.aimpoint_ecf
    else:
        assert snapshot.aimpoint_lon is not None
        assert snapshot.aimpoint_lat is not None
        aimpoint_ecf = _lla_to_ecf(snapshot.aimpoint_lon, snapshot.aimpoint_lat)

    delta = tuple(target - current for target, current in zip(aimpoint_ecf, position_ecf))
    east_basis, north_basis = _local_horizontal_basis(position_lon, position_lat)
    los_east = _dot(delta, east_basis)
    los_north = _dot(delta, north_basis)

    if snapshot.position_ecf is None and snapshot.aimpoint_ecf is None:
        assert snapshot.position_lon is not None
        assert snapshot.position_lat is not None
        assert snapshot.aimpoint_lon is not None
        assert snapshot.aimpoint_lat is not None
        distance = _haversine_distance_m(
            snapshot.position_lon,
            snapshot.position_lat,
            snapshot.aimpoint_lon,
            snapshot.aimpoint_lat,
        )
    else:
        distance = math.dist(position_ecf, aimpoint_ecf)

    if snapshot.position_ecf is None and snapshot.aimpoint_ecf is None:
        horizontal_los = math.hypot(los_east, los_north)
        if horizontal_los > 1e-9:
            scale = distance / horizontal_los
            los_east *= scale
            los_north *= scale

    if snapshot.velocity_ecf is not None:
        velocity_east = _dot(snapshot.velocity_ecf, east_basis)
        velocity_north = _dot(snapshot.velocity_ecf, north_basis)
    else:
        assert snapshot.velocity_east_mps is not None
        assert snapshot.velocity_north_mps is not None
        velocity_east = snapshot.velocity_east_mps
        velocity_north = snapshot.velocity_north_mps
    return distance, los_east, los_north, velocity_east, velocity_north


def _predicted_closest_approach_miss(
    los_east: float,
    los_north: float,
    velocity_east: float,
    velocity_north: float,
) -> float:
    speed_squared = velocity_east * velocity_east + velocity_north * velocity_north
    if speed_squared <= 1e-18:
        raise ValueError("horizontal velocity must be nonzero")
    closest_time = max(
        0.0,
        (los_east * velocity_east + los_north * velocity_north) / speed_squared,
    )
    miss_east = los_east - velocity_east * closest_time
    miss_north = los_north - velocity_north * closest_time
    return math.hypot(miss_east, miss_north)


def _heading_error_degrees(
    los_east: float,
    los_north: float,
    velocity_east: float,
    velocity_north: float,
) -> float:
    if math.hypot(velocity_east, velocity_north) <= 1e-9:
        raise ValueError("horizontal velocity must be nonzero")
    if math.hypot(los_east, los_north) <= 1e-9:
        return 0.0
    velocity_heading = math.atan2(velocity_east, velocity_north)
    los_heading = math.atan2(los_east, los_north)
    error = (los_heading - velocity_heading + math.pi) % (2.0 * math.pi) - math.pi
    return math.degrees(error)


def _local_horizontal_basis(
    lon: float, lat: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    longitude = math.radians(lon)
    latitude = math.radians(lat)
    sin_lon = math.sin(longitude)
    cos_lon = math.cos(longitude)
    sin_lat = math.sin(latitude)
    cos_lat = math.cos(latitude)
    return (
        (-sin_lon, cos_lon, 0.0),
        (-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat),
    )


def _dot(
    first: tuple[float, float, float], second: tuple[float, float, float]
) -> float:
    return sum(left * right for left, right in zip(first, second))


def _haversine_distance_m(
    first_lon: float, first_lat: float, second_lon: float, second_lat: float
) -> float:
    first_latitude = math.radians(first_lat)
    second_latitude = math.radians(second_lat)
    longitude_delta = ((second_lon - first_lon + 180.0) % 360.0) - 180.0
    sin_lat = math.sin((second_latitude - first_latitude) / 2.0)
    sin_lon = math.sin(math.radians(longitude_delta) / 2.0)
    haversine = (
        sin_lat * sin_lat
        + math.cos(first_latitude)
        * math.cos(second_latitude)
        * sin_lon
        * sin_lon
    )
    return 2.0 * _EARTH_MEAN_RADIUS_M * math.asin(
        math.sqrt(min(1.0, max(0.0, haversine)))
    )


def _lla_to_ecf(lon: float, lat: float) -> tuple[float, float, float]:
    longitude = math.radians(lon)
    latitude = math.radians(lat)
    sin_latitude = math.sin(latitude)
    prime_vertical_radius = _WGS84_SEMI_MAJOR_AXIS_M / math.sqrt(
        1.0 - _WGS84_ECCENTRICITY_SQUARED * sin_latitude * sin_latitude
    )
    return (
        prime_vertical_radius * math.cos(latitude) * math.cos(longitude),
        prime_vertical_radius * math.cos(latitude) * math.sin(longitude),
        prime_vertical_radius
        * (1.0 - _WGS84_ECCENTRICITY_SQUARED)
        * sin_latitude,
    )


def _ecf_to_lon_lat(position: tuple[float, float, float]) -> tuple[float, float]:
    x, y, z = position
    horizontal = math.hypot(x, y)
    if math.hypot(horizontal, z) <= 1e-9:
        raise ValueError("position_ecf must not be the Earth centre")
    longitude = math.atan2(y, x)
    if horizontal <= 1e-9:
        latitude = math.copysign(math.pi / 2.0, z)
    else:
        latitude = math.atan2(z, horizontal * (1.0 - _WGS84_ECCENTRICITY_SQUARED))
        for _ in range(8):
            sin_latitude = math.sin(latitude)
            prime_vertical_radius = _WGS84_SEMI_MAJOR_AXIS_M / math.sqrt(
                1.0
                - _WGS84_ECCENTRICITY_SQUARED
                * sin_latitude
                * sin_latitude
            )
            altitude = horizontal / math.cos(latitude) - prime_vertical_radius
            denominator = horizontal * (
                1.0
                - _WGS84_ECCENTRICITY_SQUARED
                * prime_vertical_radius
                / (prime_vertical_radius + altitude)
            )
            latitude = math.atan2(z, denominator)
    return math.degrees(longitude), math.degrees(latitude)
