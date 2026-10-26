"""Pure deterministic S-pattern holding control for staging H reserves."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal


H_MISSILE_TYPE = 21000

LEFT = 0
STRAIGHT = 1
RIGHT = 2

STAGING = "STAGING"
_PACKAGE_STATES = frozenset((STAGING, "DISPATCHED", "FALLBACK"))
_EARTH_MEAN_RADIUS_KM = 6371.0088
_GEOMETRY_EPSILON_KM = 1e-9
_HEADING_EPSILON_RAD = 1e-9

Phase = Literal["left_leg", "right_leg", "transition"]
LegPhase = Literal["left_leg", "right_leg"]


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


def _geographic_pair(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must be a (longitude, latitude) tuple")
    longitude = _finite(value[0], f"{name}[0]")
    latitude = _finite(value[1], f"{name}[1]")
    if not -180.0 <= longitude <= 180.0:
        raise ValueError(f"{name} longitude must be in [-180, 180]")
    if not -90.0 <= latitude <= 90.0:
        raise ValueError(f"{name} latitude must be in [-90, 90]")
    return longitude, latitude


def _velocity_pair(value: object) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("velocity must be an (east, north) tuple")
    east = _finite(value[0], "velocity[0]")
    north = _finite(value[1], "velocity[1]")
    if math.hypot(east, north) <= _GEOMETRY_EPSILON_KM:
        raise ValueError("velocity must be nonzero")
    return east, north


def _local_offset_km(
    origin: tuple[float, float], point: tuple[float, float]
) -> tuple[float, float]:
    """Return local east/north displacement using a wrapped tangent plane."""

    origin_lon, origin_lat = origin
    point_lon, point_lat = point
    longitude_delta = ((point_lon - origin_lon + 180.0) % 360.0) - 180.0
    latitude_delta = math.radians(point_lat - origin_lat)
    mean_latitude = math.radians((origin_lat + point_lat) / 2.0)
    east = (
        _EARTH_MEAN_RADIUS_KM
        * math.radians(longitude_delta)
        * math.cos(mean_latitude)
    )
    north = _EARTH_MEAN_RADIUS_KM * latitude_delta
    return east, north


@dataclass(frozen=True)
class HoldingSnapshot:
    """One caller-owned reserve observation and baseline motion action.

    Geographic pairs are ``(longitude, latitude)`` in degrees. Velocity is
    local ``(east, north)`` in any consistent nonzero speed unit.
    """

    entity_id: int
    type: int
    step: int
    position: tuple[float, float]
    velocity: tuple[float, float]
    anchor: tuple[float, float]
    forward_target: tuple[float, float]
    baseline_action: int
    state: str

    def __post_init__(self) -> None:
        _nonnegative_int(self.entity_id, "entity_id")
        _nonnegative_int(self.type, "type")
        _nonnegative_int(self.step, "step")
        object.__setattr__(
            self, "position", _geographic_pair(self.position, "position")
        )
        object.__setattr__(self, "velocity", _velocity_pair(self.velocity))
        object.__setattr__(self, "anchor", _geographic_pair(self.anchor, "anchor"))
        object.__setattr__(
            self,
            "forward_target",
            _geographic_pair(self.forward_target, "forward_target"),
        )
        axis_east, axis_north = _local_offset_km(
            self.anchor, self.forward_target
        )
        if math.hypot(axis_east, axis_north) <= _GEOMETRY_EPSILON_KM:
            raise ValueError("anchor and forward_target must define a nonzero axis")
        if type(self.baseline_action) is not int or self.baseline_action not in (
            LEFT,
            STRAIGHT,
            RIGHT,
        ):
            raise ValueError(
                "baseline_action must be LEFT=0, STRAIGHT=1, or RIGHT=2"
            )
        if type(self.state) is not str or self.state not in _PACKAGE_STATES:
            raise ValueError("state must be STAGING, DISPATCHED, or FALLBACK")


@dataclass(frozen=True)
class ReserveHoldingManeuverConfig:
    """Corridor geometry and bounded discrete-control timing."""

    enabled_type: int = H_MISSILE_TYPE
    corridor_half_width_km: float = 6.0
    heading_offset_deg: float = 35.0
    max_turn_steps: int = 6
    straight_transition_steps: int = 2
    latch: bool = True

    def __post_init__(self) -> None:
        if type(self.enabled_type) is not int or self.enabled_type != H_MISSILE_TYPE:
            raise ValueError("enabled_type must be the H missile type 21000")
        half_width = _finite(
            self.corridor_half_width_km, "corridor_half_width_km"
        )
        if half_width <= 0.0:
            raise ValueError("corridor_half_width_km must be positive")
        heading_offset = _finite(self.heading_offset_deg, "heading_offset_deg")
        if not 0.0 < heading_offset < 90.0:
            raise ValueError("heading_offset_deg must be in (0, 90)")
        _positive_int(self.max_turn_steps, "max_turn_steps")
        _positive_int(self.straight_transition_steps, "straight_transition_steps")
        if type(self.latch) is not bool:
            raise ValueError("latch must be a boolean")
        object.__setattr__(self, "corridor_half_width_km", half_width)
        object.__setattr__(self, "heading_offset_deg", heading_offset)


Config = ReserveHoldingManeuverConfig


@dataclass(frozen=True)
class HoldingDecision:
    """Final action and local-corridor diagnostics for one observation."""

    action: int
    overridden: bool
    phase: Phase
    cross_track: float
    forward_progress: float
    reason: str

    def __post_init__(self) -> None:
        if type(self.action) is not int or self.action not in (LEFT, STRAIGHT, RIGHT):
            raise ValueError("action must be LEFT=0, STRAIGHT=1, or RIGHT=2")
        if type(self.overridden) is not bool:
            raise ValueError("overridden must be a boolean")
        if self.phase not in ("left_leg", "right_leg", "transition"):
            raise ValueError("phase must be left_leg, right_leg, or transition")
        object.__setattr__(
            self, "cross_track", _finite(self.cross_track, "cross_track")
        )
        object.__setattr__(
            self,
            "forward_progress",
            _finite(self.forward_progress, "forward_progress"),
        )
        if type(self.reason) is not str or not self.reason:
            raise ValueError("reason must be a nonempty string")


Decision = HoldingDecision


@dataclass
class _OwnerState:
    phase: Phase
    next_leg: LegPhase | None = None
    transition_remaining: int = 0
    consecutive_turn_steps: int = 0


class ReserveHoldingManeuverSkill:
    """Keep staging H reserves advancing in a bounded S-pattern corridor."""

    def __init__(
        self, config: ReserveHoldingManeuverConfig | None = None
    ) -> None:
        if config is not None and not isinstance(
            config, ReserveHoldingManeuverConfig
        ):
            raise ValueError("config must be a ReserveHoldingManeuverConfig")
        self.config = config or ReserveHoldingManeuverConfig()
        self._owners: dict[int, _OwnerState] = {}

    def decide(self, snapshot: HoldingSnapshot) -> HoldingDecision:
        if not isinstance(snapshot, HoldingSnapshot):
            raise ValueError("snapshot must be a HoldingSnapshot")

        (
            forward_progress,
            cross_track,
            forward_east,
            forward_north,
            velocity_cross,
        ) = self._geometry(snapshot)

        if snapshot.type != self.config.enabled_type:
            return self._passthrough(
                snapshot,
                cross_track,
                forward_progress,
                velocity_cross,
                "type_passthrough",
            )
        if snapshot.state != STAGING:
            return self._passthrough(
                snapshot,
                cross_track,
                forward_progress,
                velocity_cross,
                "state_passthrough",
            )

        owner = self._owners.get(snapshot.entity_id)
        if owner is None:
            owner = _OwnerState(
                self._initial_phase(cross_track, velocity_cross)
            )
            self._owners[snapshot.entity_id] = owner

        if owner.phase == "transition":
            if owner.transition_remaining > 0:
                return self._transition_decision(
                    snapshot, owner, cross_track, forward_progress, "transition"
                )
            assert owner.next_leg is not None
            owner.phase = owner.next_leg
            owner.next_leg = None

        half_width = self.config.corridor_half_width_km
        if owner.phase == "left_leg" and cross_track >= half_width:
            return self._start_transition(
                snapshot,
                owner,
                "right_leg",
                cross_track,
                forward_progress,
                "upper_boundary_transition",
            )
        if owner.phase == "right_leg" and cross_track <= -half_width:
            return self._start_transition(
                snapshot,
                owner,
                "left_leg",
                cross_track,
                forward_progress,
                "lower_boundary_transition",
            )

        assert owner.phase in ("left_leg", "right_leg")
        desired_east, desired_north = self._desired_heading(
            forward_east, forward_north, owner.phase
        )
        velocity_east, velocity_north = snapshot.velocity
        heading_error = math.atan2(
            velocity_east * desired_north - velocity_north * desired_east,
            velocity_east * desired_east + velocity_north * desired_north,
        )
        if abs(heading_error) <= _HEADING_EPSILON_RAD:
            owner.consecutive_turn_steps = 0
            return self._decision(
                snapshot,
                STRAIGHT,
                owner.phase,
                cross_track,
                forward_progress,
                "heading_aligned",
            )

        if owner.consecutive_turn_steps >= self.config.max_turn_steps:
            owner.consecutive_turn_steps = 0
            return self._decision(
                snapshot,
                STRAIGHT,
                owner.phase,
                cross_track,
                forward_progress,
                "max_turn_steps_guard",
            )

        action = LEFT if heading_error > 0.0 else RIGHT
        owner.consecutive_turn_steps += 1
        return self._decision(
            snapshot,
            action,
            owner.phase,
            cross_track,
            forward_progress,
            f"{owner.phase}_heading_correction",
        )

    def reset(self, entity_id: int | None = None) -> None:
        """Clear one owner's maneuver state, or every owner when omitted."""

        if entity_id is None:
            self._owners.clear()
            return
        _nonnegative_int(entity_id, "entity_id")
        self._owners.pop(entity_id, None)

    def _passthrough(
        self,
        snapshot: HoldingSnapshot,
        cross_track: float,
        forward_progress: float,
        velocity_cross: float,
        reason: str,
    ) -> HoldingDecision:
        owner = self._owners.get(snapshot.entity_id)
        phase: Phase = (
            owner.phase
            if owner is not None
            else self._initial_phase(cross_track, velocity_cross)
        )
        if not self.config.latch:
            self._owners.pop(snapshot.entity_id, None)
        return self._decision(
            snapshot,
            snapshot.baseline_action,
            phase,
            cross_track,
            forward_progress,
            reason,
        )

    def _start_transition(
        self,
        snapshot: HoldingSnapshot,
        owner: _OwnerState,
        next_leg: LegPhase,
        cross_track: float,
        forward_progress: float,
        reason: str,
    ) -> HoldingDecision:
        owner.phase = "transition"
        owner.next_leg = next_leg
        owner.transition_remaining = self.config.straight_transition_steps
        owner.consecutive_turn_steps = 0
        return self._transition_decision(
            snapshot, owner, cross_track, forward_progress, reason
        )

    def _transition_decision(
        self,
        snapshot: HoldingSnapshot,
        owner: _OwnerState,
        cross_track: float,
        forward_progress: float,
        reason: str,
    ) -> HoldingDecision:
        owner.transition_remaining -= 1
        return self._decision(
            snapshot,
            STRAIGHT,
            "transition",
            cross_track,
            forward_progress,
            reason,
        )

    def _desired_heading(
        self, forward_east: float, forward_north: float, phase: LegPhase
    ) -> tuple[float, float]:
        angle = math.radians(self.config.heading_offset_deg)
        side = 1.0 if phase == "left_leg" else -1.0
        left_east, left_north = -forward_north, forward_east
        return (
            math.cos(angle) * forward_east + side * math.sin(angle) * left_east,
            math.cos(angle) * forward_north + side * math.sin(angle) * left_north,
        )

    def _initial_phase(
        self, cross_track: float, velocity_cross: float
    ) -> LegPhase:
        if cross_track >= self.config.corridor_half_width_km:
            return "right_leg"
        if cross_track <= -self.config.corridor_half_width_km:
            return "left_leg"
        return "right_leg" if velocity_cross < 0.0 else "left_leg"

    @staticmethod
    def _geometry(
        snapshot: HoldingSnapshot,
    ) -> tuple[float, float, float, float, float]:
        axis_east, axis_north = _local_offset_km(
            snapshot.anchor, snapshot.forward_target
        )
        axis_length = math.hypot(axis_east, axis_north)
        forward_east = axis_east / axis_length
        forward_north = axis_north / axis_length
        left_east, left_north = -forward_north, forward_east
        position_east, position_north = _local_offset_km(
            snapshot.anchor, snapshot.position
        )
        forward_progress = (
            position_east * forward_east + position_north * forward_north
        )
        cross_track = position_east * left_east + position_north * left_north
        velocity_cross = (
            snapshot.velocity[0] * left_east + snapshot.velocity[1] * left_north
        )
        return (
            forward_progress,
            cross_track,
            forward_east,
            forward_north,
            velocity_cross,
        )

    @staticmethod
    def _decision(
        snapshot: HoldingSnapshot,
        action: int,
        phase: Phase,
        cross_track: float,
        forward_progress: float,
        reason: str,
    ) -> HoldingDecision:
        return HoldingDecision(
            action=action,
            overridden=action != snapshot.baseline_action,
            phase=phase,
            cross_track=cross_track,
            forward_progress=forward_progress,
            reason=reason,
        )
