"""Pure H/M retarget selection over caller-supplied legal snapshots."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from math import acos, asin, ceil, cos, degrees, hypot, isfinite, radians, sin, sqrt
from types import MappingProxyType


H_MISSILE_TYPES = frozenset((21000, 21001))
PUBLIC_TARGET_TYPES = frozenset((9400, 9600))


def _nonnegative_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _finite_number(value: object, name: str, *, minimum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and numeric < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _positive_finite_number(value: object, name: str) -> None:
    _finite_number(value, name)
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")


def _positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _probability(value: object, name: str) -> None:
    _finite_number(value, name, minimum=0.0)
    if float(value) > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class RetargetMemberSnapshot:
    entity_id: int
    missile_type: int
    current_target_id: int
    alive: bool
    stage: int
    current_distance_km: float
    communication_group: Hashable
    is_core: bool
    retarget_count: int
    last_retarget_step: int | None
    lon: float
    lat: float
    legal_target_ids: tuple[int, ...]
    fresh_target_ids: tuple[int, ...] = ()
    velocity_x: float | None = None
    velocity_y: float | None = None

    def __post_init__(self) -> None:
        _nonnegative_int(self.entity_id, "entity_id")
        if (
            isinstance(self.missile_type, bool)
            or not isinstance(self.missile_type, int)
            or self.missile_type not in H_MISSILE_TYPES
        ):
            raise ValueError("missile_type must be H=21000 or M=21001")
        _nonnegative_int(self.current_target_id, "current_target_id")
        if not isinstance(self.alive, bool):
            raise ValueError("alive must be a boolean")
        _nonnegative_int(self.stage, "stage")
        _finite_number(self.current_distance_km, "current_distance_km", minimum=0.0)
        if (
            self.communication_group is None
            or isinstance(self.communication_group, bool)
            or not isinstance(self.communication_group, Hashable)
        ):
            raise ValueError("communication_group must be a non-null hashable value")
        if isinstance(self.communication_group, int) and self.communication_group < 0:
            raise ValueError("communication_group integer must be nonnegative")
        if isinstance(self.communication_group, str) and not self.communication_group:
            raise ValueError("communication_group string must not be empty")
        if not isinstance(self.is_core, bool):
            raise ValueError("is_core must be a boolean")
        _nonnegative_int(self.retarget_count, "retarget_count")
        if self.last_retarget_step is not None:
            _nonnegative_int(self.last_retarget_step, "last_retarget_step")
        _finite_number(self.lon, "lon")
        _finite_number(self.lat, "lat")
        if not isinstance(self.legal_target_ids, tuple):
            raise ValueError("legal_target_ids must be a tuple")
        for target_id in self.legal_target_ids:
            _nonnegative_int(target_id, "legal target_id")
        if len(self.legal_target_ids) != len(set(self.legal_target_ids)):
            raise ValueError("legal_target_ids must be unique")
        if not isinstance(self.fresh_target_ids, tuple):
            raise ValueError("fresh_target_ids must be a tuple")
        for target_id in self.fresh_target_ids:
            _nonnegative_int(target_id, "fresh target_id")
        if len(self.fresh_target_ids) != len(set(self.fresh_target_ids)):
            raise ValueError("fresh_target_ids must be unique")
        if (self.velocity_x is None) != (self.velocity_y is None):
            raise ValueError("velocity_x and velocity_y must both be set or both be None")
        if self.velocity_x is not None:
            _finite_number(self.velocity_x, "velocity_x")
            _finite_number(self.velocity_y, "velocity_y")


@dataclass(frozen=True)
class RetargetTargetSnapshot:
    target_id: int
    type: int
    value: float
    lon: float
    lat: float
    destruction_probability: float
    projected_probability: float
    assigned_count: int

    def __post_init__(self) -> None:
        _nonnegative_int(self.target_id, "target_id")
        if isinstance(self.type, bool) or not isinstance(self.type, int):
            raise ValueError("target type must be public 9400 or 9600")
        if self.type not in PUBLIC_TARGET_TYPES:
            raise ValueError("target type must be public 9400 or 9600")
        _finite_number(self.value, "value", minimum=0.0)
        _finite_number(self.lon, "lon")
        _finite_number(self.lat, "lat")
        _probability(self.destruction_probability, "destruction_probability")
        _probability(self.projected_probability, "projected_probability")
        _nonnegative_int(self.assigned_count, "assigned_count")


@dataclass(frozen=True)
class RetargetRequest:
    entity_id: int
    old_target_id: int
    new_target_id: int
    reason: str
    target_destruction_probability: float


@dataclass(frozen=True)
class RetargetDecision:
    requests: tuple[RetargetRequest, ...]
    eligible_member_ids: tuple[int, ...]
    budget: int
    rejected_reasons: Mapping[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.rejected_reasons, Mapping):
            raise ValueError("rejected_reasons must be a mapping")
        reasons = dict(self.rejected_reasons)
        for entity_id, reason in reasons.items():
            _nonnegative_int(entity_id, "rejected entity_id")
            if not isinstance(reason, str) or not reason:
                raise ValueError("rejected reason must be a nonempty string")
        object.__setattr__(self, "rejected_reasons", MappingProxyType(reasons))


@dataclass(frozen=True)
class DynamicRetargetConfig:
    destruction_threshold: float = 0.90
    max_stage: int = 2
    min_current_distance_km: float = 15.0
    cooldown_steps: int = 25
    max_retargets: int = 1
    max_fraction_per_step: float = 0.20
    distance_weight: float = 0.05
    max_target_separation_km: float = 15.0
    max_turn_angle_degrees: float = 60.0
    max_candidate_distance_km: float = 60.0
    max_per_old_target_per_step: int = 1

    def __post_init__(self) -> None:
        _probability(self.destruction_threshold, "destruction_threshold")
        if float(self.destruction_threshold) == 0.0:
            raise ValueError("destruction_threshold must be in (0, 1]")
        _nonnegative_int(self.max_stage, "max_stage")
        _finite_number(
            self.min_current_distance_km,
            "min_current_distance_km",
            minimum=0.0,
        )
        _nonnegative_int(self.cooldown_steps, "cooldown_steps")
        _nonnegative_int(self.max_retargets, "max_retargets")
        _probability(self.max_fraction_per_step, "max_fraction_per_step")
        if float(self.max_fraction_per_step) == 0.0:
            raise ValueError("max_fraction_per_step must be in (0, 1]")
        _finite_number(self.distance_weight, "distance_weight", minimum=0.0)
        _positive_finite_number(
            self.max_target_separation_km,
            "max_target_separation_km",
        )
        _positive_finite_number(
            self.max_turn_angle_degrees,
            "max_turn_angle_degrees",
        )
        if float(self.max_turn_angle_degrees) > 180.0:
            raise ValueError("max_turn_angle_degrees must be at most 180")
        _positive_finite_number(
            self.max_candidate_distance_km,
            "max_candidate_distance_km",
        )
        _positive_int(
            self.max_per_old_target_per_step,
            "max_per_old_target_per_step",
        )


class DynamicRetargetSkill:
    """Select bounded semantic retarget requests within one communication group."""

    def __init__(self, config: DynamicRetargetConfig | None = None) -> None:
        if config is not None and not isinstance(config, DynamicRetargetConfig):
            raise ValueError("config must be a DynamicRetargetConfig")
        self.config = config or DynamicRetargetConfig()

    def decide(
        self,
        *,
        step: int,
        members: Sequence[RetargetMemberSnapshot],
        targets: Sequence[RetargetTargetSnapshot],
    ) -> RetargetDecision:
        _nonnegative_int(step, "step")
        ordered_members = self._validate_members(members)
        ordered_targets = self._validate_targets(targets)
        targets_by_id = {target.target_id: target for target in ordered_targets}

        eligible: list[
            tuple[RetargetMemberSnapshot, RetargetTargetSnapshot, tuple[RetargetTargetSnapshot, ...]]
        ] = []
        rejected_reasons: dict[int, str] = {}
        for member in ordered_members:
            current = targets_by_id.get(member.current_target_id)
            reason, candidates = self._evaluate_eligibility(
                step,
                member,
                current,
                ordered_targets,
            )
            if reason != "eligible":
                rejected_reasons[member.entity_id] = reason
                continue
            assert current is not None
            eligible.append((member, current, candidates))

        eligible_ids = tuple(member.entity_id for member, _, _ in eligible)
        if not eligible:
            return RetargetDecision((), eligible_ids, 0, rejected_reasons)

        budget = max(
            1,
            ceil(len(eligible) * float(self.config.max_fraction_per_step)),
        )
        effective_assigned = {
            target.target_id: target.assigned_count for target in ordered_targets
        }
        requests: list[RetargetRequest] = []
        selected_ids: set[int] = set()
        selected_by_old_target: dict[int, int] = {}
        for member, current, candidates in eligible:
            if len(requests) >= budget:
                break
            selected_count = selected_by_old_target.get(member.current_target_id, 0)
            if selected_count >= self.config.max_per_old_target_per_step:
                continue
            destination = min(
                candidates,
                key=lambda target: self._candidate_key(
                    member,
                    target,
                    effective_assigned[target.target_id],
                ),
            )
            effective_assigned[destination.target_id] += 1
            selected_ids.add(member.entity_id)
            selected_by_old_target[member.current_target_id] = selected_count + 1
            requests.append(RetargetRequest(
                entity_id=member.entity_id,
                old_target_id=member.current_target_id,
                new_target_id=destination.target_id,
                reason="current_target_delivered_destruction_threshold_met",
                target_destruction_probability=current.destruction_probability,
            ))
        for member, _, _ in eligible:
            if member.entity_id in selected_ids:
                continue
            if (
                selected_by_old_target.get(member.current_target_id, 0)
                >= self.config.max_per_old_target_per_step
            ):
                rejected_reasons[member.entity_id] = "old_target_step_limit"
            else:
                rejected_reasons[member.entity_id] = "step_budget"
        return RetargetDecision(
            tuple(requests),
            eligible_ids,
            budget,
            rejected_reasons,
        )

    def eligibility_reason(
        self,
        member: RetargetMemberSnapshot,
        current: RetargetTargetSnapshot | None,
        targets: Sequence[RetargetTargetSnapshot],
        *,
        step: int,
    ) -> str:
        """Return the deterministic first reason preventing a retarget."""
        _nonnegative_int(step, "step")
        if not isinstance(member, RetargetMemberSnapshot):
            raise ValueError("member must be a RetargetMemberSnapshot")
        if current is not None and not isinstance(current, RetargetTargetSnapshot):
            raise ValueError("current must be a RetargetTargetSnapshot or None")
        ordered_targets = self._validate_targets(targets)
        reason, _ = self._evaluate_eligibility(
            step,
            member,
            current,
            ordered_targets,
        )
        return reason

    @staticmethod
    def _validate_members(
        members: Sequence[RetargetMemberSnapshot],
    ) -> tuple[RetargetMemberSnapshot, ...]:
        materialized = tuple(members)
        if any(not isinstance(item, RetargetMemberSnapshot) for item in materialized):
            raise ValueError("members must contain RetargetMemberSnapshot values")
        ordered = tuple(sorted(materialized, key=lambda item: item.entity_id))
        ids = tuple(item.entity_id for item in ordered)
        if len(ids) != len(set(ids)):
            raise ValueError("members must have unique entity IDs")
        groups = {item.communication_group for item in ordered}
        if len(groups) > 1:
            raise ValueError("members must belong to one communication_group")
        return ordered

    @staticmethod
    def _validate_targets(
        targets: Sequence[RetargetTargetSnapshot],
    ) -> tuple[RetargetTargetSnapshot, ...]:
        materialized = tuple(targets)
        if any(not isinstance(item, RetargetTargetSnapshot) for item in materialized):
            raise ValueError("targets must contain RetargetTargetSnapshot values")
        ordered = tuple(sorted(materialized, key=lambda item: item.target_id))
        ids = tuple(item.target_id for item in ordered)
        if len(ids) != len(set(ids)):
            raise ValueError("targets must have unique target IDs")
        return ordered

    def _evaluate_eligibility(
        self,
        step: int,
        member: RetargetMemberSnapshot,
        current: RetargetTargetSnapshot | None,
        targets: tuple[RetargetTargetSnapshot, ...],
    ) -> tuple[str, tuple[RetargetTargetSnapshot, ...]]:
        if current is None:
            return "target_probability_below_threshold", ()
        if not member.alive:
            return "member_not_alive", ()
        if member.is_core:
            return "core_member", ()
        if member.stage > self.config.max_stage:
            return "stage_too_late", ()
        if member.current_distance_km < self.config.min_current_distance_km:
            return "too_close", ()
        if member.retarget_count >= self.config.max_retargets:
            return "max_retargets", ()
        if (
            member.last_retarget_step is not None
            and step - member.last_retarget_step < self.config.cooldown_steps
        ):
            return "cooldown", ()
        if current.destruction_probability < self.config.destruction_threshold:
            return "target_probability_below_threshold", ()

        candidates = tuple(
            target
            for target in targets
            if target.target_id != member.current_target_id
            and target.target_id in member.legal_target_ids
        )
        if not candidates:
            return "no_eligible_candidate", ()
        candidates = tuple(
            target
            for target in candidates
            if target.target_id in member.fresh_target_ids
        )
        if not candidates:
            return "candidate_track_stale", ()
        candidates = tuple(
            target
            for target in candidates
            if target.destruction_probability < self.config.destruction_threshold
            and target.projected_probability < self.config.destruction_threshold
        )
        if not candidates:
            return "no_eligible_candidate", ()
        candidates = tuple(
            target
            for target in candidates
            if _wgs84_distance_km(current.lon, current.lat, target.lon, target.lat)
            <= self.config.max_target_separation_km
        )
        if not candidates:
            return "target_separation", ()
        candidates = tuple(
            target
            for target in candidates
            if _wgs84_distance_km(member.lon, member.lat, target.lon, target.lat)
            <= self.config.max_candidate_distance_km
        )
        if not candidates:
            return "candidate_too_far", ()
        candidates = tuple(
            target
            for target in candidates
            if self._turn_angle_degrees(member, target)
            <= self.config.max_turn_angle_degrees
        )
        if not candidates:
            return "turn_angle", ()
        return "eligible", candidates

    @staticmethod
    def _turn_angle_degrees(
        member: RetargetMemberSnapshot,
        target: RetargetTargetSnapshot,
    ) -> float:
        if member.velocity_x is None or member.velocity_y is None:
            return float("inf")
        velocity_norm = hypot(member.velocity_x, member.velocity_y)
        if velocity_norm <= 1e-12:
            return float("inf")
        longitude_delta = ((target.lon - member.lon + 180.0) % 360.0) - 180.0
        mean_latitude = radians((member.lat + target.lat) / 2.0)
        direction_x = radians(longitude_delta) * cos(mean_latitude)
        direction_y = radians(target.lat - member.lat)
        direction_norm = hypot(direction_x, direction_y)
        if direction_norm <= 1e-15:
            return 0.0
        cosine = (
            member.velocity_x * direction_x + member.velocity_y * direction_y
        ) / (velocity_norm * direction_norm)
        return degrees(acos(min(1.0, max(-1.0, cosine))))

    def _candidate_key(
        self,
        member: RetargetMemberSnapshot,
        target: RetargetTargetSnapshot,
        assigned_count: int,
    ) -> tuple[float, float, float, int, int]:
        shortfall = max(
            0.0,
            float(self.config.destruction_threshold - target.projected_probability),
        )
        distance_km = _wgs84_distance_km(member.lon, member.lat, target.lon, target.lat)
        desirability = (
            (1.0 - float(target.destruction_probability))
            * (1.0 + float(target.value))
            * (1.0 + shortfall)
            / (1.0 + assigned_count)
            - float(self.config.distance_weight) * distance_km
        )
        return (
            -desirability,
            distance_km,
            float(target.destruction_probability),
            assigned_count,
            target.target_id,
        )


def _wgs84_distance_km(
    first_lon: float,
    first_lat: float,
    second_lon: float,
    second_lat: float,
) -> float:
    mean_radius_km = 6_371.0088
    first_latitude = radians(first_lat)
    second_latitude = radians(second_lat)
    longitude_delta = ((second_lon - first_lon + 180.0) % 360.0) - 180.0
    sin_lat = sin((second_latitude - first_latitude) / 2.0)
    sin_lon = sin(radians(longitude_delta) / 2.0)
    haversine = (
        sin_lat * sin_lat
        + cos(first_latitude) * cos(second_latitude) * sin_lon * sin_lon
    )
    return 2.0 * mean_radius_km * asin(sqrt(min(1.0, max(0.0, haversine))))
