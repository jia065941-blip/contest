"""Pure deterministic second-echelon reserve dispatch state machine."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from math import acos, degrees, hypot, isfinite
from types import MappingProxyType


H_MISSILE_TYPE = 21000
M_MISSILE_TYPE = 21001
L_MISSILE_TYPE = 21002
SUPPORTED_MEMBER_TYPES = frozenset(
    (H_MISSILE_TYPE, M_MISSILE_TYPE, L_MISSILE_TYPE)
)
PUBLIC_TARGET_TYPES = frozenset((9400, 9600))
HIDDEN_NEST_TYPE = 9500
SUPPORTED_TARGET_TYPES = PUBLIC_TARGET_TYPES | {HIDDEN_NEST_TYPE}
PACKAGE_STATES = frozenset(("STAGING", "DISPATCHED", "FALLBACK"))


def _nonnegative_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _finite_number(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    positive: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and numeric < minimum:
        raise ValueError(f"{name} must be at least {minimum:g}")
    if positive and numeric <= 0.0:
        raise ValueError(f"{name} must be positive")


def _identifier(value: object, name: str) -> None:
    _nonnegative_int(value, name)


def _hashable_value(value: object, name: str) -> None:
    if value is None or isinstance(value, bool) or not isinstance(value, Hashable):
        raise ValueError(f"{name} must be a non-null hashable value")
    if isinstance(value, int) and value < 0:
        raise ValueError(f"{name} integer must be nonnegative")
    if isinstance(value, str) and not value:
        raise ValueError(f"{name} string must not be empty")


def _position(value: object, name: str) -> None:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must be a two-number tuple")
    _finite_number(value[0], f"{name}[0]")
    _finite_number(value[1], f"{name}[1]")


def _unique_ids(value: object, name: str, *, allow_empty: bool = True) -> None:
    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be a tuple")
    if not allow_empty and not value:
        raise ValueError(f"{name} must not be empty")
    for item in value:
        _identifier(item, f"{name} item")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} must contain unique IDs")


@dataclass(frozen=True)
class ReserveMemberSnapshot:
    """One reserve member and the targets its owner may legally command."""

    id: int
    type: int
    stage: int
    group: Hashable
    current_target: int
    legal_target_ids: tuple[int, ...]
    position: tuple[float, float]
    retarget_count: int
    heading_east: float | None = None
    heading_north: float | None = None
    remaining_steps: int | None = None

    def __post_init__(self) -> None:
        _identifier(self.id, "id")
        if (
            isinstance(self.type, bool)
            or not isinstance(self.type, int)
            or self.type not in SUPPORTED_MEMBER_TYPES
        ):
            raise ValueError("type must be H=21000, M=21001, or L=21002")
        _nonnegative_int(self.stage, "stage")
        _hashable_value(self.group, "group")
        _identifier(self.current_target, "current_target")
        _unique_ids(self.legal_target_ids, "legal_target_ids")
        _position(self.position, "position")
        _nonnegative_int(self.retarget_count, "retarget_count")
        if (self.heading_east is None) != (self.heading_north is None):
            raise ValueError(
                "heading_east and heading_north must both be set or both be None"
            )
        if self.heading_east is not None:
            _finite_number(self.heading_east, "heading_east")
            _finite_number(self.heading_north, "heading_north")
        if self.remaining_steps is not None:
            _nonnegative_int(self.remaining_steps, "remaining_steps")


@dataclass(frozen=True)
class TargetDeficitSnapshot:
    """Caller-authorized target deficit and its evidence provenance."""

    target_id: int
    type: int
    weight: float
    missing_damage: float
    priority: float
    position: tuple[float, float]
    responsibility_region: Hashable
    evidence_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _identifier(self.target_id, "target_id")
        if (
            isinstance(self.type, bool)
            or not isinstance(self.type, int)
            or self.type not in SUPPORTED_TARGET_TYPES
        ):
            raise ValueError("type must be public 9400/9600 or hidden 9500")
        _finite_number(self.weight, "weight", positive=True)
        _finite_number(self.missing_damage, "missing_damage", positive=True)
        _finite_number(self.priority, "priority", minimum=0.0)
        _position(self.position, "position")
        _hashable_value(self.responsibility_region, "responsibility_region")
        _unique_ids(self.evidence_ids, "evidence_ids")


@dataclass(frozen=True)
class ReservePackageSnapshot:
    """Immutable definition and lifecycle snapshot for an atomic reserve package."""

    package_id: int
    member_ids: tuple[int, ...]
    member_types: tuple[int, ...]
    state: str
    provisional_target_id: int
    fallback_target_id: int
    responsibility_region: Hashable
    dispatch_count: int
    fallback_position: tuple[float, float] | None = None
    target_id: int | None = None
    retarget_count: int = 0

    def __post_init__(self) -> None:
        _identifier(self.package_id, "package_id")
        _unique_ids(self.member_ids, "member_ids", allow_empty=False)
        if not isinstance(self.member_types, tuple):
            raise ValueError("member_types must be a tuple")
        if len(self.member_types) != len(self.member_ids):
            raise ValueError("member_types must align with member_ids")
        for member_type in self.member_types:
            if (
                isinstance(member_type, bool)
                or not isinstance(member_type, int)
                or member_type not in SUPPORTED_MEMBER_TYPES
            ):
                raise ValueError(
                    "member_types items must be H=21000, M=21001, or L=21002"
                )
        if not isinstance(self.state, str) or self.state not in PACKAGE_STATES:
            raise ValueError("state must be STAGING, DISPATCHED, or FALLBACK")
        _identifier(self.provisional_target_id, "provisional_target_id")
        _identifier(self.fallback_target_id, "fallback_target_id")
        if self.provisional_target_id == self.fallback_target_id:
            raise ValueError("fallback_target_id must differ from provisional_target_id")
        _hashable_value(self.responsibility_region, "responsibility_region")
        _nonnegative_int(self.dispatch_count, "dispatch_count")
        if self.fallback_position is not None:
            _position(self.fallback_position, "fallback_position")
        if self.target_id is not None:
            _identifier(self.target_id, "target_id")
        _nonnegative_int(self.retarget_count, "retarget_count")
        if self.state == "STAGING" and self.dispatch_count != 0:
            raise ValueError("STAGING packages must have dispatch_count=0")
        if self.state != "STAGING" and self.dispatch_count == 0:
            raise ValueError("terminal packages must have a positive dispatch_count")
        if self.state == "STAGING" and self.target_id is not None:
            raise ValueError("STAGING packages must not have target_id")


@dataclass(frozen=True)
class ReachabilityDiagnostics:
    """Conservative per-package reachability timing, in km/degrees/seconds."""

    distance: float
    turn_angle: float | None
    turn_time: float | None
    travel_time: float
    total_eta: float | None
    remaining_time: float | None

    def __post_init__(self) -> None:
        _finite_number(self.distance, "distance", minimum=0.0)
        _finite_number(self.travel_time, "travel_time", minimum=0.0)
        for name in ("turn_angle", "turn_time", "total_eta", "remaining_time"):
            value = getattr(self, name)
            if value is not None:
                _finite_number(value, name, minimum=0.0)
        if (self.turn_angle is None) != (self.turn_time is None):
            raise ValueError("turn_angle and turn_time must both be set or both be None")
        if (self.turn_time is None) != (self.total_eta is None):
            raise ValueError("turn_time and total_eta must both be set or both be None")


@dataclass(frozen=True)
class ReserveDispatchRequest:
    """One indivisible package-level target change."""

    member_ids: tuple[int, ...]
    old_target_id: int
    new_target_id: int
    reason: str
    deficit: TargetDeficitSnapshot | None
    diagnostics: ReachabilityDiagnostics

    def __post_init__(self) -> None:
        _unique_ids(self.member_ids, "member_ids", allow_empty=False)
        _identifier(self.old_target_id, "old_target_id")
        _identifier(self.new_target_id, "new_target_id")
        if self.old_target_id == self.new_target_id:
            raise ValueError("new_target_id must differ from old_target_id")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a nonempty string")
        if self.deficit is not None and not isinstance(
            self.deficit, TargetDeficitSnapshot
        ):
            raise ValueError("deficit must be a TargetDeficitSnapshot or None")
        if not isinstance(self.diagnostics, ReachabilityDiagnostics):
            raise ValueError("diagnostics must be ReachabilityDiagnostics")


@dataclass(frozen=True)
class Decision:
    """Dispatches and complete persisted package state after one decision."""

    requests: tuple[ReserveDispatchRequest, ...]
    packages: tuple[ReservePackageSnapshot, ...]
    rejected_reasons: Mapping[int, str] = field(default_factory=dict)
    diagnostics: Mapping[int, ReachabilityDiagnostics] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.requests, tuple) or any(
            not isinstance(item, ReserveDispatchRequest) for item in self.requests
        ):
            raise ValueError("requests must contain ReserveDispatchRequest values")
        if not isinstance(self.packages, tuple) or any(
            not isinstance(item, ReservePackageSnapshot) for item in self.packages
        ):
            raise ValueError("packages must contain ReservePackageSnapshot values")
        if not isinstance(self.rejected_reasons, Mapping):
            raise ValueError("rejected_reasons must be a mapping")
        reasons = dict(self.rejected_reasons)
        for package_id, reason in reasons.items():
            _identifier(package_id, "rejected package_id")
            if not isinstance(reason, str) or not reason:
                raise ValueError("rejected reason must be a nonempty string")
        if not isinstance(self.diagnostics, Mapping):
            raise ValueError("diagnostics must be a mapping")
        diagnostics = dict(self.diagnostics)
        for package_id, item in diagnostics.items():
            _identifier(package_id, "diagnostic package_id")
            if not isinstance(item, ReachabilityDiagnostics):
                raise ValueError(
                    "diagnostics values must be ReachabilityDiagnostics"
                )
        object.__setattr__(self, "rejected_reasons", MappingProxyType(reasons))
        object.__setattr__(self, "diagnostics", MappingProxyType(diagnostics))


ReserveDecision = Decision


@dataclass(frozen=True)
class ReserveEchelonConfig:
    max_dispatch_stage: int | None = None
    max_dispatches: int = 1
    deadline_step: int = 300
    require_same_group_evidence: bool = True
    max_target_distance_km: float | None = None
    h_max_distance_km: float = 80.0
    l_max_distance_km: float = 100.0
    h_max_turn_angle_degrees: float | None = None
    l_max_turn_angle_degrees: float | None = None
    seconds_per_step: float = 1.0
    h_nominal_speed_mps: float = 200.0
    l_nominal_speed_mps: float = 100.0
    h_nominal_turn_rate_deg_s: float = 20.0
    l_nominal_turn_rate_deg_s: float = 15.0
    turn_safety_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.max_dispatch_stage is not None:
            _nonnegative_int(self.max_dispatch_stage, "max_dispatch_stage")
        _positive_int(self.max_dispatches, "max_dispatches")
        _nonnegative_int(self.deadline_step, "deadline_step")
        if not isinstance(self.require_same_group_evidence, bool):
            raise ValueError("require_same_group_evidence must be a boolean")
        if self.max_target_distance_km is not None:
            _finite_number(
                self.max_target_distance_km,
                "max_target_distance_km",
                positive=True,
            )
        for name in (
            "h_max_distance_km",
            "l_max_distance_km",
            "seconds_per_step",
            "h_nominal_speed_mps",
            "l_nominal_speed_mps",
            "h_nominal_turn_rate_deg_s",
            "l_nominal_turn_rate_deg_s",
        ):
            _finite_number(getattr(self, name), name, positive=True)
        for name in (
            "h_max_turn_angle_degrees",
            "l_max_turn_angle_degrees",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            _finite_number(value, name, positive=True)
            if float(value) > 180.0:
                raise ValueError(f"{name} must be at most 180")
        _finite_number(
            self.turn_safety_seconds,
            "turn_safety_seconds",
            minimum=0.0,
        )


Config = ReserveEchelonConfig


class ReserveEchelonSkill:
    """Persist and dispatch H-only or L-only reserve packages atomically."""

    def __init__(self, config: ReserveEchelonConfig | None = None) -> None:
        if config is not None and not isinstance(config, ReserveEchelonConfig):
            raise ValueError("config must be a ReserveEchelonConfig")
        self.config = config or ReserveEchelonConfig()
        self._packages: dict[int, ReservePackageSnapshot] = {}
        self._definitions: dict[int, tuple[object, ...]] = {}

    @property
    def state(self) -> tuple[ReservePackageSnapshot, ...]:
        """Return the current immutable package states in deterministic order."""
        return tuple(self._packages[key] for key in sorted(self._packages))

    def reset(self) -> None:
        """Forget all registered packages and their terminal states."""
        self._packages.clear()
        self._definitions.clear()

    def decide(
        self,
        *,
        step: int,
        members: Sequence[ReserveMemberSnapshot],
        deficits: Sequence[TargetDeficitSnapshot],
        packages: Sequence[ReservePackageSnapshot],
    ) -> Decision:
        _nonnegative_int(step, "step")
        ordered_members = self._validate_members(members)
        ordered_deficits = self._validate_deficits(deficits)
        ordered_packages = self._validate_packages(packages)
        members_by_id = {item.id: item for item in ordered_members}
        self._register_packages(ordered_packages)
        effective_packages = tuple(
            self._packages[item.package_id] for item in ordered_packages
        )
        self._validate_package_members(effective_packages, members_by_id)

        requests: list[ReserveDispatchRequest] = []
        rejected: dict[int, str] = {}
        diagnostics: dict[int, ReachabilityDiagnostics] = {}
        for supplied in ordered_packages:
            package = self._packages[supplied.package_id]
            package_members = tuple(
                members_by_id[member_id]
                for member_id in package.member_ids
                if member_id in members_by_id
            )
            if not package_members:
                rejected[package.package_id] = "no_surviving_members"
                continue
            if package.state == "DISPATCHED":
                rejected[package.package_id] = "already_dispatched"
                continue
            if package.state == "FALLBACK":
                rejected[package.package_id] = "fallback_committed"
                continue
            if package.dispatch_count >= self.config.max_dispatches or any(
                member.retarget_count >= self.config.max_dispatches
                for member in package_members
            ):
                rejected[package.package_id] = "max_dispatches"
                continue
            if self.config.max_dispatch_stage is not None and any(
                member.stage > self.config.max_dispatch_stage
                for member in package_members
            ):
                rejected[package.package_id] = "stage_too_late"
                continue

            logical_candidates = tuple(
                deficit
                for deficit in ordered_deficits
                if self._logically_eligible_deficit(
                    package,
                    package_members,
                    deficit,
                    members_by_id,
                )
            )
            assessed_candidates = tuple(
                (
                    deficit,
                    *self._reachability_assessment(
                        package_members,
                        deficit.position,
                    ),
                )
                for deficit in logical_candidates
            )
            candidates = tuple(
                (deficit, diagnostic)
                for deficit, reason, diagnostic in assessed_candidates
                if reason is None
            )
            if candidates:
                chosen, diagnostic = min(
                    candidates,
                    key=lambda item: self._candidate_key(
                        package_members, item[0]
                    ),
                )
                request = ReserveDispatchRequest(
                    member_ids=tuple(member.id for member in package_members),
                    old_target_id=package.provisional_target_id,
                    new_target_id=chosen.target_id,
                    reason="target_deficit",
                    deficit=chosen,
                    diagnostics=diagnostic,
                )
                diagnostics[package.package_id] = diagnostic
                self._packages[package.package_id] = replace(
                    package,
                    state="DISPATCHED",
                    dispatch_count=package.dispatch_count + 1,
                    target_id=chosen.target_id,
                )
                requests.append(request)
                continue

            if step >= self.config.deadline_step:
                (
                    fallback_reason,
                    diagnostic,
                ) = self._fallback_reachability_assessment(
                    package, package_members, ordered_deficits
                )
                if fallback_reason is not None:
                    rejected[package.package_id] = fallback_reason
                    if diagnostic is not None:
                        diagnostics[package.package_id] = diagnostic
                    continue
                assert diagnostic is not None
                request = ReserveDispatchRequest(
                    member_ids=tuple(member.id for member in package_members),
                    old_target_id=package.provisional_target_id,
                    new_target_id=package.fallback_target_id,
                    reason="deadline_no_deficit_fallback",
                    deficit=None,
                    diagnostics=diagnostic,
                )
                diagnostics[package.package_id] = diagnostic
                self._packages[package.package_id] = replace(
                    package,
                    state="FALLBACK",
                    dispatch_count=package.dispatch_count + 1,
                    target_id=package.fallback_target_id,
                )
                requests.append(request)
            elif logical_candidates:
                _, reason, diagnostic = min(
                    assessed_candidates,
                    key=lambda item: self._candidate_key(
                        package_members, item[0]
                    ),
                )
                assert reason is not None
                rejected[package.package_id] = reason
                diagnostics[package.package_id] = diagnostic
            elif ordered_deficits:
                rejected[package.package_id] = "no_eligible_deficit"
            else:
                rejected[package.package_id] = "no_deficit"

        return Decision(tuple(requests), self.state, rejected, diagnostics)

    @staticmethod
    def _materialize(values: object, name: str) -> tuple[object, ...]:
        if isinstance(values, (str, bytes)):
            raise ValueError(f"{name} must be a sequence")
        try:
            return tuple(values)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(f"{name} must be a sequence") from exc

    @classmethod
    def _validate_members(
        cls, members: Sequence[ReserveMemberSnapshot]
    ) -> tuple[ReserveMemberSnapshot, ...]:
        materialized = cls._materialize(members, "members")
        if any(not isinstance(item, ReserveMemberSnapshot) for item in materialized):
            raise ValueError("members must contain ReserveMemberSnapshot values")
        ordered = tuple(sorted(materialized, key=lambda item: item.id))
        ids = tuple(item.id for item in ordered)
        if len(ids) != len(set(ids)):
            raise ValueError("members must have unique IDs")
        return ordered

    @classmethod
    def _validate_deficits(
        cls, deficits: Sequence[TargetDeficitSnapshot]
    ) -> tuple[TargetDeficitSnapshot, ...]:
        materialized = cls._materialize(deficits, "deficits")
        if any(not isinstance(item, TargetDeficitSnapshot) for item in materialized):
            raise ValueError("deficits must contain TargetDeficitSnapshot values")
        ordered = tuple(sorted(materialized, key=lambda item: item.target_id))
        ids = tuple(item.target_id for item in ordered)
        if len(ids) != len(set(ids)):
            raise ValueError("deficits must have unique target IDs")
        return ordered

    @classmethod
    def _validate_packages(
        cls, packages: Sequence[ReservePackageSnapshot]
    ) -> tuple[ReservePackageSnapshot, ...]:
        materialized = cls._materialize(packages, "packages")
        if any(not isinstance(item, ReservePackageSnapshot) for item in materialized):
            raise ValueError("packages must contain ReservePackageSnapshot values")
        ordered = tuple(sorted(materialized, key=lambda item: item.package_id))
        ids = tuple(item.package_id for item in ordered)
        if len(ids) != len(set(ids)):
            raise ValueError("packages must have unique package IDs")
        member_ids = tuple(
            member_id for package in ordered for member_id in package.member_ids
        )
        if len(member_ids) != len(set(member_ids)):
            raise ValueError("a member may belong to only one package per decision")
        return ordered

    @staticmethod
    def _definition(package: ReservePackageSnapshot) -> tuple[object, ...]:
        return (
            package.member_ids,
            package.member_types,
            package.provisional_target_id,
            package.fallback_target_id,
            package.responsibility_region,
            package.fallback_position,
        )

    def _register_packages(
        self, packages: tuple[ReservePackageSnapshot, ...]
    ) -> None:
        for package in packages:
            definition = self._definition(package)
            known = self._definitions.get(package.package_id)
            if known is not None and known != definition:
                raise ValueError(
                    f"package {package.package_id} definition cannot change"
                )
            if known is None:
                self._definitions[package.package_id] = definition
                self._packages[package.package_id] = package

    @staticmethod
    def _validate_package_members(
        packages: tuple[ReservePackageSnapshot, ...],
        members_by_id: Mapping[int, ReserveMemberSnapshot],
    ) -> None:
        for package in packages:
            members = tuple(
                members_by_id[member_id]
                for member_id in package.member_ids
                if member_id in members_by_id
            )
            if not members:
                continue
            expected_types = dict(zip(
                package.member_ids, package.member_types, strict=True
            ))
            actual_types = tuple(member.type for member in members)
            if any(
                member.type != expected_types[member.id] for member in members
            ):
                raise ValueError(
                    f"package {package.package_id} member_types do not match members"
                )
            unique_types = frozenset(actual_types)
            if unique_types not in (
                frozenset((H_MISSILE_TYPE,)),
                frozenset((L_MISSILE_TYPE,)),
            ):
                raise ValueError(
                    "version 0.4 supports only H-only and L-only packages"
                )
            groups = {member.group for member in members}
            if len(groups) != 1:
                raise ValueError(
                    f"package {package.package_id} members must share one group"
                )
            if package.state == "STAGING" and any(
                member.current_target != package.provisional_target_id
                for member in members
            ):
                raise ValueError(
                    f"package {package.package_id} STAGING members must use "
                    "provisional_target_id"
                )
            if any(
                package.fallback_target_id not in member.legal_target_ids
                for member in members
            ):
                raise ValueError(
                    f"package {package.package_id} fallback target must be legal "
                    "for every member"
                )

    def _logically_eligible_deficit(
        self,
        package: ReservePackageSnapshot,
        members: tuple[ReserveMemberSnapshot, ...],
        deficit: TargetDeficitSnapshot,
        members_by_id: Mapping[int, ReserveMemberSnapshot],
    ) -> bool:
        if deficit.responsibility_region != package.responsibility_region:
            return False
        if deficit.target_id == package.provisional_target_id:
            return False
        if any(deficit.target_id not in member.legal_target_ids for member in members):
            return False
        package_type = members[0].type
        if package_type == H_MISSILE_TYPE and deficit.type not in PUBLIC_TARGET_TYPES:
            return False
        if package_type == L_MISSILE_TYPE and deficit.type != HIDDEN_NEST_TYPE:
            return False
        if self.config.require_same_group_evidence:
            group = members[0].group
            if not any(
                evidence_id in members_by_id
                and members_by_id[evidence_id].group == group
                for evidence_id in deficit.evidence_ids
            ):
                return False
        return True

    def _fallback_reachability_assessment(
        self,
        package: ReservePackageSnapshot,
        members: tuple[ReserveMemberSnapshot, ...],
        deficits: tuple[TargetDeficitSnapshot, ...],
    ) -> tuple[str | None, ReachabilityDiagnostics | None]:
        position = package.fallback_position
        if position is None:
            position = next(
                (
                    deficit.position
                    for deficit in deficits
                    if deficit.target_id == package.fallback_target_id
                ),
                None,
            )
        if position is None:
            return "fallback_position_unavailable", None
        return self._reachability_assessment(members, position)

    def _reachability_assessment(
        self,
        members: tuple[ReserveMemberSnapshot, ...],
        target_position: tuple[float, float],
    ) -> tuple[str | None, ReachabilityDiagnostics]:
        assessed = tuple(
            (
                member,
                self._member_reachability_diagnostics(member, target_position),
            )
            for member in members
        )

        distance_failures = []
        for member, diagnostic in assessed:
            distance_limit = self._distance_limit_km(member.type)
            if self.config.max_target_distance_km is not None:
                distance_limit = min(
                    distance_limit,
                    float(self.config.max_target_distance_km),
                )
            if diagnostic.distance > distance_limit:
                distance_failures.append(
                    (diagnostic.distance - distance_limit, diagnostic)
                )
        if distance_failures:
            return (
                "distance_exceeded",
                max(distance_failures, key=lambda item: item[0])[1],
            )

        missing_heading = tuple(
            diagnostic
            for _, diagnostic in assessed
            if diagnostic.turn_angle is None
        )
        if missing_heading:
            return "missing_heading", missing_heading[0]

        turn_failures = []
        for member, diagnostic in assessed:
            turn_limit = self._turn_limit_degrees(member.type)
            if (
                turn_limit is not None
                and diagnostic.turn_angle is not None
                and diagnostic.turn_angle > turn_limit
            ):
                turn_failures.append(
                    (diagnostic.turn_angle - turn_limit, diagnostic)
                )
        if turn_failures:
            return (
                "turn_angle_exceeded",
                max(turn_failures, key=lambda item: item[0])[1],
            )

        time_failures = tuple(
            (
                diagnostic.total_eta - diagnostic.remaining_time,
                diagnostic,
            )
            for _, diagnostic in assessed
            if diagnostic.total_eta is not None
            and diagnostic.remaining_time is not None
            and diagnostic.total_eta > diagnostic.remaining_time
        )
        if time_failures:
            return (
                "insufficient_time",
                max(time_failures, key=lambda item: item[0])[1],
            )

        return None, max(
            (diagnostic for _, diagnostic in assessed),
            key=lambda diagnostic: (
                -1.0 if diagnostic.total_eta is None else diagnostic.total_eta
            ),
        )

    def _member_reachability_diagnostics(
        self,
        member: ReserveMemberSnapshot,
        target_position: tuple[float, float],
    ) -> ReachabilityDiagnostics:
        direction_east = float(target_position[0]) - float(member.position[0])
        direction_north = float(target_position[1]) - float(member.position[1])
        distance = hypot(direction_east, direction_north)
        travel_time = distance * 1000.0 / self._nominal_speed_mps(member.type)
        remaining_time = (
            None
            if member.remaining_steps is None
            else member.remaining_steps * float(self.config.seconds_per_step)
        )

        if distance <= 1e-12:
            turn_angle = 0.0
        elif member.heading_east is None or member.heading_north is None:
            turn_angle = None
        else:
            heading_norm = hypot(member.heading_east, member.heading_north)
            if heading_norm <= 1e-12:
                turn_angle = None
            else:
                cosine = (
                    member.heading_east * direction_east
                    + member.heading_north * direction_north
                ) / (heading_norm * distance)
                turn_angle = degrees(acos(min(1.0, max(-1.0, cosine))))

        turn_time = (
            None
            if turn_angle is None
            else abs(turn_angle) / self._nominal_turn_rate_deg_s(member.type)
        )
        total_eta = (
            None
            if turn_time is None
            else turn_time + travel_time + float(self.config.turn_safety_seconds)
        )
        return ReachabilityDiagnostics(
            distance=distance,
            turn_angle=turn_angle,
            turn_time=turn_time,
            travel_time=travel_time,
            total_eta=total_eta,
            remaining_time=remaining_time,
        )

    def _distance_limit_km(self, member_type: int) -> float:
        if member_type == H_MISSILE_TYPE:
            return float(self.config.h_max_distance_km)
        return float(self.config.l_max_distance_km)

    def _turn_limit_degrees(self, member_type: int) -> float | None:
        if member_type == H_MISSILE_TYPE:
            value = self.config.h_max_turn_angle_degrees
        else:
            value = self.config.l_max_turn_angle_degrees
        return None if value is None else float(value)

    def _nominal_speed_mps(self, member_type: int) -> float:
        if member_type == H_MISSILE_TYPE:
            return float(self.config.h_nominal_speed_mps)
        return float(self.config.l_nominal_speed_mps)

    def _nominal_turn_rate_deg_s(self, member_type: int) -> float:
        if member_type == H_MISSILE_TYPE:
            return float(self.config.h_nominal_turn_rate_deg_s)
        return float(self.config.l_nominal_turn_rate_deg_s)

    @staticmethod
    def _distance_to_deficit(
        members: tuple[ReserveMemberSnapshot, ...],
        deficit: TargetDeficitSnapshot,
    ) -> float:
        center_x = sum(float(member.position[0]) for member in members) / len(members)
        center_y = sum(float(member.position[1]) for member in members) / len(members)
        return hypot(
            float(deficit.position[0]) - center_x,
            float(deficit.position[1]) - center_y,
        )

    def _candidate_key(
        self,
        members: tuple[ReserveMemberSnapshot, ...],
        deficit: TargetDeficitSnapshot,
    ) -> tuple[float, float, float, int]:
        distance = self._distance_to_deficit(members, deficit)
        score = (
            float(deficit.weight)
            * float(deficit.missing_damage)
            / (1.0 + distance)
        )
        return (-score, -float(deficit.priority), distance, deficit.target_id)
