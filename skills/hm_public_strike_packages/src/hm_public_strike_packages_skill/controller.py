"""Pure deterministic H/M allocation for legally known public targets."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from typing import Sequence


ACTION_SET_ACC_Z = 0
ACTION_ATTACK_TARGET = 4
H_MISSILE_TYPE = 21000
M_MISSILE_TYPE = 21001
PUBLIC_TARGET_TYPES = frozenset((9400, 9600))
_EPSILON = 1e-12


def _nonnegative_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _positive_finite(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    if not isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")


@dataclass(frozen=True)
class HMMissileSnapshot:
    entity_id: int
    missile_type: int
    alive: bool = True
    launched: bool = True
    baseline_action: int = 1

    def __post_init__(self) -> None:
        _nonnegative_int(self.entity_id, "H/M missile entity_id")
        if (
            isinstance(self.missile_type, bool)
            or not isinstance(self.missile_type, int)
            or self.missile_type not in (H_MISSILE_TYPE, M_MISSILE_TYPE)
        ):
            raise ValueError("missile_type must be H=21000 or M=21001")
        if not isinstance(self.alive, bool) or not isinstance(self.launched, bool):
            raise ValueError("alive and launched must be booleans")
        if isinstance(self.baseline_action, bool) or self.baseline_action not in (0, 1, 2):
            raise ValueError("baseline_action must be LEFT=0, STOP=1, or RIGHT=2")


@dataclass(frozen=True)
class PublicTargetTrack:
    target_id: int
    target_type: int
    lon: float
    lat: float
    observed_step: float
    initial_health: float
    value: float

    def __post_init__(self) -> None:
        _nonnegative_int(self.target_id, "Public target_id")
        if (
            isinstance(self.target_type, bool)
            or not isinstance(self.target_type, int)
            or self.target_type not in PUBLIC_TARGET_TYPES
        ):
            raise ValueError("Public target type must be 9400 or 9600")
        for field_name in ("lon", "lat", "observed_step"):
            field_value = getattr(self, field_name)
            if isinstance(field_value, bool) or not isinstance(field_value, (int, float)):
                raise ValueError(f"{field_name} must be finite")
            if not isfinite(float(field_value)):
                raise ValueError(f"{field_name} must be finite")
        if self.observed_step < 0:
            raise ValueError("observed_step must be nonnegative")
        _positive_finite(self.initial_health, "initial_health")
        _positive_finite(self.value, "value")


@dataclass(frozen=True)
class PublicStrikeConfig:
    h_damage: float = 20.0
    m_damage: float = 5.0
    hit_rate_9400: float = 0.8
    hit_rate_9600: float = 0.6
    health_9400: float = 24.0
    health_9600: float = 18.0
    value_9400: float = 5.0
    value_9600: float = 2.0
    safety_factor: float = 1.25
    reserve_h: int = 0
    reserve_m: int = 0
    lock_assignments: bool = True

    def __post_init__(self) -> None:
        for name in (
            "h_damage", "m_damage", "health_9400", "health_9600",
            "value_9400", "value_9600", "safety_factor",
        ):
            _positive_finite(getattr(self, name), name)
        for name in ("hit_rate_9400", "hit_rate_9600"):
            _positive_finite(getattr(self, name), name)
            if float(getattr(self, name)) > 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        _nonnegative_int(self.reserve_h, "reserve_h")
        _nonnegative_int(self.reserve_m, "reserve_m")
        if not isinstance(self.lock_assignments, bool):
            raise ValueError("lock_assignments must be a boolean")

    def default_health(self, target_type: int) -> float:
        _validate_target_type(target_type)
        return self.health_9400 if target_type == 9400 else self.health_9600

    def default_value(self, target_type: int) -> float:
        _validate_target_type(target_type)
        return self.value_9400 if target_type == 9400 else self.value_9600


HMPublicStrikeConfig = PublicStrikeConfig


@dataclass(frozen=True)
class PublicStrikePackage:
    target: PublicTargetTrack
    member_ids: tuple[int, ...]
    required_damage: float
    expected_damage: float


@dataclass(frozen=True)
class AttackTargetAction:
    entity_id: int
    target_id: int

    def __post_init__(self) -> None:
        _nonnegative_int(self.entity_id, "Attack entity_id")
        _nonnegative_int(self.target_id, "Attack target_id")


@dataclass(frozen=True)
class PublicStrikeDecision:
    baseline_actions: tuple[tuple[int, int], ...]
    attack_actions: tuple[AttackTargetAction, ...]
    packages: tuple[PublicStrikePackage, ...]
    public_targets: tuple[PublicTargetTrack, ...]
    unassigned_member_ids: tuple[int, ...]
    allocation_changed: bool


def expected_effective_damage(
    missile_type: int,
    target_type: int,
    config: PublicStrikeConfig | None = None,
) -> float:
    """Return configured nominal damage multiplied by public hit probability."""
    settings = config or PublicStrikeConfig()
    if (
        isinstance(missile_type, bool)
        or not isinstance(missile_type, int)
        or missile_type not in (H_MISSILE_TYPE, M_MISSILE_TYPE)
    ):
        raise ValueError("missile_type must be H=21000 or M=21001")
    _validate_target_type(target_type)
    damage = settings.h_damage if missile_type == H_MISSILE_TYPE else settings.m_damage
    hit_rate = settings.hit_rate_9400 if target_type == 9400 else settings.hit_rate_9600
    return float(damage * hit_rate)


class HMPublicStrikePackagesSkill:
    """Allocate active H/M missiles without reading simulator health state."""

    def __init__(
        self,
        config: PublicStrikeConfig | None = None,
        *,
        public_targets: Sequence[PublicTargetTrack] = (),
    ) -> None:
        self.config = config or PublicStrikeConfig()
        initial = self._validate_targets(public_targets)
        self._initial_targets = {item.target_id: item for item in initial}
        self._targets = dict(self._initial_targets)
        self._assignments: dict[int, int] = {}
        self._member_types: dict[int, int] = {}
        self._packages: tuple[PublicStrikePackage, ...] = ()
        self._assigned_active_ids: frozenset[int] = frozenset()

    def decide(
        self,
        *,
        step: int,
        members: Sequence[HMMissileSnapshot],
        track_updates: Sequence[PublicTargetTrack] = (),
    ) -> PublicStrikeDecision:
        _nonnegative_int(step, "step")
        ordered_members = self._validate_members(members)
        refreshed_target_ids = self._ingest_track_updates(track_updates)
        active = tuple(item for item in ordered_members if item.alive and item.launched)
        active_ids = frozenset(item.entity_id for item in active)
        baseline = tuple((item.entity_id, item.baseline_action) for item in active)

        should_allocate = bool(self._targets) and not self._assignments
        if self._targets and not self.config.lock_assignments and self._assignments:
            should_allocate = active_ids != self._assigned_active_ids

        previous = dict(self._assignments)
        if should_allocate:
            self._assignments, self._member_types = self._allocate(active)
            self._assigned_active_ids = active_ids
        self._packages = self._build_packages(active_ids)
        allocation_changed = self._assignments != previous

        requests = {
            (member_id, target_id)
            for member_id, target_id in self._assignments.items()
            if allocation_changed and previous.get(member_id) != target_id
        }
        requests.update(
            (member_id, target_id)
            for member_id, target_id in self._assignments.items()
            if target_id in refreshed_target_ids and member_id in active_ids
        )
        assigned_ids = frozenset(self._assignments)
        unassigned = tuple(item.entity_id for item in active if item.entity_id not in assigned_ids)
        return PublicStrikeDecision(
            baseline_actions=baseline,
            attack_actions=tuple(AttackTargetAction(*item) for item in sorted(requests)),
            packages=self._packages,
            public_targets=tuple(sorted(self._targets.values(), key=lambda item: item.target_id)),
            unassigned_member_ids=unassigned,
            allocation_changed=allocation_changed,
        )

    def reset(self) -> None:
        """Clear assignments and discard refreshes while retaining the legal directory."""
        self._targets = dict(self._initial_targets)
        self._assignments.clear()
        self._member_types.clear()
        self._packages = ()
        self._assigned_active_ids = frozenset()

    @staticmethod
    def _validate_members(
        members: Sequence[HMMissileSnapshot],
    ) -> tuple[HMMissileSnapshot, ...]:
        ordered = tuple(sorted(members, key=lambda item: item.entity_id))
        ids = tuple(item.entity_id for item in ordered)
        if len(ids) != len(set(ids)):
            raise ValueError("H/M missile members must have unique entity IDs")
        return ordered

    @staticmethod
    def _validate_targets(
        targets: Sequence[PublicTargetTrack],
    ) -> tuple[PublicTargetTrack, ...]:
        ordered = tuple(sorted(targets, key=lambda item: item.target_id))
        ids = tuple(item.target_id for item in ordered)
        if len(ids) != len(set(ids)):
            raise ValueError("Public targets must have unique target IDs")
        return ordered

    def _ingest_track_updates(self, updates: Sequence[PublicTargetTrack]) -> frozenset[int]:
        newest: dict[int, PublicTargetTrack] = {}
        for update in sorted(updates, key=lambda item: (item.target_id, item.observed_step)):
            current = self._targets.get(update.target_id)
            if current is None:
                raise ValueError("Track update target_id is not in the initial public directory")
            if update.target_type != current.target_type:
                raise ValueError("Track update target_type does not match the public directory")
            if (
                float(update.initial_health) != float(current.initial_health)
                or float(update.value) != float(current.value)
            ):
                raise ValueError("Track updates cannot change initial_health or value")
            prior_update = newest.get(update.target_id)
            if prior_update is not None and update.observed_step == prior_update.observed_step:
                if (update.lon, update.lat) != (prior_update.lon, prior_update.lat):
                    raise ValueError("Conflicting coordinates at the same observed_step")
            if prior_update is None or update.observed_step > prior_update.observed_step:
                newest[update.target_id] = update

        refreshed: set[int] = set()
        for target_id, update in newest.items():
            current = self._targets[target_id]
            if update.observed_step < current.observed_step:
                continue
            if update.observed_step == current.observed_step:
                if (update.lon, update.lat) != (current.lon, current.lat):
                    raise ValueError("Conflicting coordinates at the same observed_step")
                continue
            self._targets[target_id] = update
            if (update.lon, update.lat) != (current.lon, current.lat):
                refreshed.add(target_id)
        return frozenset(refreshed)

    def _allocate(
        self, active: tuple[HMMissileSnapshot, ...],
    ) -> tuple[dict[int, int], dict[int, int]]:
        by_type = {
            missile_type: [item for item in active if item.missile_type == missile_type]
            for missile_type in (H_MISSILE_TYPE, M_MISSILE_TYPE)
        }
        available: dict[int, list[HMMissileSnapshot]] = {}
        for missile_type, reserve in (
            (H_MISSILE_TYPE, self.config.reserve_h),
            (M_MISSILE_TYPE, self.config.reserve_m),
        ):
            keep = max(0, len(by_type[missile_type]) - reserve)
            available[missile_type] = by_type[missile_type][:keep]

        targets = tuple(sorted(self._targets.values(), key=lambda item: item.target_id))
        plan = self._coverage_plan(
            targets,
            len(available[H_MISSILE_TYPE]),
            len(available[M_MISSILE_TYPE]),
        )
        assignments: dict[int, int] = {}
        member_types: dict[int, int] = {}
        damage_by_target = {target.target_id: 0.0 for target in targets}

        if plan is not None:
            for target in targets:
                h_count, m_count = plan[target.target_id]
                for missile_type, count in (
                    (H_MISSILE_TYPE, h_count), (M_MISSILE_TYPE, m_count),
                ):
                    for item in available[missile_type][:count]:
                        assignments[item.entity_id] = target.target_id
                        member_types[item.entity_id] = missile_type
                        damage_by_target[target.target_id] += expected_effective_damage(
                            missile_type, target.target_type, self.config,
                        )
                    del available[missile_type][:count]
        else:
            self._allocate_scarce(
                targets, available, assignments, member_types, damage_by_target,
            )

        leftovers = sorted(
            available[H_MISSILE_TYPE] + available[M_MISSILE_TYPE],
            key=lambda item: item.entity_id,
        )
        for item in leftovers:
            target = min(
                targets,
                key=lambda candidate: self._extra_priority(candidate, damage_by_target[candidate.target_id]),
            )
            assignments[item.entity_id] = target.target_id
            member_types[item.entity_id] = item.missile_type
            damage_by_target[target.target_id] += expected_effective_damage(
                item.missile_type, target.target_type, self.config,
            )
        return assignments, member_types

    def _coverage_plan(
        self,
        targets: tuple[PublicTargetTrack, ...],
        h_available: int,
        m_available: int,
    ) -> dict[int, tuple[int, int]] | None:
        ordered = tuple(sorted(
            targets,
            key=lambda target: self._minimum_priority(target, 0.0),
        ))
        # State key is exact H/M use; value retains least overkill and stable H placement.
        states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {
            (0, 0): (0.0, ()),
        }
        for target in ordered:
            required = self._required(target)
            h_damage = expected_effective_damage(
                H_MISSILE_TYPE, target.target_type, self.config,
            )
            m_damage = expected_effective_damage(
                M_MISSILE_TYPE, target.target_type, self.config,
            )
            maximum_useful_h = min(
                h_available, max(0, ceil((required - _EPSILON) / h_damage)),
            )
            choices = []
            for h_count in range(maximum_useful_h + 1):
                remaining = max(0.0, required - h_count * h_damage)
                m_count = max(0, ceil((remaining - _EPSILON) / m_damage))
                if m_count <= m_available:
                    overkill = h_count * h_damage + m_count * m_damage - required
                    choices.append((h_count, m_count, overkill))

            next_states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}
            for (used_h, used_m), (prior_overkill, counts) in states.items():
                for h_count, m_count, overkill in choices:
                    key = (used_h + h_count, used_m + m_count)
                    if key[0] > h_available or key[1] > m_available:
                        continue
                    candidate = (prior_overkill + overkill, counts + (h_count,))
                    previous = next_states.get(key)
                    if previous is None or (
                        candidate[0], tuple(-count for count in candidate[1])
                    ) < (
                        previous[0], tuple(-count for count in previous[1])
                    ):
                        next_states[key] = candidate
            states = next_states

        if not states:
            return None
        (h_used, m_used), (_, counts) = min(
            states.items(),
            key=lambda item: (
                item[1][0],
                item[0][0] + item[0][1],
                tuple(-count for count in item[1][1]),
                item[0],
            ),
        )
        assert h_used <= h_available and m_used <= m_available
        return {
            target.target_id: (h_count, max(
                0,
                ceil((
                    self._required(target)
                    - h_count * expected_effective_damage(
                        H_MISSILE_TYPE, target.target_type, self.config,
                    )
                    - _EPSILON
                ) / expected_effective_damage(
                    M_MISSILE_TYPE, target.target_type, self.config,
                )),
            ))
            for target, h_count in zip(ordered, counts, strict=True)
        }

    def _allocate_scarce(
        self,
        targets: tuple[PublicTargetTrack, ...],
        available: dict[int, list[HMMissileSnapshot]],
        assignments: dict[int, int],
        member_types: dict[int, int],
        damage_by_target: dict[int, float],
    ) -> None:
        pool = sorted(
            available[H_MISSILE_TYPE] + available[M_MISSILE_TYPE],
            key=lambda item: item.entity_id,
        )
        available[H_MISSILE_TYPE].clear()
        available[M_MISSILE_TYPE].clear()
        while pool:
            target = min(
                targets,
                key=lambda candidate: self._minimum_priority(
                    candidate, damage_by_target[candidate.target_id],
                ),
            )
            gap = max(0.0, self._required(target) - damage_by_target[target.target_id])
            item = min(
                pool,
                key=lambda candidate: (
                    abs(gap - expected_effective_damage(
                        candidate.missile_type, target.target_type, self.config,
                    )),
                    expected_effective_damage(
                        candidate.missile_type, target.target_type, self.config,
                    ) > gap,
                    candidate.entity_id,
                ),
            )
            pool.remove(item)
            assignments[item.entity_id] = target.target_id
            member_types[item.entity_id] = item.missile_type
            damage_by_target[target.target_id] += expected_effective_damage(
                item.missile_type, target.target_type, self.config,
            )

    def _minimum_priority(
        self, target: PublicTargetTrack, assigned_damage: float,
    ) -> tuple[float, float, int]:
        required = self._required(target)
        shortfall_ratio = max(0.0, required - assigned_damage) / required
        return (-shortfall_ratio, -float(target.value), target.target_id)

    def _extra_priority(
        self, target: PublicTargetTrack, assigned_damage: float,
    ) -> tuple[float, float, int]:
        required = self._required(target)
        shortfall_ratio = (required - assigned_damage) / required
        return (-float(target.value), -shortfall_ratio, target.target_id)

    def _required(self, target: PublicTargetTrack) -> float:
        return float(target.initial_health * self.config.safety_factor)

    def _build_packages(self, active_ids: frozenset[int]) -> tuple[PublicStrikePackage, ...]:
        packages = []
        for target in sorted(self._targets.values(), key=lambda item: item.target_id):
            member_ids = tuple(sorted(
                member_id for member_id, target_id in self._assignments.items()
                if target_id == target.target_id and member_id in active_ids
            ))
            if not member_ids:
                continue
            expected = sum(
                expected_effective_damage(self._member_types[member_id], target.target_type, self.config)
                for member_id in member_ids
            )
            packages.append(PublicStrikePackage(
                target=target,
                member_ids=member_ids,
                required_damage=self._required(target),
                expected_damage=expected,
            ))
        return tuple(packages)


def _validate_target_type(target_type: int) -> None:
    if (
        isinstance(target_type, bool)
        or not isinstance(target_type, int)
        or target_type not in PUBLIC_TARGET_TYPES
    ):
        raise ValueError("target_type must be public 9400 or 9600")
