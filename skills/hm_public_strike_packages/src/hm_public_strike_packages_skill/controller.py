"""Pure deterministic H/M allocation for legally known public targets."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, comb, fsum, isfinite
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


def _positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


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
    allocation_mode: str = "full_coverage_mixed"
    target_kill_probability: float = 0.85
    min_h_per_mother: int = 2
    min_m_per_mother: int = 0
    min_h_per_sentry: int = 1
    max_h_per_mother: int = 2
    h_per_sentry_package: int = 2
    m_per_h_mixed_package: int = 2
    m_only_sentry_package: int = 6
    target_priority_axis: str = "east"
    m_target_policy: str = "complete_h_sentries"
    allow_m_only_sentry_packages: bool = False

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
        if not isinstance(self.allocation_mode, str) or self.allocation_mode not in (
            "full_coverage_mixed", "deterministic_thresholds", "threshold_packages",
            "kill_probability", "expected_damage",
        ):
            raise ValueError(
                "allocation_mode must be 'full_coverage_mixed', "
                "'deterministic_thresholds', 'threshold_packages', "
                "'kill_probability', or 'expected_damage'"
            )
        _positive_finite(self.target_kill_probability, "target_kill_probability")
        if float(self.target_kill_probability) > 1.0:
            raise ValueError("target_kill_probability must be in (0, 1]")
        _nonnegative_int(self.reserve_h, "reserve_h")
        _nonnegative_int(self.reserve_m, "reserve_m")
        _nonnegative_int(self.min_h_per_mother, "min_h_per_mother")
        _nonnegative_int(self.min_m_per_mother, "min_m_per_mother")
        _nonnegative_int(self.min_h_per_sentry, "min_h_per_sentry")
        _nonnegative_int(self.max_h_per_mother, "max_h_per_mother")
        _positive_int(self.h_per_sentry_package, "h_per_sentry_package")
        _positive_int(self.m_per_h_mixed_package, "m_per_h_mixed_package")
        _positive_int(self.m_only_sentry_package, "m_only_sentry_package")
        if (
            self.allocation_mode not in {
                "full_coverage_mixed", "deterministic_thresholds",
                "threshold_packages",
            }
            and self.min_h_per_mother > self.max_h_per_mother
        ):
            raise ValueError("min_h_per_mother cannot exceed max_h_per_mother")
        if self.target_priority_axis not in {"east", "west", "north", "south", "none"}:
            raise ValueError(
                "target_priority_axis must be east, west, north, south, or none"
            )
        if not isinstance(self.m_target_policy, str) or self.m_target_policy not in {
            "complete_h_sentries", "incomplete_h_sentries", "mothers",
        }:
            raise ValueError(
                "m_target_policy must be 'complete_h_sentries', "
                "'incomplete_h_sentries', or 'mothers'"
            )
        if not isinstance(self.lock_assignments, bool):
            raise ValueError("lock_assignments must be a boolean")
        if not isinstance(self.allow_m_only_sentry_packages, bool):
            raise ValueError("allow_m_only_sentry_packages must be a boolean")

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
    kill_probability: float = 0.0
    nominal_applied_damage: float = 0.0
    threshold_satisfied: bool = False


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


def kill_probability(
    h_count: int,
    m_count: int,
    target_type: int,
    config: PublicStrikeConfig | None = None,
) -> float:
    """Return exact P(total discrete H/M hit damage >= target health)."""
    _nonnegative_int(h_count, "h_count")
    _nonnegative_int(m_count, "m_count")
    _validate_target_type(target_type)
    if config is not None and not isinstance(config, PublicStrikeConfig):
        raise ValueError("config must be a PublicStrikeConfig")
    settings = config or PublicStrikeConfig()
    return _kill_probability_for_health(
        h_count,
        m_count,
        target_type,
        settings.default_health(target_type),
        settings,
    )


def _kill_probability_for_health(
    h_count: int,
    m_count: int,
    target_type: int,
    health: float,
    config: PublicStrikeConfig,
) -> float:
    if h_count * config.h_damage + m_count * config.m_damage < health:
        return 0.0
    hit_rate = config.hit_rate_9400 if target_type == 9400 else config.hit_rate_9600
    miss_rate = 1.0 - hit_rate
    probability = fsum(
        comb(h_count, h_hits)
        * hit_rate ** h_hits
        * miss_rate ** (h_count - h_hits)
        * comb(m_count, m_hits)
        * hit_rate ** m_hits
        * miss_rate ** (m_count - m_hits)
        for h_hits in range(h_count + 1)
        for m_hits in range(m_count + 1)
        if h_hits * config.h_damage + m_hits * config.m_damage >= health
    )
    return min(1.0, max(0.0, probability))


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
        if self.config.allocation_mode == "full_coverage_mixed":
            return self._allocate_by_full_coverage_mixed(active)
        if self.config.allocation_mode == "deterministic_thresholds":
            return self._allocate_by_deterministic_thresholds(active)
        if self.config.allocation_mode == "threshold_packages":
            return self._allocate_by_threshold_packages(active)
        if self.config.allocation_mode == "kill_probability":
            return self._allocate_by_kill_probability(active)
        return self._allocate_by_expected_damage(active)

    def _available_by_type(
        self, active: tuple[HMMissileSnapshot, ...],
    ) -> dict[int, list[HMMissileSnapshot]]:
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
        return available

    def _allocate_by_full_coverage_mixed(
        self, active: tuple[HMMissileSnapshot, ...],
    ) -> tuple[dict[int, int], dict[int, int]]:
        available = self._available_by_type(active)
        targets = tuple(sorted(self._targets.values(), key=lambda item: item.target_id))
        mothers = tuple(sorted(
            (target for target in targets if target.target_type == 9400),
            key=lambda target: (-float(target.value), target.target_id),
        ))
        sentries = tuple(sorted(
            (target for target in targets if target.target_type == 9600),
            key=lambda target: (self._target_priority(target), target.target_id),
        ))
        assignments: dict[int, int] = {}
        member_types: dict[int, int] = {}
        counts = {target.target_id: [0, 0] for target in targets}

        def assign_next(missile_type: int, target: PublicTargetTrack) -> bool:
            if not available[missile_type]:
                return False
            item = available[missile_type].pop(0)
            assignments[item.entity_id] = target.target_id
            member_types[item.entity_id] = item.missile_type
            count_index = 0 if missile_type == H_MISSILE_TYPE else 1
            counts[target.target_id][count_index] += 1
            return True

        def assign_count(
            missile_type: int, target: PublicTargetTrack, count: int,
        ) -> None:
            for _ in range(count):
                if not assign_next(missile_type, target):
                    break

        # Preserve scarce-H fairness: finish each allocation round before any
        # target receives its next H missile.
        for _ in range(self.config.min_h_per_mother):
            for mother in mothers:
                if not assign_next(H_MISSILE_TYPE, mother):
                    break

        for sentry in sentries:
            if not assign_next(H_MISSILE_TYPE, sentry):
                break

        # Upgrade the highest-priority one-H sentries only after every sentry
        # has had the opportunity to receive its coverage seed.
        for sentry in sentries:
            if not available[H_MISSILE_TYPE]:
                break
            if counts[sentry.target_id][0] == 1:
                assign_next(H_MISSILE_TYPE, sentry)

        # Inventories beyond full 2H coverage remain deterministic redundancy.
        h_redundancy_targets = sentries or mothers
        while available[H_MISSILE_TYPE] and h_redundancy_targets:
            target = min(
                h_redundancy_targets,
                key=lambda item: (
                    counts[item.target_id][0],
                    self._target_priority(item)
                    if item.target_type == 9600
                    else -float(item.value),
                    item.target_id,
                ),
            )
            assign_next(H_MISSILE_TYPE, target)

        one_h_sentries = tuple(
            sentry for sentry in sentries if counts[sentry.target_id][0] == 1
        )
        zero_h_sentries = tuple(
            sentry for sentry in sentries if counts[sentry.target_id][0] == 0
        )

        # A 1H+2M package applies exactly 12+6 damage to a standard 9600.
        # Complete as many such packages as possible before spending six M on
        # an M-only package; this maximizes deterministic target coverage.
        for sentry in one_h_sentries:
            assign_count(
                M_MISSILE_TYPE,
                sentry,
                min(
                    self.config.m_per_h_mixed_package,
                    len(available[M_MISSILE_TYPE]),
                ),
            )

        m_only_sentries: list[PublicTargetTrack] = []
        for sentry in zero_h_sentries:
            if len(available[M_MISSILE_TYPE]) < self.config.m_only_sentry_package:
                break
            assign_count(
                M_MISSILE_TYPE, sentry, self.config.m_only_sentry_package,
            )
            m_only_sentries.append(sentry)

        # M redundancy must not dilute the coverage-first plan. When 1H
        # sentries exist, only they receive the tail; with universal 2H
        # coverage, M is explicitly redundant and balanced across sentries.
        if one_h_sentries:
            m_redundancy_targets = one_h_sentries
        elif sentries and all(
            counts[sentry.target_id][0] >= self.config.h_per_sentry_package
            for sentry in sentries
        ):
            m_redundancy_targets = sentries
        elif m_only_sentries:
            m_redundancy_targets = tuple(m_only_sentries)
        else:
            m_redundancy_targets = sentries or mothers

        while available[M_MISSILE_TYPE] and m_redundancy_targets:
            target = min(
                m_redundancy_targets,
                key=lambda item: (
                    counts[item.target_id][1],
                    self._target_priority(item)
                    if item.target_type == 9600
                    else -float(item.value),
                    item.target_id,
                ),
            )
            assign_next(M_MISSILE_TYPE, target)

        return assignments, member_types

    def _allocate_by_deterministic_thresholds(
        self, active: tuple[HMMissileSnapshot, ...],
    ) -> tuple[dict[int, int], dict[int, int]]:
        available = self._available_by_type(active)
        targets = tuple(sorted(self._targets.values(), key=lambda item: item.target_id))
        mothers = tuple(sorted(
            (target for target in targets if target.target_type == 9400),
            key=lambda target: (-float(target.value), target.target_id),
        ))
        sentries = tuple(sorted(
            (target for target in targets if target.target_type == 9600),
            key=lambda target: (self._target_priority(target), target.target_id),
        ))
        assignments: dict[int, int] = {}
        member_types: dict[int, int] = {}
        counts = {target.target_id: [0, 0] for target in targets}

        def assign_next(missile_type: int, target: PublicTargetTrack) -> bool:
            if not available[missile_type]:
                return False
            item = available[missile_type].pop(0)
            assignments[item.entity_id] = target.target_id
            member_types[item.entity_id] = item.missile_type
            count_index = 0 if missile_type == H_MISSILE_TYPE else 1
            counts[target.target_id][count_index] += 1
            return True

        def assign_count(
            missile_type: int, target: PublicTargetTrack, count: int,
        ) -> None:
            for _ in range(count):
                if not assign_next(missile_type, target):
                    break

        def threshold_satisfied(target: PublicTargetTrack) -> bool:
            h_count, m_count = counts[target.target_id]
            nominal = (
                h_count * expected_effective_damage(
                    H_MISSILE_TYPE, target.target_type, self.config,
                )
                + m_count * expected_effective_damage(
                    M_MISSILE_TYPE, target.target_type, self.config,
                )
            )
            return nominal + _EPSILON >= target.initial_health

        # The environment-specific launch scripts choose two H per mother for
        # V12 and three for V13. M missiles are never assigned to mothers.
        for _ in range(self.config.min_h_per_mother):
            for mother in mothers:
                if not assign_next(H_MISSILE_TYPE, mother):
                    break

        next_sentry = 0
        package_size = self.config.h_per_sentry_package
        h_package_sentry_ids: set[int] = set()
        while (
            next_sentry < len(sentries)
            and len(available[H_MISSILE_TYPE]) >= package_size
        ):
            target = sentries[next_sentry]
            assign_count(H_MISSILE_TYPE, target, package_size)
            h_package_sentry_ids.add(target.target_id)
            next_sentry += 1

        mixed_sentry_id: int | None = None
        if (
            next_sentry < len(sentries)
            and available[H_MISSILE_TYPE]
            and len(available[M_MISSILE_TYPE])
            >= self.config.m_per_h_mixed_package
        ):
            target = sentries[next_sentry]
            assign_next(H_MISSILE_TYPE, target)
            assign_count(
                M_MISSILE_TYPE, target, self.config.m_per_h_mixed_package,
            )
            mixed_sentry_id = target.target_id
            next_sentry += 1

        if self.config.allow_m_only_sentry_packages:
            while (
                next_sentry < len(sentries)
                and len(available[M_MISSILE_TYPE])
                >= self.config.m_only_sentry_package
            ):
                assign_count(
                    M_MISSILE_TYPE,
                    sentries[next_sentry],
                    self.config.m_only_sentry_package,
                )
                next_sentry += 1

        complete_sentries = tuple(
            target for target in sentries if threshold_satisfied(target)
        )
        complete_mothers = tuple(
            target for target in mothers if threshold_satisfied(target)
        )

        # Tail inventory is redundancy, not another claimed coverage target.
        # Prefer the last complete sentry so the priority-axis coverage prefix
        # remains visible and deterministic in package diagnostics.
        h_redundancy_target = (
            complete_sentries[-1]
            if complete_sentries
            else complete_mothers[-1]
            if complete_mothers
            else sentries[0]
            if sentries
            else mothers[0]
            if mothers
            else None
        )
        while available[H_MISSILE_TYPE] and h_redundancy_target is not None:
            assign_next(H_MISSILE_TYPE, h_redundancy_target)

        if self.config.allow_m_only_sentry_packages:
            # v0.7 compatibility: M-only packages and their tail remain on the
            # last complete sentry (or the first sentry when none is complete).
            m_redundancy_target = (
                complete_sentries[-1]
                if complete_sentries
                else sentries[0]
                if sentries
                else None
            )
            while available[M_MISSILE_TYPE] and m_redundancy_target is not None:
                assign_next(M_MISSILE_TYPE, m_redundancy_target)
        else:
            h_led_sentries = tuple(
                target for target in sentries if counts[target.target_id][0] > 0
            )
            while available[M_MISSILE_TYPE] and h_led_sentries:
                target = min(
                    h_led_sentries,
                    key=lambda item: (
                        counts[item.target_id][1],
                        0
                        if item.target_id == mixed_sentry_id
                        else 1
                        if item.target_id in h_package_sentry_ids
                        else 2,
                        self._target_priority(item),
                        item.target_id,
                    ),
                )
                assign_next(M_MISSILE_TYPE, target)

        return assignments, member_types

    def _allocate_by_threshold_packages(
        self, active: tuple[HMMissileSnapshot, ...],
    ) -> tuple[dict[int, int], dict[int, int]]:
        available = self._available_by_type(active)
        targets = tuple(sorted(self._targets.values(), key=lambda item: item.target_id))
        mothers = tuple(sorted(
            (target for target in targets if target.target_type == 9400),
            key=lambda target: (-float(target.value), target.target_id),
        ))
        sentries = tuple(sorted(
            (target for target in targets if target.target_type == 9600),
            key=lambda target: (self._target_priority(target), target.target_id),
        ))
        assignments: dict[int, int] = {}
        member_types: dict[int, int] = {}
        counts = {target.target_id: [0, 0] for target in targets}

        def assign_next(missile_type: int, target: PublicTargetTrack) -> bool:
            if not available[missile_type]:
                return False
            item = available[missile_type].pop(0)
            assignments[item.entity_id] = target.target_id
            member_types[item.entity_id] = item.missile_type
            count_index = 0 if missile_type == H_MISSILE_TYPE else 1
            counts[target.target_id][count_index] += 1
            return True

        # Round-based seeding is deterministic and keeps scarce H inventory fair.
        for _ in range(self.config.min_h_per_mother):
            for mother in mothers:
                if not assign_next(H_MISSILE_TYPE, mother):
                    break

        package_size = self.config.h_per_sentry_package
        for sentry in sentries:
            if len(available[H_MISSILE_TYPE]) < package_size:
                break
            for _ in range(package_size):
                assign_next(H_MISSILE_TYPE, sentry)

        incomplete_sentries = tuple(
            sentry for sentry in sentries
            if counts[sentry.target_id][0] < package_size
        )
        if incomplete_sentries:
            partial_target = incomplete_sentries[0]
            while available[H_MISSILE_TYPE]:
                assign_next(H_MISSILE_TYPE, partial_target)

        # Inventories above the one-package threshold are still fully consumed.
        h_fallback = sentries or mothers
        while available[H_MISSILE_TYPE] and h_fallback:
            target = min(
                h_fallback,
                key=lambda item: (
                    counts[item.target_id][0],
                    self._target_priority(item) if item.target_type == 9600 else -float(item.value),
                    item.target_id,
                ),
            )
            assign_next(H_MISSILE_TYPE, target)

        complete_sentries = tuple(
            sentry for sentry in sentries
            if counts[sentry.target_id][0] >= package_size
        )
        incomplete_sentries = tuple(
            sentry for sentry in sentries
            if counts[sentry.target_id][0] < package_size
        )
        if self.config.m_target_policy == "complete_h_sentries":
            partial_sentries = tuple(
                sentry for sentry in sentries
                if 0 < counts[sentry.target_id][0] < package_size
            )
            uncovered_sentries = tuple(
                sentry for sentry in sentries
                if counts[sentry.target_id][0] == 0
            )
            m_targets = (
                complete_sentries
                or partial_sentries
                or uncovered_sentries
                or mothers
            )
        elif self.config.m_target_policy == "incomplete_h_sentries":
            # v0.5 compatibility: prefer every sentry below the H threshold.
            if incomplete_sentries:
                m_targets = incomplete_sentries
            elif mothers:
                m_targets = mothers
            else:
                m_targets = sentries
        else:
            m_targets = mothers or sentries

        while available[M_MISSILE_TYPE] and m_targets:
            target = min(
                m_targets,
                key=lambda item: (
                    counts[item.target_id][1],
                    counts[item.target_id][0]
                    if self.config.m_target_policy == "incomplete_h_sentries"
                    else 0,
                    self._target_priority(item) if item.target_type == 9600 else -float(item.value),
                    item.target_id,
                ),
            )
            assign_next(M_MISSILE_TYPE, target)

        return assignments, member_types

    def _allocate_by_kill_probability(
        self, active: tuple[HMMissileSnapshot, ...],
    ) -> tuple[dict[int, int], dict[int, int]]:
        available = self._available_by_type(active)
        targets = tuple(sorted(self._targets.values(), key=lambda item: item.target_id))
        assignments: dict[int, int] = {}
        member_types: dict[int, int] = {}
        counts = {target.target_id: [0, 0] for target in targets}

        mothers = tuple(target for target in targets if target.target_type == 9400)
        sentries = tuple(target for target in targets if target.target_type == 9600)

        def assign_core(target_group, missile_type: int, count: int) -> None:
            count_index = 0 if missile_type == H_MISSILE_TYPE else 1
            for _ in range(count):
                for target in target_group:
                    if not available[missile_type]:
                        return
                    item = available[missile_type].pop(0)
                    assignments[item.entity_id] = target.target_id
                    member_types[item.entity_id] = item.missile_type
                    counts[target.target_id][count_index] += 1

        # Bundle-level seeding avoids the zero single-missile marginal that
        # previously starved every mother of H missiles.
        assign_core(mothers, H_MISSILE_TYPE, self.config.min_h_per_mother)
        assign_core(mothers, M_MISSILE_TYPE, self.config.min_m_per_mother)
        assign_core(sentries, H_MISSILE_TYPE, self.config.min_h_per_sentry)

        # A few isolated M missiles have zero discrete kill-probability gain on
        # an H-covered 9600. Keep all remaining M in balanced mother packages
        # instead of relying on a one-missile greedy step to discover a
        # four-or-more-M complementary bundle.
        if mothers:
            while available[M_MISSILE_TYPE]:
                target = min(
                    mothers,
                    key=lambda item: (counts[item.target_id][1], item.target_id),
                )
                item = available[M_MISSILE_TYPE].pop(0)
                assignments[item.entity_id] = target.target_id
                member_types[item.entity_id] = item.missile_type
                counts[target.target_id][1] += 1

        pool = sorted(
            available[H_MISSILE_TYPE] + available[M_MISSILE_TYPE],
            key=lambda item: item.entity_id,
        )
        current_probability = {
            target.target_id: _kill_probability_for_health(
                *counts[target.target_id],
                target.target_type,
                target.initial_health,
                self.config,
            )
            for target in targets
        }

        while pool:
            best: tuple[
                tuple[int, float, float, float, int, int],
                HMMissileSnapshot,
                PublicTargetTrack,
                float,
            ] | None = None
            for item in pool:
                count_index = 0 if item.missile_type == H_MISSILE_TYPE else 1
                for target in targets:
                    target_counts = counts[target.target_id]
                    if (
                        item.missile_type == H_MISSILE_TYPE
                        and target.target_type == 9400
                        and target_counts[0] >= self.config.max_h_per_mother
                    ):
                        continue
                    next_counts = target_counts.copy()
                    next_counts[count_index] += 1
                    next_probability = _kill_probability_for_health(
                        *next_counts,
                        target.target_type,
                        target.initial_health,
                        self.config,
                    )
                    probability = current_probability[target.target_id]
                    marginal_gain = max(0.0, next_probability - probability)
                    key = (
                        int(probability >= self.config.target_kill_probability),
                        -float(target.value) * marginal_gain,
                        probability,
                        self._target_priority(target),
                        target.target_id,
                        item.entity_id,
                    )
                    if best is None or key < best[0]:
                        best = (key, item, target, next_probability)
            if best is None:
                break
            _, item, target, next_probability = best
            pool.remove(item)
            assignments[item.entity_id] = target.target_id
            member_types[item.entity_id] = item.missile_type
            count_index = 0 if item.missile_type == H_MISSILE_TYPE else 1
            counts[target.target_id][count_index] += 1
            current_probability[target.target_id] = next_probability

        return assignments, member_types

    def _target_priority(self, target: PublicTargetTrack) -> float:
        axis = self.config.target_priority_axis
        if axis == "east":
            return -target.lon
        if axis == "west":
            return target.lon
        if axis == "north":
            return -target.lat
        if axis == "south":
            return target.lat
        return 0.0

    def _allocate_by_expected_damage(
        self, active: tuple[HMMissileSnapshot, ...],
    ) -> tuple[dict[int, int], dict[int, int]]:
        available = self._available_by_type(active)

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
            h_count = sum(
                self._member_types[member_id] == H_MISSILE_TYPE for member_id in member_ids
            )
            m_count = len(member_ids) - h_count
            packages.append(PublicStrikePackage(
                target=target,
                member_ids=member_ids,
                required_damage=self._required(target),
                expected_damage=expected,
                kill_probability=_kill_probability_for_health(
                    h_count,
                    m_count,
                    target.target_type,
                    target.initial_health,
                    self.config,
                ),
                nominal_applied_damage=expected,
                threshold_satisfied=(
                    expected + _EPSILON >= target.initial_health
                ),
            ))
        return tuple(packages)


def _validate_target_type(target_type: int) -> None:
    if (
        isinstance(target_type, bool)
        or not isinstance(target_type, int)
        or target_type not in PUBLIC_TARGET_TYPES
    ):
        raise ValueError("target_type must be public 9400 or 9600")
