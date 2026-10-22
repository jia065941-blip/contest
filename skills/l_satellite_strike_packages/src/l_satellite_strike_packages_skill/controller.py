"""Deterministic satellite discovery and L-only hidden-nest allocation."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from typing import Sequence


ACTION_SET_ACC_Z = 0
ACTION_USE_SATELLITE = 3
ACTION_ATTACK_TARGET = 4
L_MISSILE_TYPE = 21002
HIDDEN_NEST_TYPE = 9500


@dataclass(frozen=True)
class LMissileSnapshot:
    entity_id: int
    alive: bool = True
    launched: bool = True
    baseline_action: int = 1

    def __post_init__(self) -> None:
        if self.entity_id < 0:
            raise ValueError("L missile entity_id must be nonnegative")
        if self.baseline_action not in (0, 1, 2):
            raise ValueError("baseline_action must be LEFT=0, STOP=1, or RIGHT=2")


@dataclass(frozen=True)
class DarkNestTrack:
    target_id: int
    lon: float
    lat: float
    observed_step: float

    def __post_init__(self) -> None:
        if self.target_id < 0:
            raise ValueError("Hidden-nest target_id must be nonnegative")
        if not all(isfinite(value) for value in (self.lon, self.lat, self.observed_step)):
            raise ValueError("Hidden-nest track values must be finite")


@dataclass(frozen=True)
class SatelliteStrikeConfig:
    satellite_trigger_step: int = 1
    satellite_leader_id: int | None = None
    allocation_mode: str = "equal"
    package_size: int | None = None
    package_count: int | None = None
    reserve_count: int = 0
    reserve_fraction: float = 0.0
    min_package_size: int = 1
    max_package_size: int | None = None
    expected_target_count: int | None = None
    lock_assignments: bool = True
    reassign_lost_members: bool = False

    def __post_init__(self) -> None:
        if self.satellite_trigger_step < 0:
            raise ValueError("satellite_trigger_step must be nonnegative")
        if self.satellite_leader_id is not None and self.satellite_leader_id < 0:
            raise ValueError("satellite_leader_id must be nonnegative")
        if self.allocation_mode not in {"equal", "fixed_size"}:
            raise ValueError("allocation_mode must be 'equal' or 'fixed_size'")
        if self.allocation_mode == "fixed_size" and self.package_size is None:
            raise ValueError("fixed_size allocation requires package_size")
        if self.package_size is not None and self.package_size <= 0:
            raise ValueError("package_size must be positive")
        if self.package_count is not None and self.package_count <= 0:
            raise ValueError("package_count must be positive")
        if self.reserve_count < 0:
            raise ValueError("reserve_count must be nonnegative")
        if not 0.0 <= self.reserve_fraction < 1.0:
            raise ValueError("reserve_fraction must be in [0, 1)")
        if self.reserve_count and self.reserve_fraction:
            raise ValueError("Use reserve_count or reserve_fraction, not both")
        if self.min_package_size <= 0:
            raise ValueError("min_package_size must be positive")
        if self.max_package_size is not None and self.max_package_size < self.min_package_size:
            raise ValueError("max_package_size must be at least min_package_size")
        if self.expected_target_count is not None and self.expected_target_count <= 0:
            raise ValueError("expected_target_count must be positive")
        if self.lock_assignments and self.reassign_lost_members:
            raise ValueError("reassign_lost_members requires lock_assignments=False")


@dataclass(frozen=True)
class StrikePackage:
    target: DarkNestTrack
    member_ids: tuple[int, ...]


@dataclass(frozen=True)
class SatelliteRequest:
    requester_id: int
    reason: str = "discover_hidden_nests"

    def __post_init__(self) -> None:
        if self.requester_id < 0:
            raise ValueError("Satellite requester_id must be nonnegative")


@dataclass(frozen=True)
class AttackTargetAction:
    entity_id: int
    target_id: int

    def __post_init__(self) -> None:
        if self.entity_id < 0 or self.target_id < 0:
            raise ValueError("Attack entity and target IDs must be nonnegative")


@dataclass(frozen=True)
class SatelliteStrikeDecision:
    baseline_actions: tuple[tuple[int, int], ...]
    satellite_request: SatelliteRequest | None
    attack_actions: tuple[AttackTargetAction, ...]
    packages: tuple[StrikePackage, ...]
    discovered_targets: tuple[DarkNestTrack, ...]
    unassigned_member_ids: tuple[int, ...]
    allocation_changed: bool


class LSatelliteStrikePackagesSkill:
    """Request one global satellite use and allocate only L missiles."""

    def __init__(self, config: SatelliteStrikeConfig | None = None) -> None:
        self.config = config or SatelliteStrikeConfig()
        self._satellite_requested = False
        self._tracks: dict[int, DarkNestTrack] = {}
        self._assignments: dict[int, int] = {}
        self._packages: tuple[StrikePackage, ...] = ()
        self._assigned_alive_ids: frozenset[int] = frozenset()

    def decide(
        self,
        *,
        step: int,
        members: Sequence[LMissileSnapshot],
        tracks: Sequence[DarkNestTrack] = (),
    ) -> SatelliteStrikeDecision:
        if step < 0:
            raise ValueError("step must be nonnegative")
        ordered_members = self._validate_members(members)
        self._ingest_tracks(tracks)

        active = tuple(item for item in ordered_members if item.alive and item.launched)
        alive_ids = tuple(item.entity_id for item in active)
        baseline_actions = tuple((item.entity_id, item.baseline_action) for item in active)
        satellite_request = None
        if not self._satellite_requested and step >= self.config.satellite_trigger_step:
            leader = self._satellite_leader(alive_ids)
            if leader is not None:
                satellite_request = SatelliteRequest(leader)
                self._satellite_requested = True

        allocation_changed = False
        attack_actions: list[AttackTargetAction] = []
        if self._ready_to_allocate():
            should_allocate = not self._packages
            if not self.config.lock_assignments and not should_allocate:
                should_allocate = (
                    self.config.reassign_lost_members
                    and frozenset(alive_ids) != self._assigned_alive_ids
                ) or set(self._tracks) != {item.target.target_id for item in self._packages}
            if should_allocate:
                previous = dict(self._assignments)
                self._packages, unassigned = self._allocate(alive_ids)
                self._assignments = {
                    member_id: package.target.target_id
                    for package in self._packages
                    for member_id in package.member_ids
                }
                self._assigned_alive_ids = frozenset(alive_ids)
                allocation_changed = self._assignments != previous
                if allocation_changed:
                    for member_id, target_id in sorted(self._assignments.items()):
                        if previous.get(member_id) != target_id:
                            attack_actions.append(AttackTargetAction(member_id, target_id))
            else:
                assigned = frozenset(self._assignments)
                unassigned = tuple(entity_id for entity_id in alive_ids if entity_id not in assigned)
        else:
            unassigned = alive_ids

        return SatelliteStrikeDecision(
            baseline_actions=baseline_actions,
            satellite_request=satellite_request,
            attack_actions=tuple(attack_actions),
            packages=self._packages,
            discovered_targets=tuple(sorted(self._tracks.values(), key=lambda item: item.target_id)),
            unassigned_member_ids=unassigned,
            allocation_changed=allocation_changed,
        )

    def reset(self) -> None:
        self._satellite_requested = False
        self._tracks.clear()
        self._assignments.clear()
        self._packages = ()
        self._assigned_alive_ids = frozenset()

    @staticmethod
    def _validate_members(members: Sequence[LMissileSnapshot]) -> tuple[LMissileSnapshot, ...]:
        ordered = tuple(sorted(members, key=lambda item: item.entity_id))
        ids = tuple(item.entity_id for item in ordered)
        if len(ids) != len(set(ids)):
            raise ValueError("L missile members must have unique entity IDs")
        return ordered

    def _satellite_leader(self, alive_ids: tuple[int, ...]) -> int | None:
        if not alive_ids:
            return None
        configured = self.config.satellite_leader_id
        if configured is None:
            return alive_ids[0]
        return configured if configured in alive_ids else None

    def _ingest_tracks(self, tracks: Sequence[DarkNestTrack]) -> None:
        for track in sorted(tracks, key=lambda item: (item.target_id, item.observed_step)):
            previous = self._tracks.get(track.target_id)
            if previous is None or track.observed_step > previous.observed_step:
                self._tracks[track.target_id] = track

    def _ready_to_allocate(self) -> bool:
        expected = self.config.expected_target_count
        return bool(self._tracks) and (expected is None or len(self._tracks) >= expected)

    def _allocate(self, alive_ids: tuple[int, ...]) -> tuple[tuple[StrikePackage, ...], tuple[int, ...]]:
        reserve = self.config.reserve_count
        if self.config.reserve_fraction:
            reserve = floor(len(alive_ids) * self.config.reserve_fraction)
        reserve = min(reserve, len(alive_ids))
        available = alive_ids[:len(alive_ids) - reserve] if reserve else alive_ids
        targets = tuple(sorted(self._tracks.values(), key=lambda item: item.target_id))
        if self.config.package_count is not None:
            targets = targets[:self.config.package_count]
        if not targets:
            return (), alive_ids

        sizes = self._package_sizes(len(available), len(targets))
        packages = []
        cursor = 0
        for target, size in zip(targets, sizes, strict=True):
            member_ids = available[cursor:cursor + size]
            cursor += size
            if member_ids:
                packages.append(StrikePackage(target, member_ids))
        assigned = frozenset(member for package in packages for member in package.member_ids)
        unassigned = tuple(entity_id for entity_id in alive_ids if entity_id not in assigned)
        return tuple(packages), unassigned

    def _package_sizes(self, available: int, targets: int) -> tuple[int, ...]:
        if self.config.allocation_mode == "fixed_size":
            assert self.config.package_size is not None
            size = self.config.package_size
            if size < self.config.min_package_size:
                raise ValueError("package_size is below min_package_size")
            if self.config.max_package_size is not None and size > self.config.max_package_size:
                raise ValueError("package_size exceeds max_package_size")
            required = size * targets
            if required > available:
                raise ValueError(
                    f"fixed_size allocation requires {required} L missiles, got {available}"
                )
            return (size,) * targets

        base, remainder = divmod(available, targets)
        if base < self.config.min_package_size:
            raise ValueError(
                f"equal allocation cannot provide min_package_size={self.config.min_package_size} "
                f"to {targets} targets from {available} L missiles"
            )
        sizes = [base + (index < remainder) for index in range(targets)]
        maximum = self.config.max_package_size
        if maximum is not None:
            sizes = [min(size, maximum) for size in sizes]
        return tuple(int(size) for size in sizes)
