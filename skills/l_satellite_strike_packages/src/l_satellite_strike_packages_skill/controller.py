"""Deterministic satellite discovery and L-only hidden-nest allocation."""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, floor, isfinite, radians, sin, sqrt
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
    lon: float = 0.0
    lat: float = 0.0
    wave: int = 1

    def __post_init__(self) -> None:
        if self.entity_id < 0:
            raise ValueError("L missile entity_id must be nonnegative")
        if self.baseline_action not in (0, 1, 2):
            raise ValueError("baseline_action must be LEFT=0, STOP=1, or RIGHT=2")
        if not all(isfinite(value) for value in (self.lon, self.lat)):
            raise ValueError("L missile lon and lat must be finite")
        if self.wave not in (1, 2):
            raise ValueError("L missile wave must be 1 or 2")


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
    allocation_mode: str = "distance_balanced"
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
        if self.allocation_mode not in {"distance_balanced", "equal", "fixed_size"}:
            raise ValueError(
                "allocation_mode must be 'distance_balanced', 'equal', or 'fixed_size'"
            )
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
                self._packages, unassigned = self._allocate(active)
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

    def _allocate(
        self, active: tuple[LMissileSnapshot, ...],
    ) -> tuple[tuple[StrikePackage, ...], tuple[int, ...]]:
        alive_ids = tuple(item.entity_id for item in active)
        reserve = self.config.reserve_count
        if self.config.reserve_fraction:
            reserve = floor(len(alive_ids) * self.config.reserve_fraction)
        reserve = min(reserve, len(alive_ids))
        available = active[:len(active) - reserve] if reserve else active
        targets = tuple(sorted(self._tracks.values(), key=lambda item: item.target_id))
        if self.config.package_count is not None:
            targets = targets[:self.config.package_count]
        if not targets:
            return (), alive_ids

        sizes = self._package_sizes(len(available), len(targets))
        assignable = available[:sum(sizes)]
        if self.config.allocation_mode == "distance_balanced":
            packages = self._distance_balanced_packages(assignable, targets, sizes)
            assigned = frozenset(member for package in packages for member in package.member_ids)
            unassigned = tuple(entity_id for entity_id in alive_ids if entity_id not in assigned)
            return packages, unassigned

        packages = []
        cursor = 0
        for target, size in zip(targets, sizes, strict=True):
            member_ids = tuple(item.entity_id for item in assignable[cursor:cursor + size])
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

    def _distance_balanced_packages(
        self,
        members: tuple[LMissileSnapshot, ...],
        targets: tuple[DarkNestTrack, ...],
        sizes: tuple[int, ...],
    ) -> tuple[StrikePackage, ...]:
        wave1 = tuple(item for item in members if item.wave == 1)
        wave2 = tuple(item for item in members if item.wave == 2)
        wave1_quotas, wave2_quotas = self._wave_quotas(sizes, len(wave1))
        target_members: list[list[int]] = [[] for _ in targets]
        for wave_members, quotas in ((wave1, wave1_quotas), (wave2, wave2_quotas)):
            for member, target_index in self._minimum_cost_assignment(
                wave_members, targets, quotas,
            ):
                target_members[target_index].append(member.entity_id)
        return tuple(
            StrikePackage(target, tuple(sorted(member_ids)))
            for target, member_ids in zip(targets, target_members, strict=True)
            if member_ids
        )

    @staticmethod
    def _wave_quotas(
        sizes: tuple[int, ...], wave1_count: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Split balanced total quotas into balanced per-wave quotas."""
        target_count = len(sizes)
        assigned = sum(sizes)
        wave2_count = assigned - wave1_count
        wave1_base, wave1_extra_count = divmod(wave1_count, target_count)
        wave2_base, wave2_extra_count = divmod(wave2_count, target_count)
        total_extra_count = assigned % target_count

        wave1_extras = set(range(wave1_extra_count))
        if wave1_extra_count + wave2_extra_count < target_count:
            wave2_extras = set(range(wave1_extra_count, wave1_extra_count + wave2_extra_count))
        else:
            wave2_extras = set(range(total_extra_count))
            wave2_extras.update(range(wave1_extra_count, target_count))

        wave1_quotas = tuple(
            wave1_base + int(index in wave1_extras) for index in range(target_count)
        )
        wave2_quotas = tuple(
            wave2_base + int(index in wave2_extras) for index in range(target_count)
        )
        if tuple(first + second for first, second in zip(
            wave1_quotas, wave2_quotas, strict=True,
        )) != sizes:
            raise RuntimeError("Unable to construct balanced per-wave target quotas")
        return wave1_quotas, wave2_quotas

    @staticmethod
    def _minimum_cost_assignment(
        members: tuple[LMissileSnapshot, ...],
        targets: tuple[DarkNestTrack, ...],
        quotas: tuple[int, ...],
    ) -> tuple[tuple[LMissileSnapshot, int], ...]:
        """Solve a deterministic minimum-distance assignment to target slots."""
        slots = tuple(
            target_index
            for target_index, quota in enumerate(quotas)
            for _ in range(quota)
        )
        if not members:
            return ()
        if len(members) != len(slots):
            raise RuntimeError("Wave members and target quotas must have equal size")

        costs = tuple(
            tuple(_wgs84_distance(member, targets[target_index]) for target_index in slots)
            for member in members
        )
        size = len(members)
        row_potential = [0.0] * (size + 1)
        column_potential = [0.0] * (size + 1)
        matched_row = [0] * (size + 1)
        previous_column = [0] * (size + 1)

        # Hungarian assignment is O(n^3), which is small for the 214-member limit.
        for row in range(1, size + 1):
            matched_row[0] = row
            minimum = [float("inf")] * (size + 1)
            used = [False] * (size + 1)
            column = 0
            while True:
                used[column] = True
                current_row = matched_row[column]
                delta = float("inf")
                next_column = 0
                for candidate in range(1, size + 1):
                    if used[candidate]:
                        continue
                    reduced = (
                        costs[current_row - 1][candidate - 1]
                        - row_potential[current_row]
                        - column_potential[candidate]
                    )
                    if reduced < minimum[candidate]:
                        minimum[candidate] = reduced
                        previous_column[candidate] = column
                    choice = (minimum[candidate], matched_row[candidate] != 0, candidate)
                    if choice < (delta, matched_row[next_column] != 0, next_column):
                        delta = minimum[candidate]
                        next_column = candidate
                for candidate in range(size + 1):
                    if used[candidate]:
                        row_potential[matched_row[candidate]] += delta
                        column_potential[candidate] -= delta
                    else:
                        minimum[candidate] -= delta
                column = next_column
                if matched_row[column] == 0:
                    break
            while True:
                previous = previous_column[column]
                matched_row[column] = matched_row[previous]
                column = previous
                if column == 0:
                    break

        target_by_row = [0] * size
        for column in range(1, size + 1):
            target_by_row[matched_row[column] - 1] = slots[column - 1]
        return tuple(zip(members, target_by_row, strict=True))


def _wgs84_distance(member: LMissileSnapshot, target: DarkNestTrack) -> float:
    """Approximate WGS84 surface distance with a mean-radius haversine."""
    mean_radius_m = 6_371_008.8
    member_lat = radians(member.lat)
    target_lat = radians(target.lat)
    longitude_delta = ((target.lon - member.lon + 180.0) % 360.0) - 180.0
    sin_lat = sin(target_lat / 2.0 - member_lat / 2.0)
    sin_lon = sin(radians(longitude_delta) / 2.0)
    haversine = sin_lat * sin_lat + cos(member_lat) * cos(target_lat) * sin_lon * sin_lon
    return 2.0 * mean_radius_m * asin(sqrt(min(1.0, max(0.0, haversine))))
