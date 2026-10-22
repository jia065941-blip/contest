"""Pure event-driven L-response package release controller."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ResponsePackagePlan:
    package_id: str
    response_group_index: int
    sequence_index: int
    member_ids: tuple[int, ...]


@dataclass(frozen=True)
class MissionTargetBelief:
    response_group_index: int
    target_id: int
    expected_hp: float
    projected_expected_hp: float
    confidence: float
    confirmed_destroyed: bool
    initial_health: float = 0.0
    delivered_expected_damage: float = 0.0
    confidence_source: str = "heuristic_delivery"


@dataclass(frozen=True)
class MissionReplanConfig:
    required_discoveries: int
    release_delay_steps: int = 20
    delivery_timeout_steps: int = 700
    subsequent_release_interval_steps: int = 250
    hard_deadline_steps: int = 900
    initial_release_fraction: float = 0.50
    minimum_initial_members: int = 3
    destroyed_confidence_threshold: float = 0.75


@dataclass(frozen=True)
class MissionReplanDecision:
    launch_package_ids: tuple[str, ...]
    launch_member_ids: tuple[int, ...]
    held_package_ids: tuple[str, ...]
    held_member_ids: tuple[int, ...]
    cancelled_package_ids: tuple[str, ...]
    cancelled_member_ids: tuple[int, ...]
    target_ids_by_response_index: tuple[tuple[int, int], ...]


class MissionReplanSkill:
    """Release one L package per hidden target, then supplement on timeout."""

    def __init__(
        self,
        config: MissionReplanConfig,
        packages: tuple[ResponsePackagePlan, ...],
    ) -> None:
        self.config = config
        self._packages_by_index: dict[int, tuple[ResponsePackagePlan, ...]] = {}
        owned = []
        for package in packages:
            if not 2 <= len(package.member_ids) <= 4:
                raise ValueError("response package size must be 2-4")
            owned.extend(package.member_ids)
            self._packages_by_index.setdefault(
                int(package.response_group_index), []
            ).append(package)
        if len(owned) != len(set(owned)):
            raise ValueError("response package members must be unique")
        self._packages_by_index = {
            index: tuple(sorted(values, key=lambda item: item.sequence_index))
            for index, values in self._packages_by_index.items()
        }
        self.reset()

    def reset(self) -> None:
        self._discoveries: list[int] = []
        self._discovery_completion_step: int | None = None
        self._released: set[str] = set()
        self._cancelled: set[str] = set()
        self._release_step: dict[str, int] = {}
        self._first_release_step_by_index: dict[int, int] = {}
        self._initial_package_count_by_index: dict[int, int] = {}

    def update_discoveries(self, *, step: int, target_ids: tuple[int, ...]) -> None:
        for target_id in target_ids:
            value = int(target_id)
            if value not in self._discoveries:
                self._discoveries.append(value)
        if (
            self._discovery_completion_step is None
            and len(self._discoveries) >= self.config.required_discoveries
        ):
            self._discovery_completion_step = int(step)

    def decide(
        self,
        *,
        step: int,
        beliefs: tuple[MissionTargetBelief, ...],
    ) -> MissionReplanDecision:
        beliefs_by_index = {
            int(item.response_group_index): item for item in beliefs
        }
        activation_step = (
            None
            if self._discovery_completion_step is None
            else self._discovery_completion_step + self.config.release_delay_steps
        )
        newly_released = []
        newly_cancelled = []
        for response_index, packages in sorted(self._packages_by_index.items()):
            belief = beliefs_by_index.get(response_index)
            if belief is None or activation_step is None or step < activation_step:
                continue
            target_satisfied = belief.confirmed_destroyed or (
                belief.expected_hp <= 0.0
                and belief.confidence >= max(
                    0.95, self.config.destroyed_confidence_threshold
                )
                and belief.delivered_expected_damage
                >= belief.initial_health * 1.25
            )
            pending = [
                package for package in packages
                if package.package_id not in self._released
                and package.package_id not in self._cancelled
            ]
            if target_satisfied:
                for package in pending:
                    self._cancelled.add(package.package_id)
                    newly_cancelled.append(package)
                continue
            released = [
                package for package in packages if package.package_id in self._released
            ]
            if not released and pending:
                total_members = sum(len(package.member_ids) for package in packages)
                desired = min(
                    total_members,
                    max(
                        self.config.minimum_initial_members,
                        math.ceil(
                            total_members * self.config.initial_release_fraction
                        ),
                    ),
                )
                released_members = 0
                while pending and released_members < desired:
                    package = pending.pop(0)
                    self._released.add(package.package_id)
                    self._release_step[package.package_id] = int(step)
                    newly_released.append(package)
                    released_members += len(package.member_ids)
                self._first_release_step_by_index[response_index] = int(step)
                self._initial_package_count_by_index[response_index] = sum(
                    package.package_id in self._released for package in packages
                )
                continue
            if not pending or belief.expected_hp <= 0.0:
                continue
            first_release = self._first_release_step_by_index.get(
                response_index,
                min(self._release_step[package.package_id] for package in released),
            )
            latest = max(
                self._release_step[package.package_id] for package in released
            )
            interval = (
                self.config.delivery_timeout_steps
                if len(released)
                <= self._initial_package_count_by_index.get(response_index, 1)
                else self.config.subsequent_release_interval_steps
            )
            hard_deadline = step - first_release >= self.config.hard_deadline_steps
            if hard_deadline:
                for package in pending:
                    self._released.add(package.package_id)
                    self._release_step[package.package_id] = int(step)
                    newly_released.append(package)
                continue
            if step - latest >= interval:
                package = pending[0]
                self._released.add(package.package_id)
                self._release_step[package.package_id] = int(step)
                newly_released.append(package)

        held = [
            package
            for packages in self._packages_by_index.values()
            for package in packages
            if package.package_id not in self._released
            and package.package_id not in self._cancelled
        ]
        return MissionReplanDecision(
            launch_package_ids=tuple(item.package_id for item in newly_released),
            launch_member_ids=tuple(
                member for item in newly_released for member in item.member_ids
            ),
            held_package_ids=tuple(item.package_id for item in held),
            held_member_ids=tuple(member for item in held for member in item.member_ids),
            cancelled_package_ids=tuple(
                item.package_id for item in newly_cancelled
            ),
            cancelled_member_ids=tuple(
                member for item in newly_cancelled for member in item.member_ids
            ),
            target_ids_by_response_index=tuple(
                (index, self._discoveries[index])
                for index in sorted(self._packages_by_index)
                if index < len(self._discoveries)
            ),
        )

    @property
    def released_member_ids(self) -> frozenset[int]:
        return frozenset(
            member
            for packages in self._packages_by_index.values()
            for package in packages
            if package.package_id in self._released
            for member in package.member_ids
        )

    @property
    def cancelled_member_ids(self) -> frozenset[int]:
        return frozenset(
            member
            for packages in self._packages_by_index.values()
            for package in packages
            if package.package_id in self._cancelled
            for member in package.member_ids
        )
