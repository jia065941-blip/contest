"""Stable, engine-independent data contract for all baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Position:
    lon: float
    lat: float
    alt: float = 0.0


@dataclass(frozen=True)
class Platform:
    entity_id: int
    kind: str
    position: Position
    alive: bool = True
    launched: bool = False


@dataclass(frozen=True)
class Target:
    entity_id: int
    position: Position
    value: float = 1.0
    health: float = 1.0
    alive: bool = True


@dataclass(frozen=True)
class Observation:
    step: int
    platforms: tuple[Platform, ...]
    targets: tuple[Target, ...]
    satellite_remaining: int = 0

    def active_platforms(self) -> tuple[Platform, ...]:
        return tuple(p for p in self.platforms if p.alive and not p.launched)

    def active_targets(self) -> tuple[Target, ...]:
        return tuple(t for t in self.targets if t.alive and t.health > 0)


@dataclass(frozen=True)
class DeploymentSlot:
    position: Position
    allowed_kinds: tuple[str, ...] = ()

    def accepts(self, platform: Platform) -> bool:
        return not self.allowed_kinds or platform.kind in self.allowed_kinds


@dataclass(frozen=True)
class GlobalRules:
    target_capacity: int = 2
    min_launch_interval: int = 1
    satellite_budget: int = 0
    wave_by_kind: Mapping[str, int] = field(default_factory=lambda: {"H": 0, "M": 10, "L": 20})
    value_weight: float = 1.0
    distance_weight: float = 0.25
    coverage_weight: float = 0.75
    cost_by_kind: Mapping[str, float] = field(default_factory=dict)
    replan_interval: int = 10

    def __post_init__(self) -> None:
        if self.target_capacity < 1:
            raise ValueError("target_capacity must be positive")
        if self.min_launch_interval < 0 or self.satellite_budget < 0 or self.replan_interval < 1:
            raise ValueError("rule bounds must be non-negative and replan_interval positive")


@dataclass(frozen=True)
class Assignment:
    platform_id: int
    target_id: int
    launch_step: int
    score: float


@dataclass(frozen=True)
class Decision:
    assignments: tuple[Assignment, ...] = ()
    satellite_users: tuple[int, ...] = ()
    deployment: Mapping[int, Position] = field(default_factory=dict)

    def assigned_target_ids(self) -> set[int]:
        return {assignment.target_id for assignment in self.assignments}


def by_id(items: Sequence[Platform | Target]) -> dict[int, Platform | Target]:
    return {item.entity_id: item for item in items}

