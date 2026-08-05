"""Deployment slot construction and deterministic slot assignment."""

from __future__ import annotations

from .domain import DeploymentSlot, Platform, Position
from .geometry import spaced_points


def make_slots(polygon: list[Position], count: int, min_distance_km: float) -> tuple[DeploymentSlot, ...]:
    return tuple(DeploymentSlot(position=point) for point in spaced_points(polygon, count, min_distance_km))


def assign_slots(platforms: tuple[Platform, ...], slots: tuple[DeploymentSlot, ...]) -> dict[int, Position]:
    available = list(slots)
    result: dict[int, Position] = {}
    for platform in sorted(platforms, key=lambda item: item.entity_id):
        index = next((i for i, slot in enumerate(available) if slot.accepts(platform)), None)
        if index is None:
            raise ValueError(f"no deployment slot for platform {platform.entity_id}")
        result[platform.entity_id] = available.pop(index).position
    return result

