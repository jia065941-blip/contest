"""Convert semantic deployment points into caller-owned runtime action rows."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import TypeAlias

from .controller import DeploymentPlan


DeploymentAction: TypeAlias = tuple[int, int, float, float, float]


def to_deployment_actions(plan: DeploymentPlan) -> tuple[DeploymentAction, ...]:
    """Return centralized ``(0, id, lon, lat, altitude)`` deployment rows."""
    if not isinstance(plan, DeploymentPlan):
        raise ValueError("plan must be DeploymentPlan")
    return tuple(
        (
            0,
            point.entity_id,
            point.longitude,
            point.latitude,
            point.altitude,
        )
        for point in plan.points
    )


def route_deployment_actions(
    plan: DeploymentPlan,
    owners: Mapping[int, Hashable],
) -> dict[Hashable, tuple[DeploymentAction, ...]]:
    """Group action rows by an external entity-to-owner mapping."""
    if not isinstance(owners, Mapping):
        raise ValueError("owners must be a mapping from entity ID to owner")
    actions = to_deployment_actions(plan)
    grouped: dict[Hashable, list[DeploymentAction]] = {}
    for action in actions:
        entity_id = action[1]
        if entity_id not in owners:
            raise ValueError(f"missing owner for entity {entity_id}")
        owner = owners[entity_id]
        try:
            grouped.setdefault(owner, []).append(action)
        except TypeError as exc:
            raise ValueError(f"owner for entity {entity_id} must be hashable") from exc
    return {owner: tuple(rows) for owner, rows in grouped.items()}


__all__ = [
    "DeploymentAction",
    "route_deployment_actions",
    "to_deployment_actions",
]
