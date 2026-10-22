"""Materialize semantic L package decisions as owner-routed engine rows."""

from __future__ import annotations

from math import isfinite

from .controller import (
    ACTION_ATTACK_TARGET,
    ACTION_SET_ACC_Z,
    ACTION_USE_SATELLITE,
    SatelliteStrikeDecision,
)

EngineAction = tuple[float, float, float, float]


def materialize_engine_actions(
    decision: SatelliteStrikeDecision,
    *,
    maneuver_values: tuple[float, float, float] = (-1.0, 0.0, 1.0),
) -> dict[int, tuple[EngineAction, ...]]:
    """Return commands keyed by their owning L entity for per-agent dispatch."""
    if len(maneuver_values) != 3 or not all(isfinite(value) for value in maneuver_values):
        raise ValueError("maneuver_values must contain three finite values")
    routed: dict[int, list[EngineAction]] = {}
    for entity_id, category in decision.baseline_actions:
        if category not in (0, 1, 2):
            raise ValueError("Baseline maneuver category must be 0, 1, or 2")
        routed.setdefault(entity_id, []).append((
            float(ACTION_SET_ACC_Z), float(entity_id),
            float(maneuver_values[category]), 0.0,
        ))
    if decision.satellite_request is not None:
        entity_id = decision.satellite_request.requester_id
        routed.setdefault(entity_id, []).append((
            float(ACTION_USE_SATELLITE), float(entity_id), 0.0, 0.0,
        ))
    for request in decision.attack_actions:
        routed.setdefault(request.entity_id, []).append((
            float(ACTION_ATTACK_TARGET), float(request.entity_id),
            float(request.target_id), 0.0,
        ))
    return {entity_id: tuple(actions) for entity_id, actions in sorted(routed.items())}
