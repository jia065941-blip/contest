"""Materialize semantic H/M decisions as owner-routed engine action rows."""

from __future__ import annotations

from .controller import ACTION_ATTACK_TARGET, ACTION_SET_ACC_Z, PublicStrikeDecision


EngineAction = tuple[float, float, float, float]


def materialize_engine_actions(
    decision: PublicStrikeDecision,
) -> dict[int, tuple[EngineAction, ...]]:
    """Map categories to -1/0/1 and append attacks under each action owner."""
    routed: dict[int, list[EngineAction]] = {}
    seen_baseline: set[int] = set()
    for entity_id, category in decision.baseline_actions:
        if entity_id in seen_baseline:
            raise ValueError("Baseline action owners must be unique")
        if isinstance(category, bool) or category not in (0, 1, 2):
            raise ValueError("Baseline maneuver category must be 0, 1, or 2")
        seen_baseline.add(entity_id)
        routed.setdefault(entity_id, []).append((
            float(ACTION_SET_ACC_Z), float(entity_id), float(category - 1), 0.0,
        ))
    for request in decision.attack_actions:
        if request.entity_id not in seen_baseline:
            raise ValueError("Attack request owner must have an active baseline action")
        routed.setdefault(request.entity_id, []).append((
            float(ACTION_ATTACK_TARGET), float(request.entity_id),
            float(request.target_id), 0.0,
        ))
    return {entity_id: tuple(actions) for entity_id, actions in sorted(routed.items())}
