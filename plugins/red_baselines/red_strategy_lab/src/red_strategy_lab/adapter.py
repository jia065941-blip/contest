"""Boundary between the experiment package and the existing simulation environment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .domain import Decision, Observation, Platform, Position, Target

RED_SIDE = 0
AIRCRAFT_TYPES = {21000, 21001, 21002}


def observation_from_engine(raw: Mapping[str, object], initial_targets: Sequence[Target], launched_ids: set[int] | None = None) -> Observation:
    """Convert the public training observation format into the lab contract."""
    entities = raw.get("entities", {})
    if not isinstance(entities, Mapping):
        raise ValueError("observation entities must be a mapping")
    platforms: list[Platform] = []
    launched_ids = launched_ids or set()
    for raw_id, data in entities.items():
        if not isinstance(data, Mapping) or data.get("side") != RED_SIDE or data.get("type") not in AIRCRAFT_TYPES:
            continue
        position = data.get("position", {})
        if not isinstance(position, Mapping):
            continue
        platforms.append(Platform(
            entity_id=int(raw_id),
            kind={21000: "H", 21001: "M", 21002: "L"}[int(data["type"])],
            position=Position(float(position["lon"]), float(position["lat"]), float(position.get("alt", 0.0))),
            alive=float(data.get("health", 0.0)) > 0,
            launched=int(raw_id) in launched_ids,
        ))
    return Observation(step=int(raw.get("step", 0)), platforms=tuple(platforms), targets=tuple(initial_targets))


def launch_rows(decision: Decision, targets: Sequence[Target], current_step: int) -> list[list[float]]:
    """Return rows accepted by the existing CommandConverter launch branch."""
    target_map = {target.entity_id: target for target in targets}
    rows: list[list[float]] = []
    for assignment in decision.assignments:
        if assignment.launch_step != current_step:
            continue
        target = target_map.get(assignment.target_id)
        if target is not None:
            rows.append([1.0, float(assignment.platform_id), target.position.lon, target.position.lat])
    return rows
