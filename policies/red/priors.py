"""Mission priors explicitly provided by core before red-side search begins."""

from __future__ import annotations

from .contracts import Position
from .baselines import TargetPrior


_VALUE_BY_TYPE = {9400: 10.0, 9600: 6.0, 9500: 3.0}


def initial_targets_from_observation(initial_observation: dict) -> tuple[TargetPrior, ...]:
    """Convert the complete core-provided initialization catalogue as-is."""

    targets: list[TargetPrior] = []
    for raw_id, entity in initial_observation.get("entities", {}).items():
        location = entity.get("position", {})
        entity_type = int(entity["type"])
        targets.append(
            TargetPrior(
                entity_id=int(raw_id),
                entity_type=entity_type,
                position=Position(
                    lon=float(location.get("lon", 0.0)),
                    lat=float(location.get("lat", 0.0)),
                    alt=float(location.get("alt", 0.0)),
                ),
                value=_VALUE_BY_TYPE.get(entity_type, 1.0),
            )
        )
    return tuple(sorted(targets, key=lambda item: item.entity_id))
