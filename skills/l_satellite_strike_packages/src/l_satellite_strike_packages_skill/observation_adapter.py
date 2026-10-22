"""Convert caller-owned isolated observations into immutable legal track DTOs."""

from __future__ import annotations

from typing import Mapping

from .controller import DarkNestTrack, HIDDEN_NEST_TYPE, L_MISSILE_TYPE


def tracks_from_isolated_observations(
    observations: Mapping[int, Mapping], *, step: int,
) -> tuple[DarkNestTrack, ...]:
    """Extract only 9500 tracks reported by the corresponding L platform."""
    newest: dict[int, DarkNestTrack] = {}
    for entity_id in sorted(observations):
        observation = observations[entity_id]
        own = observation.get("self", {})
        own_id = int(_field(own, "entity_id", entity_id))
        if own_id != int(entity_id):
            raise ValueError("Observation key must match self.entity_id")
        if int(_field(own, "type", -1)) != L_MISSILE_TYPE:
            raise ValueError("Only isolated L missile observations are accepted")
        for raw_id, value in (_field(own, "detectInfo", {}) or {}).items():
            if int(_field(value, "entity_type", -1)) != HIDDEN_NEST_TYPE:
                continue
            lla = _field(value, "lla", None)
            if lla is None:
                continue
            target_id = int(_field(value, "entity_id", raw_id))
            track = DarkNestTrack(
                target_id=target_id,
                lon=float(_field(lla, "x", 0.0)),
                lat=float(_field(lla, "y", 0.0)),
                observed_step=float(_field(value, "time", step)),
            )
            previous = newest.get(target_id)
            if previous is None or track.observed_step > previous.observed_step:
                newest[target_id] = track
    return tuple(sorted(newest.values(), key=lambda item: item.target_id))


def _field(value, name: str, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)
