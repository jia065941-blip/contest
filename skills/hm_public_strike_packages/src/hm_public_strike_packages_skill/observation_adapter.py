"""Build and refresh public tracks from caller-owned legal observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite

from .controller import (
    H_MISSILE_TYPE,
    M_MISSILE_TYPE,
    PUBLIC_TARGET_TYPES,
    PublicStrikeConfig,
    PublicTargetTrack,
)


def public_targets_from_directory(
    directory: Mapping[int, object] | Sequence[object],
    *,
    observed_step: float = 0.0,
    config: PublicStrikeConfig | None = None,
) -> tuple[PublicTargetTrack, ...]:
    """Convert a caller-supplied public directory, never simulator state, to DTOs."""
    settings = config or PublicStrikeConfig()
    if (
        isinstance(observed_step, bool)
        or not isinstance(observed_step, (int, float))
        or not isfinite(float(observed_step))
        or observed_step < 0
    ):
        raise ValueError("observed_step must be nonnegative and finite")
    entries = directory.items() if isinstance(directory, Mapping) else enumerate(directory)
    tracks: list[PublicTargetTrack] = []
    for raw_id, value in entries:
        target_id = int(_field(value, "entity_id", _field(value, "target_id", raw_id)))
        if isinstance(directory, Mapping) and target_id != int(raw_id):
            raise ValueError("Directory key must match target entity_id")
        target_type = int(_field(value, "entity_type", _field(value, "target_type", -1)))
        if target_type not in PUBLIC_TARGET_TYPES:
            raise ValueError("Public directory accepts only target types 9400/9600")
        lon, lat = _coordinates(value)
        tracks.append(PublicTargetTrack(
            target_id=target_id,
            target_type=target_type,
            lon=lon,
            lat=lat,
            observed_step=float(_field(value, "time", observed_step)),
            initial_health=float(_field(
                value, "initial_health", settings.default_health(target_type),
            )),
            value=float(_field(value, "value", settings.default_value(target_type))),
        ))
    ordered = tuple(sorted(tracks, key=lambda item: item.target_id))
    if len({item.target_id for item in ordered}) != len(ordered):
        raise ValueError("Public directory target IDs must be unique")
    return ordered


def track_updates_from_isolated_observations(
    observations: Mapping[int, object],
    *,
    public_targets: Sequence[PublicTargetTrack],
    step: int,
) -> tuple[PublicTargetTrack, ...]:
    """Extract only known 9400/9600 tracks observed by isolated H/M owners."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    known = {item.target_id: item for item in public_targets}
    if len(known) != len(public_targets):
        raise ValueError("public_targets must have unique target IDs")
    newest: dict[int, PublicTargetTrack] = {}
    for entity_id in sorted(observations):
        observation = observations[entity_id]
        own = _field(observation, "self", {})
        own_id = int(_field(own, "entity_id", entity_id))
        if own_id != int(entity_id):
            raise ValueError("Observation key must match self.entity_id")
        if int(_field(own, "type", -1)) not in (H_MISSILE_TYPE, M_MISSILE_TYPE):
            raise ValueError("Only isolated H/M missile observations are accepted")
        detect_info = _field(own, "detectInfo", {}) or {}
        if not isinstance(detect_info, Mapping):
            raise ValueError("detectInfo must be a mapping")
        for raw_id, value in sorted(detect_info.items(), key=lambda item: int(item[0])):
            target_type = int(_field(value, "entity_type", -1))
            if target_type not in PUBLIC_TARGET_TYPES:
                continue
            target_id = int(_field(value, "entity_id", raw_id))
            if target_id != int(raw_id):
                raise ValueError("detectInfo key must match entity_id")
            original = known.get(target_id)
            if original is None:
                continue
            if target_type != original.target_type:
                raise ValueError("Detected target type does not match public directory")
            lon, lat = _coordinates(value)
            update = PublicTargetTrack(
                target_id=target_id,
                target_type=target_type,
                lon=lon,
                lat=lat,
                observed_step=float(_field(value, "time", step)),
                initial_health=original.initial_health,
                value=original.value,
            )
            previous = newest.get(target_id)
            if previous is not None and update.observed_step == previous.observed_step:
                if (update.lon, update.lat) != (previous.lon, previous.lat):
                    raise ValueError("Conflicting coordinates at the same observed_step")
            if previous is None or update.observed_step > previous.observed_step:
                newest[target_id] = update
    return tuple(sorted(newest.values(), key=lambda item: item.target_id))


tracks_from_isolated_observations = track_updates_from_isolated_observations


def _coordinates(value: object) -> tuple[float, float]:
    lla = _field(value, "lla", None)
    if lla is not None:
        lon = _field(lla, "x", None)
        lat = _field(lla, "y", None)
    else:
        position = _field(value, "position", None)
        source = position if position is not None else value
        lon = _field(source, "lon", None)
        lat = _field(source, "lat", None)
    if lon is None or lat is None:
        raise ValueError("A public target coordinate requires lon/lat or lla.x/lla.y")
    lon_value, lat_value = float(lon), float(lat)
    if not isfinite(lon_value) or not isfinite(lat_value):
        raise ValueError("Public target coordinates must be finite")
    return lon_value, lat_value


def _field(value: object, name: str, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)
