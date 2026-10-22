"""Stateful raw-observation encoder for the V13 33-feature Actor ABI."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _vector(value: Any) -> np.ndarray:
    return np.asarray([float(_field(value, axis, 0.0)) for axis in ("x", "y", "z")],
                      dtype=np.float64)


@dataclass(frozen=True)
class EnvironmentConfig:
    unit_speeds_mps: tuple[float, float, float]
    goal_distance_scale_m: float
    threat_distance_scale_m: float
    relative_speed_scale_mps: float
    t_cpa_scale_s: float
    horizon_steps: int
    risk_distance_m: float
    risk_t_cpa_s: float
    risk_d_cpa_m: float
    actor_risk_gate: bool
    actor_action_history: bool
    encode_track_source: bool

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "EnvironmentConfig":
        names = cls.__dataclass_fields__
        selected = {name: values[name] for name in names}
        selected["unit_speeds_mps"] = tuple(float(v) for v in selected["unit_speeds_mps"])
        return cls(**selected)


@dataclass
class _TrackAge:
    timestamp: int
    last_changed_step: int


class EvasionObservationEncoder:
    MAX_THREATS = 3
    THREAT_START = 15
    THREAT_FEATURES = 12

    def __init__(self, *, entity_id: int, max_steps: int, threat_range_m: float) -> None:
        self.entity_id = int(entity_id)
        self.max_steps = max(1, int(max_steps))
        self.threat_range_m = max(float(threat_range_m), 1.0)
        self._previous_position: np.ndarray | None = None
        self._previous_step: int | None = None
        self._cached_step: int | None = None
        self._cached_vector: np.ndarray | None = None
        self._cached_sources: tuple[float, ...] = ()
        self._track_ages: dict[int, _TrackAge] = {}

    @property
    def threat_sources(self) -> tuple[float, ...]:
        return self._cached_sources

    def reset(self) -> None:
        self._previous_position = None
        self._previous_step = None
        self._cached_step = None
        self._cached_vector = None
        self._cached_sources = ()
        self._track_ages.clear()

    def _fresh(self, track_id: int, timestamp: int, step: int) -> bool:
        age = self._track_ages.get(track_id)
        if age is None or timestamp != age.timestamp:
            self._track_ages[track_id] = _TrackAge(timestamp, step)
            return True
        return step - age.last_changed_step <= 3

    def encode(self, observation: Mapping[str, Any], *, launch_step: int,
               maneuver_state: int) -> np.ndarray:
        step = int(observation.get("step", 0))
        if self._cached_step == step and self._cached_vector is not None:
            return self._cached_vector.copy()
        own = observation.get("self") or {}
        position = _vector(own.get("pos_ecf") or {})
        if self._previous_position is None or self._previous_step is None or step <= self._previous_step:
            velocity = np.zeros(3, dtype=np.float64)
        else:
            velocity = (position - self._previous_position) / float(step - self._previous_step)
        speed = float(np.linalg.norm(velocity))
        direction = velocity / max(speed, 1e-6)
        up = position / max(float(np.linalg.norm(position)), 1e-6)
        right = np.cross(direction, up)
        if float(np.linalg.norm(right)) <= 1e-6:
            right = np.asarray([0.0, 1.0, 0.0])
        else:
            right /= np.linalg.norm(right)
        normal = np.cross(right, direction)
        normal /= max(float(np.linalg.norm(normal)), 1e-6)
        lla = own.get("position") or {}
        entity_type = int(own.get("type", 0))
        maneuver = int(np.clip(maneuver_state, -1, 1))
        features = [
            float(np.clip(step / self.max_steps, 0.0, 1.0)),
            float(entity_type == 21000), float(entity_type == 21001), float(entity_type == 21002),
            float(np.clip(float(own.get("health", 0.0)) / 100.0, 0.0, 1.0)),
            float(np.clip(float(lla.get("alt", 0.0)) / 20_000.0, -1.0, 1.0)),
            float(np.clip(float(own.get("stage", -1.0)) / 5.0, -0.2, 1.0)),
            float(np.clip(max(0, step - int(launch_step)) / self.max_steps, 0.0, 1.0)),
            float(maneuver == -1), float(maneuver == 0), float(maneuver == 1),
            float(np.clip(speed / 2_000.0, 0.0, 2.0)), *map(float, direction),
        ]
        communication_ids = {int(value) for value in (own.get("commRangeInfo") or [])}
        threats: list[tuple[float, float, list[float]]] = []
        for raw_id, track in (own.get("detectInfo") or {}).items():
            if int(_field(track, "entity_type", -1)) != 24000:
                continue
            detect_from = int(_field(track, "detect_from", -1))
            source = 1.0 / 3.0 if detect_from == self.entity_id else (
                2.0 / 3.0 if detect_from in communication_ids else 1.0)
            track_id = int(_field(track, "entity_id", raw_id))
            if not self._fresh(track_id, int(_field(track, "time", 0)), step):
                continue
            relative_position = _vector(_field(track, "pos_ecf", {})) - position
            distance = float(np.linalg.norm(relative_position))
            if distance <= 1e-6:
                continue
            line_of_sight = relative_position / distance
            relative_velocity = _vector(_field(track, "vel_ecf", {})) - velocity
            closing = max(0.0, -float(np.dot(relative_position, relative_velocity)) / distance)
            relative_speed_sq = float(np.dot(relative_velocity, relative_velocity))
            time_to_go = max(0.0, -float(np.dot(relative_position, relative_velocity)) /
                             max(relative_speed_sq, 1e-6))
            cpa_distance = float(np.linalg.norm(relative_position + relative_velocity * time_to_go))
            local_los = [np.dot(line_of_sight, axis) for axis in (direction, right, normal)]
            local_velocity = [np.dot(relative_velocity, axis) for axis in (direction, right, normal)]
            normalized_velocity = np.clip(np.asarray(local_velocity) / 2_000.0, -2.0, 2.0)
            threats.append((distance, source, [
                1.0, *map(float, local_los),
                float(np.clip(distance / self.threat_range_m, 0.0, 2.0)),
                *map(float, normalized_velocity),
                float(np.clip(closing / 2_000.0, 0.0, 2.0)),
                float(np.clip(time_to_go / 120.0, 0.0, 2.5)),
                float(np.clip(cpa_distance / self.threat_range_m, 0.0, 2.0)),
                float(closing > 0.0),
            ]))
        threats.sort(key=lambda item: item[0])
        self._cached_sources = tuple(item[1] for item in threats[:self.MAX_THREATS])
        for slot in range(self.MAX_THREATS):
            features.extend(threats[slot][2] if slot < len(threats) else [0.0] * self.THREAT_FEATURES)
        vector = np.asarray(features, dtype=np.float32)
        if vector.shape != (51,):
            raise RuntimeError(f"Unexpected base observation shape: {vector.shape}")
        self._previous_position, self._previous_step = position, step
        self._cached_step, self._cached_vector = step, vector
        return vector.copy()


class LocalAvoidRuntimeEncoder:
    def __init__(self, *, entity_id: int, max_steps: int, config: EnvironmentConfig) -> None:
        self.config = config
        self._base = EvasionObservationEncoder(
            entity_id=entity_id, max_steps=max_steps,
            threat_range_m=config.threat_distance_scale_m,
        )

    @staticmethod
    def _target_geometry(position: Mapping[str, Any], goal: Any) -> tuple[float, float]:
        lon, lat = float(position.get("lon", 0.0)), float(position.get("lat", 0.0))
        goal_lon, goal_lat = float(_field(goal, "lon", lon)), float(_field(goal, "lat", lat))
        mean_lat = math.radians((lat + goal_lat) / 2.0)
        east = (goal_lon - lon) * 111_320.0 * math.cos(mean_lat)
        north = (goal_lat - lat) * 110_570.0
        return math.hypot(east, north), math.atan2(north, east)

    @staticmethod
    def _heading(direction: np.ndarray, lon_deg: float, lat_deg: float) -> float | None:
        if float(np.linalg.norm(direction)) <= 1e-6:
            return None
        lon, lat = math.radians(lon_deg), math.radians(lat_deg)
        dx, dy, dz = map(float, direction)
        east = -math.sin(lon) * dx + math.cos(lon) * dy
        north = -math.sin(lat) * math.cos(lon) * dx - math.sin(lat) * math.sin(lon) * dy + math.cos(lat) * dz
        return None if math.hypot(east, north) <= 1e-6 else math.atan2(north, east)

    def encode(self, observation: Mapping[str, Any], *, launch_step: int,
               maneuver_state: int, goal_position: Any) -> np.ndarray:
        base = self._base.encode(observation, launch_step=launch_step,
                                 maneuver_state=maneuver_state)
        own = observation.get("self") or {}
        position = own.get("position") or {}
        distance, goal_heading = self._target_geometry(position, goal_position)
        heading = self._heading(base[12:15], float(position.get("lon", 0.0)),
                                float(position.get("lat", 0.0)))
        heading = goal_heading if heading is None else heading
        goal_angle = (goal_heading - heading + math.pi) % (2.0 * math.pi) - math.pi
        type_slot = {21000: 0, 21001: 1, 21002: 2}.get(int(own.get("type", 0)), 0)
        features = [
            self.config.unit_speeds_mps[type_slot] / max(self.config.unit_speeds_mps),
            math.sin(heading), math.cos(heading),
            float(np.clip(distance / self.config.goal_distance_scale_m, 0.0, 2.0)),
            math.sin(goal_angle), math.cos(goal_angle),
        ]
        threats: list[tuple[float, float, list[float]]] = []
        for slot in range(EvasionObservationEncoder.MAX_THREATS):
            offset = EvasionObservationEncoder.THREAT_START + slot * EvasionObservationEncoder.THREAT_FEATURES
            if base[offset] <= 0.5:
                continue
            threat_distance = float(base[offset + 4] * self.config.threat_distance_scale_m)
            bearing_sin, bearing_cos = float(-base[offset + 2]), float(base[offset + 1])
            bearing = math.atan2(bearing_sin, bearing_cos)
            relative_speed = float(np.linalg.norm(base[offset + 5:offset + 8]) * 2_000.0)
            closing = float(base[offset + 8] * 2_000.0)
            time_to_cpa = float(base[offset + 9] * 120.0)
            cpa_distance = float(base[offset + 10] * self.config.threat_distance_scale_m)
            risky = threat_distance <= self.config.risk_distance_m or (
                closing > 0.0 and time_to_cpa <= self.config.risk_t_cpa_s
                and cpa_distance <= self.config.risk_d_cpa_m)
            if self.config.actor_risk_gate and not risky:
                continue
            source = self._base.threat_sources[slot] if slot < len(self._base.threat_sources) else 1.0 / 3.0
            threats.append((bearing, source, [
                1.0, float(np.clip(threat_distance / self.config.threat_distance_scale_m, 0.0, 2.0)),
                bearing_sin, bearing_cos,
                float(np.clip(relative_speed / self.config.relative_speed_scale_mps, 0.0, 2.0)),
                float(np.clip(closing / self.config.relative_speed_scale_mps, -2.0, 2.0)),
                float(np.clip(time_to_cpa / self.config.t_cpa_scale_s, 0.0, 2.0)),
                float(np.clip(cpa_distance / self.config.threat_distance_scale_m, 0.0, 2.0)),
                1.0 / 3.0,
            ]))
        threats.sort(key=lambda item: item[0])
        sources = [0.0, 0.0]
        for slot in range(2):
            if slot < len(threats):
                source = threats[slot][1] if self.config.encode_track_source else 1.0 / 3.0
                values = threats[slot][2]
                values[-1] = source
                features.extend(values)
                sources[slot] = source
            else:
                features.extend([0.0] * 9)
        action_context = [0.0, 0.0, 0.0]
        if self.config.actor_action_history:
            action_context[int(np.clip(maneuver_state + 1, 0, 2))] = 1.0
        elapsed = max(0, int(observation.get("step", 0)) - int(launch_step))
        features.extend([*action_context, *sources,
                         float(np.clip(elapsed / max(self.config.horizon_steps, 1), 0.0, 1.0))])
        features.extend(float(type_slot == index) for index in range(3))
        result = np.asarray(features, dtype=np.float32)
        if result.shape != (33,):
            raise RuntimeError(f"Unexpected V13 observation shape: {result.shape}")
        return result

    def reset(self) -> None:
        self._base.reset()
