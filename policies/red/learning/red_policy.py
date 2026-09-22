"""红方学习基线所需的观测、动作和共享策略抽象。

高层动作空间固定为：等待、对 5 个槽位发射、对 5 个槽位改目标、
左/右/停止机动、使用卫星。适配器负责将高层动作转换为仿真引擎已有的
四元动作，因此学习算法不需要直接回归经纬度。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_TARGET_SLOTS = 5
UNIFIED_TARGET_SLOTS = 25
UNIFIED_AGENT_IDENTITY_SLOTS = 210
UNIFIED_LOCAL_OBSERVATION_DIM = (
    21 + UNIFIED_AGENT_IDENTITY_SLOTS + UNIFIED_TARGET_SLOTS * 15 + 4
)
UNIFIED_TEAM_CONTEXT_DIM = 15
UNIFIED_TEAM_OBSERVATION_DIM = (
    UNIFIED_LOCAL_OBSERVATION_DIM + UNIFIED_TEAM_CONTEXT_DIM
)
OBJECTIVE_ENTITY_TYPES = frozenset({9400, 9500, 9600})
INTERCEPTOR_ENTITY_TYPE = 24000
INTERCEPTOR_TRACK_COUNT_NORM = 148.0
INTERCEPTOR_DISTANCE_NORM_M = 1_000_000.0
_VECTOR_EPSILON = 1e-9


def _is_objective(entity: Mapping[str, Any]) -> bool:
    name = str(entity.get("nameChn", ""))
    return (
        int(entity.get("type", -1)) in OBJECTIVE_ENTITY_TYPES
        or name.startswith(("目标", "拦截阵地", "无人船"))
    )


class HighLevelAction(IntEnum):
    MANEUVER_LEFT = 0
    STOP_MANEUVER = 1
    MANEUVER_RIGHT = 2


ACTION_DIM = len(HighLevelAction)


@dataclass(frozen=True)
class PolicyTransition:
    """一条单智能体经验；共享策略会收到所有红方导弹的经验。"""

    agent_id: int
    observation: np.ndarray
    action: int
    action_mask: np.ndarray
    reward: float
    next_observation: np.ndarray
    done: bool
    global_state: np.ndarray | None = None
    next_global_state: np.ndarray | None = None


class SharedPolicy(ABC):
    """参数共享策略接口。

    PPO/MAPPO 实现只需实现 ``select_action``，并按需覆盖其余钩子。
    """

    @abstractmethod
    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        """在合法动作中选择一个高层动作。"""

    def observe(self, transition: PolicyTransition) -> None:
        """接收环境 transition；无在线学习需求的策略可以忽略。"""

    def reset_episode(self) -> None:
        """一轮开始时调用。共享策略应能安全地被多个 Agent 重复调用。"""

    def begin_environment_step(self, full_observation: Mapping[str, Any]) -> None:
        """联合动作生成前保存全局状态的可选钩子。"""

    def end_environment_step(self, full_observation: Mapping[str, Any]) -> None:
        """环境步进后保存下一全局状态的可选钩子。"""

    def finish_environment_step(self) -> None:
        """所有智能体 transition 均提交后的可选钩子。"""

    def get_global_state_context(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        return None, None

    def save(self, path: str) -> None:
        raise NotImplementedError("该策略没有实现模型保存")

    def load(self, path: str) -> None:
        raise NotImplementedError("该策略没有实现模型加载")


class RandomMaskedPolicy(SharedPolicy):
    """无需深度学习依赖的烟雾测试策略。

    它不是性能基线，而是用来验证观测、掩码、动作转换和经验回传链路。
    """

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)
        self.transition_count = 0

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        del observation
        valid_actions = np.flatnonzero(action_mask)
        if valid_actions.size == 0:
            return int(HighLevelAction.STOP_MANEUVER)
        return int(self._rng.choice(valid_actions))

    def observe(self, transition: PolicyTransition) -> None:
        del transition
        self.transition_count += 1


class ObservationEncoder:
    """将可变结构的原始观测编码为固定长度 float32 向量。"""

    SELF_FEATURES = 21
    IDENTITY_FEATURES = UNIFIED_AGENT_IDENTITY_SLOTS
    TARGET_FEATURES = 12
    TARGET_RUNTIME_FEATURES = 3
    DETECTION_FEATURES = 4
    TASK_FEATURES = 5

    def __init__(
        self,
        init_observation: Mapping[str, Any],
        target_slots: int = DEFAULT_TARGET_SLOTS,
        max_steps: int = 1000,
        coordinate_scale: float = 1.0,
        agent_id: int = 0,
        team_size: int = 58,
        hierarchical_task_context: bool = False,
        locally_observed_target_types: Sequence[int] = (),
        include_target_runtime_state: bool = False,
        include_agent_identity: bool = False,
    ):
        self.target_slots = int(target_slots)
        self.max_steps = max(1, max_steps)
        self.coordinate_scale = max(float(coordinate_scale), 1e-6)
        self.agent_id = int(agent_id)
        self.team_size = max(1, int(team_size))
        self.hierarchical_task_context = bool(hierarchical_task_context)
        self.include_target_runtime_state = bool(include_target_runtime_state)
        self.include_agent_identity = bool(include_agent_identity)
        self.locally_observed_target_types = frozenset(
            int(value) for value in locally_observed_target_types
        )
        self._local_target_memory: dict[int, tuple[float, float, int]] = {}
        self._target_runtime_states: dict[int, tuple[float, float, float]] = {}
        self.last_entity_type = -1
        self.last_interceptor_track_count = 0
        self.last_interceptor_satellite_source: int | None = None
        self.last_target_coordinates = np.zeros(
            (self.target_slots, 2), dtype=np.float32
        )
        self.last_target_valid_mask = np.zeros(
            self.target_slots, dtype=np.bool_
        )
        entities = init_observation.get("entities", {})
        self.targets = sorted(
            (
                dict(value, entity_id=int(key))
                for key, value in entities.items()
                if _is_objective(value)
            ),
            key=lambda item: item["entity_id"],
        )[:target_slots]

    @property
    def target_feature_dim(self) -> int:
        return self.TARGET_FEATURES + (
            self.TARGET_RUNTIME_FEATURES
            if self.include_target_runtime_state else 0
        )

    @property
    def observation_dim(self) -> int:
        base_dim = (
            self.SELF_FEATURES
            + (self.IDENTITY_FEATURES if self.include_agent_identity else 0)
            + self.target_slots * self.target_feature_dim
            + self.DETECTION_FEATURES
        )
        return base_dim + (self.TASK_FEATURES if self.hierarchical_task_context else 0)

    @staticmethod
    def _field(value: Any, name: str, default: Any) -> Any:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    @classmethod
    def _vector3(cls, value: Any) -> np.ndarray | None:
        if value is None:
            return None
        components = (
            cls._field(value, "x", None),
            cls._field(value, "y", None),
            cls._field(value, "z", None),
        )
        if any(component is None for component in components):
            return None
        vector = np.asarray(components, dtype=np.float64)
        return vector if bool(np.isfinite(vector).all()) else None

    @staticmethod
    def _local_components(
        vector_ecf: np.ndarray,
        *,
        lon: float,
        lat: float,
    ) -> tuple[float, float, float]:
        longitude = np.radians(float(lon))
        latitude = np.radians(float(lat))
        sin_lon, cos_lon = np.sin(longitude), np.cos(longitude)
        sin_lat, cos_lat = np.sin(latitude), np.cos(latitude)
        x, y, z = (float(value) for value in vector_ecf)
        east = -sin_lon * x + cos_lon * y
        north = (
            -sin_lat * cos_lon * x
            - sin_lat * sin_lon * y
            + cos_lat * z
        )
        up = (
            cos_lat * cos_lon * x
            + cos_lat * sin_lon * y
            + sin_lat * z
        )
        return float(east), float(north), float(up)

    @classmethod
    def _self_motion(
        cls,
        self_info: Mapping[str, Any],
        *,
        lon: float,
        lat: float,
    ) -> tuple[float, float, float]:
        velocity = self_info.get("velocity") or {}
        speed = float(velocity.get("speed", 0.0))
        vertical_speed = float(velocity.get("up", 0.0))
        heading = float(velocity.get("heading", 0.0))
        velocity_ecf = cls._vector3(self_info.get("vel_ecf"))
        if velocity_ecf is None:
            return speed, vertical_speed, heading
        east, north, up = cls._local_components(
            velocity_ecf,
            lon=lon,
            lat=lat,
        )
        speed = float(np.linalg.norm(velocity_ecf))
        vertical_speed = up
        if float(np.hypot(east, north)) > _VECTOR_EPSILON:
            heading = float(np.arctan2(east, north))
        return speed, vertical_speed, heading

    @classmethod
    def _track_age_steps(
        cls,
        detection: Any,
        *,
        step: int,
        sim_time: Any,
        sim_step: Any,
    ) -> int:
        detection_time = float(cls._field(detection, "time", 0.0))
        elapsed = max(0.0, float(sim_time) - detection_time)
        return max(0, int(np.ceil(elapsed / float(sim_step))) - 1)

    def set_targets(self, targets: Sequence[Mapping[str, Any]]) -> None:
        """Install a stable target catalogue without silently dropping slots."""

        normalized = [dict(target) for target in targets]
        if len(normalized) > self.target_slots:
            raise ValueError(
                f"目标数量 {len(normalized)} 超过固定容量 {self.target_slots}"
            )
        self.targets = normalized
        valid_ids = {int(target["entity_id"]) for target in normalized}
        self._local_target_memory = {
            target_id: state
            for target_id, state in self._local_target_memory.items()
            if target_id in valid_ids
        }
        self._target_runtime_states = {
            target_id: state
            for target_id, state in self._target_runtime_states.items()
            if target_id in valid_ids
        }

    def set_target_runtime_states(
        self,
        states: Mapping[int, Sequence[float]],
    ) -> None:
        """Install current state only for targets in the legal actor catalogue.

        The caller may hold centralized simulator state, but this method drops
        every entity that has not already passed the actor's discovery gate.
        """

        legal_ids = {
            int(target["entity_id"])
            for target in self.targets
            if bool(target.get("actor_visible", True))
            and not bool(target.get("slot_empty", False))
        }
        normalized: dict[int, tuple[float, float, float]] = {}
        for raw_target_id, raw_state in states.items():
            target_id = int(raw_target_id)
            if target_id not in legal_ids:
                continue
            values = tuple(float(value) for value in raw_state)
            if len(values) != 3:
                raise ValueError(
                    f"目标 {target_id} 的运行状态必须为 3 维，实际为 {len(values)}"
                )
            normalized[target_id] = (
                float(np.clip(values[0], 0.0, 1.0)),
                float(np.clip(values[1], 0.0, 1.0)),
                float(bool(values[2])),
            )
        self._target_runtime_states = normalized

    def reset_episode(self) -> None:
        """Forget private tracks so discoveries never leak between episodes."""

        self._local_target_memory.clear()
        self._target_runtime_states.clear()
        self.last_entity_type = -1
        self.last_interceptor_track_count = 0
        self.last_interceptor_satellite_source = None
        self.last_target_coordinates.fill(0.0)
        self.last_target_valid_mask.fill(False)

    def _detection_position(
        self,
        detection: Any,
        *,
        step: int,
        sim_time: Any = None,
        sim_step: Any = None,
    ) -> tuple[float, float, int] | None:
        lla = self._field(detection, "lla", None)
        if lla is not None:
            raw_lon = self._field(lla, "x", None)
            raw_lat = self._field(lla, "y", None)
        else:
            position = self._field(detection, "position", None)
            raw_lon = (
                self._field(position, "lon", None)
                if position is not None else None
            )
            raw_lat = (
                self._field(position, "lat", None)
                if position is not None else None
            )
        if raw_lon is None or raw_lat is None:
            return None
        age_steps = self._track_age_steps(
            detection,
            step=step,
            sim_time=sim_time,
            sim_step=sim_step,
        )
        detection_step = max(0, int(step) - age_steps)
        return float(raw_lon), float(raw_lat), detection_step

    def _interceptor_threat_summary(
        self,
        observation: Mapping[str, Any],
        *,
        lon: float,
        lat: float,
        heading: float,
    ) -> tuple[float, float, float, float]:
        self_info = observation.get("self") or {}
        self.last_interceptor_satellite_source = None
        detect_info = self_info.get("detectInfo") or {}
        interceptor_tracks = [
            track
            for track in detect_info.values()
            if int(self._field(track, "entity_type", -1))
            == INTERCEPTOR_ENTITY_TYPE
        ]
        self.last_interceptor_track_count = len(interceptor_tracks)
        if not interceptor_tracks:
            return 0.0, 0.0, 0.0, 0.0

        own_position = self._vector3(self_info.get("pos_ecf"))
        own_velocity = self._vector3(self_info.get("vel_ecf"))
        candidates: list[
            tuple[float | None, float | None, float | None, int | None]
        ] = []
        for track in interceptor_tracks:
            track_position = self._vector3(
                self._field(track, "pos_ecf", None)
            )
            track_velocity = self._vector3(
                self._field(track, "vel_ecf", None)
            )
            relative_position = (
                track_position - own_position
                if track_position is not None and own_position is not None
                else None
            )
            distance_m: float | None = None
            bearing: float | None = None
            closest_approach_time: float | None = None
            if relative_position is not None:
                distance_m = float(np.linalg.norm(relative_position))
                east, north, _ = self._local_components(
                    relative_position,
                    lon=lon,
                    lat=lat,
                )
                if float(np.hypot(east, north)) > _VECTOR_EPSILON:
                    bearing = float(np.arctan2(east, north))
                if own_velocity is not None and track_velocity is not None:
                    relative_velocity = track_velocity - own_velocity
                    speed_squared = float(
                        np.dot(relative_velocity, relative_velocity)
                    )
                    closing_dot = float(
                        np.dot(relative_position, relative_velocity)
                    )
                    if speed_squared > _VECTOR_EPSILON and closing_dot < 0.0:
                        estimate = -closing_dot / speed_squared
                        if np.isfinite(estimate) and estimate >= 0.0:
                            closest_approach_time = float(estimate)
            else:
                track_lla = self._field(track, "lla", None)
                track_lon = self._field(track_lla, "x", None)
                track_lat = self._field(track_lla, "y", None)
                if track_lon is not None and track_lat is not None:
                    delta_lon = float(track_lon) - lon
                    delta_lat = float(track_lat) - lat
                    mean_latitude = np.radians((float(track_lat) + lat) / 2.0)
                    east_km = delta_lon * 111.32 * np.cos(mean_latitude)
                    north_km = delta_lat * 110.57
                    distance_m = float(np.hypot(east_km, north_km) * 1000.0)
                    if float(np.hypot(east_km, north_km)) > _VECTOR_EPSILON:
                        bearing = float(np.arctan2(east_km, north_km))
            satellite_source = (
                int(self._field(track, "detect_from", -1))
                if bool(self._field(track, "via_satellite", False))
                else None
            )
            candidates.append((
                closest_approach_time, distance_m, bearing, satellite_source
            ))

        approaching = [item for item in candidates if item[0] is not None]
        if approaching:
            selected = min(
                approaching,
                key=lambda item: (
                    float(item[0]),
                    float(item[1]) if item[1] is not None else float("inf"),
                ),
            )
        else:
            positioned = [item for item in candidates if item[1] is not None]
            selected = (
                min(positioned, key=lambda item: float(item[1]))
                if positioned else None
            )

        count_feature = float(
            np.clip(
                len(interceptor_tracks) / INTERCEPTOR_TRACK_COUNT_NORM,
                0.0,
                1.0,
            )
        )
        if selected is None:
            return count_feature, 0.0, 0.0, 0.0
        _, distance_m, bearing, satellite_source = selected
        self.last_interceptor_satellite_source = satellite_source
        distance_feature = float(
            np.clip(
                float(distance_m or 0.0) / INTERCEPTOR_DISTANCE_NORM_M,
                0.0,
                1.0,
            )
        )
        if bearing is None:
            return count_feature, distance_feature, 0.0, 0.0
        relative_bearing = float(bearing) - float(heading)
        return (
            count_feature,
            distance_feature,
            float(np.sin(relative_bearing)),
            float(np.cos(relative_bearing)),
        )

    def _target_states(
        self,
        observation: Mapping[str, Any],
        current_target_index: int | None,
    ) -> list[tuple[dict[str, float] | None, bool, int]]:
        self_info = observation.get("self") or {}
        step = int(observation.get("step", 0))
        sim_step = max(1.0, float(observation.get("sim_step", 1.0)))
        sim_time = float(observation.get("sim_time", step * sim_step))
        observer_id = int(observation.get("entity_id", -1))
        detect_info = self_info.get("detectInfo") or {}
        detections = {
            int(self._field(info, "entity_id", key)): info
            for key, info in detect_info.items()
        }
        states: list[tuple[dict[str, float] | None, bool, int]] = []
        coordinates = np.zeros((self.target_slots, 2), dtype=np.float32)
        valid = np.zeros(self.target_slots, dtype=np.bool_)
        for index, target in enumerate(self.targets):
            if (
                bool(target.get("slot_empty", False))
                or not bool(target.get("actor_visible", True))
            ):
                states.append((None, False, self.max_steps))
                continue
            target_id = int(target["entity_id"])
            target_type = int(target.get("type", -1))
            detection = detections.get(target_id)
            detected_position = (
                self._detection_position(
                    detection,
                    step=step,
                    sim_time=sim_time,
                    sim_step=sim_step,
                )
                if detection is not None else None
            )
            if detected_position is not None:
                self._local_target_memory[target_id] = detected_position

            private = target_type in self.locally_observed_target_types
            retained = (
                private
                and current_target_index == index
                and target_id in self._local_target_memory
            )
            if private and detected_position is None and not retained:
                # The team catalogue may own this slot, but an actor that did
                # not receive the track sees the same zeros as an empty slot.
                states.append((None, False, self.max_steps))
                continue

            memory = detected_position or self._local_target_memory.get(target_id)
            if private:
                assert memory is not None
                target_position = {"lon": memory[0], "lat": memory[1]}
                detection_age = max(0, step - memory[2])
            elif detected_position is not None:
                target_position = {
                    "lon": detected_position[0],
                    "lat": detected_position[1],
                }
                detection_age = max(0, step - detected_position[2])
            else:
                target_position = dict(target.get("position") or {})
                detection_age = self.max_steps
            if "lon" not in target_position or "lat" not in target_position:
                states.append((None, False, self.max_steps))
                continue
            coordinates[index] = (
                float(target_position["lon"]),
                float(target_position["lat"]),
            )
            valid[index] = True
            direct_detection = bool(
                detection is not None
                and detected_position is not None
                and int(self._field(detection, "detect_from", -1))
                == observer_id
                and detection_age == 0
            )
            states.append((target_position, direct_detection, detection_age))

        self.last_entity_type = int(self_info.get("type", -1))
        self.last_target_coordinates = coordinates
        self.last_target_valid_mask = valid
        return states

    def target_view(
        self,
        observation: Mapping[str, Any],
        current_target_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._target_states(observation, current_target_index)
        return (
            self.last_target_coordinates.copy(),
            self.last_target_valid_mask.copy(),
        )

    def encode(
        self,
        observation: Mapping[str, Any],
        *,
        launched: bool,
        launch_step: int,
        satellite_used: bool,
        maneuver_state: int,
        current_target_index: int | None = None,
        target_switch_elapsed: int = 0,
        task_context: Sequence[float] | None = None,
    ) -> np.ndarray:
        self_info = observation["self"]
        position = self_info["position"]
        lon = float(position["lon"])
        lat = float(position["lat"])
        step = int(observation.get("step", 0))
        entity_type = int(self_info.get("type", 0))
        elapsed = 0 if not launched else max(0, step - launch_step)
        speed, vertical_speed, heading = self._self_motion(
            self_info,
            lon=lon,
            lat=lat,
        )
        satellite_active = bool(
            self_info.get("is_using_satellite", satellite_used)
        )
        agent_slot = np.clip(
            max(0, self.agent_id - 1) / max(self.team_size - 1, 1), 0.0, 1.0
        )
        role_index = max(0, self.agent_id - 1) % 3

        features = [
            np.clip(step / self.max_steps, 0.0, 1.0),
            lon / 180.0,
            lat / 90.0,
            np.clip(float(position.get("alt", 0.0)) / 20000.0, -1.0, 1.0),
            np.clip(float(self_info.get("health", 0.0)) / 100.0, 0.0, 1.0),
            float(entity_type == 21000),
            float(entity_type == 21001),
            float(entity_type == 21002),
            float(launched),
            np.clip(elapsed / self.max_steps, 0.0, 1.0),
            float(satellite_active),
            np.clip(float(maneuver_state) / 2.0, -1.0, 1.0),
            np.clip(float(target_switch_elapsed) / 50.0, 0.0, 1.0),
            np.clip(speed / 1500.0, 0.0, 2.0),
            np.clip(vertical_speed / 500.0, -1.0, 1.0),
            np.sin(heading),
            np.cos(heading),
            agent_slot,
            float(role_index == 0),
            float(role_index == 1),
            float(role_index == 2),
        ]
        if self.include_agent_identity:
            identity = [0.0] * self.IDENTITY_FEATURES
            identity[self.agent_id - 1] = 1.0
            features.extend(identity)

        detect_info = self_info.get("detectInfo") or {}
        target_states = self._target_states(observation, current_target_index)

        for index in range(self.target_slots):
            if index >= len(self.targets):
                features.extend([0.0] * self.target_feature_dim)
                continue
            target = self.targets[index]
            target_position, detected, detection_age = target_states[index]
            if target_position is None:
                features.extend([0.0] * self.target_feature_dim)
                continue
            name = str(target.get("nameChn", ""))
            delta_lon = float(target_position["lon"]) - lon
            delta_lat = float(target_position["lat"]) - lat
            mean_latitude = np.radians((float(target_position["lat"]) + lat) / 2.0)
            east_km = delta_lon * 111.32 * np.cos(mean_latitude)
            north_km = delta_lat * 110.57
            distance_km = float(np.hypot(east_km, north_km))
            bearing = float(np.arctan2(east_km, north_km))
            relative_bearing = bearing - heading
            health_ratio, damage_ratio, alive = self._target_runtime_states.get(
                int(target["entity_id"]),
                (0.0, 0.0, 0.0),
            )
            features.extend([
                np.clip(delta_lon / 10.0, -1.0, 1.0),
                np.clip(delta_lat / 10.0, -1.0, 1.0),
                np.clip(distance_km / 1000.0, 0.0, 1.0),
                np.sin(bearing),
                np.cos(bearing),
                float(detected),
                np.clip(detection_age / self.max_steps, 0.0, 1.0),
                float(name.startswith("目标")),
                float(name.startswith("拦截阵地")),
                float(current_target_index == index),
                np.sin(relative_bearing),
                np.cos(relative_bearing),
            ])
            if self.include_target_runtime_state:
                features.extend((health_ratio, damage_ratio, alive))

        features.extend(
            self._interceptor_threat_summary(
                observation,
                lon=lon,
                lat=lat,
                heading=heading,
            )
        )
        if self.hierarchical_task_context:
            context = tuple(task_context or ())[:self.TASK_FEATURES]
            features.extend(context)
            features.extend([0.0] * (self.TASK_FEATURES - len(context)))
        return np.asarray(features, dtype=np.float32)


class GlobalStateEncoder:
    """将完整态势编码为集中式 Critic 使用的固定长度全局状态。"""

    RED_TYPES = (21000, 21001, 21002)
    BLUE_PREFIXES = ("目标", "拦截阵地", "普通雷达", "标6", "无人船")
    EXTRA_RED_FEATURES = 9

    def __init__(self, max_steps: int = 1000, target_slots: int = DEFAULT_TARGET_SLOTS):
        self.max_steps = max(1, int(max_steps))
        self.target_slots = int(target_slots)

    @property
    def state_dim(self) -> int:
        return (
            1
            + len(self.RED_TYPES) * 5
            + len(self.BLUE_PREFIXES) * 3
            + self.target_slots * 6
            + self.EXTRA_RED_FEATURES
        )

    @staticmethod
    def _mean(items, getter):
        values = [float(getter(item)) for item in items]
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _speed(entity: Mapping[str, Any]) -> float:
        velocity = entity.get("velocity") or {}
        if "speed" in velocity:
            return float(velocity["speed"])
        velocity_ecf = ObservationEncoder._vector3(entity.get("vel_ecf"))
        return (
            float(np.linalg.norm(velocity_ecf))
            if velocity_ecf is not None else 0.0
        )

    @staticmethod
    def _distance_km(first_position, second_position):
        lon1, lat1 = np.radians([
            float(first_position.get("lon", 0.0)),
            float(first_position.get("lat", 0.0)),
        ])
        lon2, lat2 = np.radians([
            float(second_position.get("lon", 0.0)),
            float(second_position.get("lat", 0.0)),
        ])
        delta_lon = lon2 - lon1
        delta_lat = lat2 - lat1
        value = (
            np.sin(delta_lat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
        )
        return float(6371.0 * 2.0 * np.arctan2(np.sqrt(value), np.sqrt(max(0.0, 1.0 - value))))

    def encode(self, observation: Mapping[str, Any]) -> np.ndarray:
        entities_map = observation.get("entities") or {}
        entities = [dict(item, entity_id=int(entity_id)) for entity_id, item in entities_map.items()]
        features = [np.clip(float(observation.get("step", 0)) / self.max_steps, 0.0, 1.0)]

        for entity_type in self.RED_TYPES:
            group = [item for item in entities if int(item.get("type", 0)) == entity_type]
            alive = [item for item in group if float(item.get("health", 0.0)) > 0]
            features.extend([
                len(alive) / max(len(group), 1),
                np.clip(self._mean(group, lambda x: x.get("health", 0.0)) / 100.0, 0.0, 1.0),
                self._mean(alive, lambda x: x.get("position", {}).get("lon", 0.0)) / 180.0,
                self._mean(alive, lambda x: x.get("position", {}).get("lat", 0.0)) / 90.0,
                np.clip(self._mean(alive, lambda x: x.get("position", {}).get("alt", 0.0)) / 20000.0, -1.0, 1.0),
            ])

        for prefix in self.BLUE_PREFIXES:
            group = [item for item in entities if str(item.get("nameChn", "")).startswith(prefix)]
            alive_count = sum(float(item.get("health", 0.0)) > 0 for item in group)
            features.extend([
                np.clip(len(group) / 200.0, 0.0, 1.0),
                alive_count / max(len(group), 1),
                np.clip(self._mean(group, lambda x: x.get("health", 0.0)) / 100.0, 0.0, 1.0),
            ])

        alive_red = [
            item for item in entities
            if int(item.get("type", 0)) in self.RED_TYPES and float(item.get("health", 0.0)) > 0
        ]
        key_targets = [
            item for item in entities
            if str(item.get("nameChn", "")).startswith("目标")
            and float(item.get("health", 0.0)) > 0
        ]
        nearest_distances = [
            min(
                self._distance_km(item.get("position", {}), target.get("position", {}))
                for target in key_targets
            )
            for item in alive_red
        ] if key_targets else []
        speeds = [self._speed(item) for item in alive_red]
        longitudes = [float(item.get("position", {}).get("lon", 0.0)) for item in alive_red]
        latitudes = [float(item.get("position", {}).get("lat", 0.0)) for item in alive_red]
        altitudes = [float(item.get("position", {}).get("alt", 0.0)) for item in alive_red]
        features.extend([
            np.clip(float(np.mean(speeds)) / 1500.0, 0.0, 2.0) if speeds else 0.0,
            np.clip(float(np.std(speeds)) / 1500.0, 0.0, 1.0) if speeds else 0.0,
            np.clip(float(np.std(longitudes)) / 10.0, 0.0, 1.0) if longitudes else 0.0,
            np.clip(float(np.std(latitudes)) / 10.0, 0.0, 1.0) if latitudes else 0.0,
            np.clip(float(np.std(altitudes)) / 20000.0, 0.0, 1.0) if altitudes else 0.0,
            np.clip(min(nearest_distances) / 1000.0, 0.0, 2.0) if nearest_distances else 0.0,
            np.clip(float(np.mean(nearest_distances)) / 1000.0, 0.0, 2.0) if nearest_distances else 0.0,
            np.clip(float(np.std(nearest_distances)) / 1000.0, 0.0, 1.0) if nearest_distances else 0.0,
            sum(speed > 1.0 for speed in speeds) / max(len(alive_red), 1),
        ])

        targets = sorted(
            [item for item in entities if _is_objective(item)],
            key=lambda item: item["entity_id"],
        )
        if len(targets) > self.target_slots:
            raise ValueError(
                f"全局目标数量 {len(targets)} 超过 critic 容量 {self.target_slots}"
            )
        for index in range(self.target_slots):
            if index >= len(targets):
                features.extend([0.0] * 6)
                continue
            target = targets[index]
            position = target.get("position", {})
            name = str(target.get("nameChn", ""))
            features.extend([
                float(position.get("lon", 0.0)) / 180.0,
                float(position.get("lat", 0.0)) / 90.0,
                np.clip(float(position.get("alt", 0.0)) / 20000.0, -1.0, 1.0),
                np.clip(float(target.get("health", 0.0)) / 100.0, 0.0, 1.0),
                float(float(target.get("health", 0.0)) > 0),
                float(name.startswith("目标")),
            ])
        return np.asarray(features, dtype=np.float32)


class LearningActionAdapter:
    """构造动作掩码，并将高层动作翻译成引擎四元动作。"""

    ACTION_SET_ACC_Z = 0
    ACTION_LAUNCH = 1

    def __init__(self, targets: Sequence[Mapping[str, Any]], target_slots: int = DEFAULT_TARGET_SLOTS):
        self.targets = sorted(
            (dict(target) for target in targets if _is_objective(target)),
            key=lambda item: int(item.get("entity_id", -1)),
        )[:target_slots]
        self.target_slots = target_slots

    def build_action_mask(
        self, *, launched: bool, satellite_used: bool, target_switch_allowed: bool = True
    ) -> np.ndarray:
        del launched, satellite_used, target_switch_allowed
        return np.ones(ACTION_DIM, dtype=np.bool_)

    def to_engine_action(self, action: int, entity_id: int) -> np.ndarray:
        action = int(action)
        if action == HighLevelAction.MANEUVER_LEFT:
            return np.asarray([[self.ACTION_SET_ACC_Z, entity_id, -1.0, 0.0]], dtype=np.float64)
        if action == HighLevelAction.MANEUVER_RIGHT:
            return np.asarray([[self.ACTION_SET_ACC_Z, entity_id, 1.0, 0.0]], dtype=np.float64)
        if action == HighLevelAction.STOP_MANEUVER:
            return np.asarray([[self.ACTION_SET_ACC_Z, entity_id, 0.0, 0.0]], dtype=np.float64)
        raise ValueError(f"未知的高层动作: {action}")

    def launch_target(self, target_index: int, entity_id: int) -> np.ndarray:
        """Build a launch command for the target assigned by the team rule."""
        return self._target_action(self.ACTION_LAUNCH, int(target_index), entity_id)

    def _target_action(self, action_type: int, target_index: int, entity_id: int) -> np.ndarray:
        if target_index >= len(self.targets):
            return np.empty((0, 4), dtype=np.float64)
        position = self.targets[target_index]["position"]
        return np.asarray([[
            action_type,
            entity_id,
            float(position["lon"]),
            float(position["lat"]),
        ]], dtype=np.float64)
