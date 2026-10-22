"""PlanSpec lower-motion adapter for the rule-based H-missile Skill."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
from mission_replan_skill import (
    MissionReplanConfig,
    MissionReplanSkill,
    MissionTargetBelief,
    ResponsePackagePlan,
)
from m_strike_package_skill import (
    MMemberSnapshot,
    MStrikeMissionContext,
    MStrikePackageSkill,
)
from l_response_package_skill import (
    LResponsePackageSkill,
    ResponseMemberSnapshot,
)
from l_search_coordination_skill import (
    LSearchCoordinationSkill,
    SearchMemberSnapshot,
)
from target_damage_belief_skill import (
    DeliveryEvidence,
    TargetDamageBeliefSkill,
    TargetHealthBelief,
    TargetPriorEvidence,
)
from three_missile_joint_skill import (
    HStrikePackageSkill,
    JointMemberSnapshot,
    JointMissionContext,
    JointThreat,
    ThreeMissileJointConfig,
    Vector2 as JointVector2,
)

from .controller import (
    H_MISSILE_TYPE,
    HMissileSnapshot,
    HSingleManeuverConfig,
    HSingleManeuverDecision,
    HSingleManeuverSkill,
    ThreatTrack,
    Vector2,
)


INTERCEPTOR_TYPE = 24000
EARTH_EAST_METRES_PER_DEGREE = 111_320.0
EARTH_NORTH_METRES_PER_DEGREE = 110_574.0


@dataclass(frozen=True)
class PlanMotionConfig:
    device: str = "cpu"
    max_steps: int = 3000


@dataclass(frozen=True)
class EncodedManeuver:
    action: int
    entity_id: int
    step: int
    decision: HSingleManeuverDecision | None
    reason: str


@dataclass(frozen=True)
class DispatchedObservation:
    delegate: Any | None
    payload: Any
    agent_id: int
    forced_action: int | None = None


@dataclass(frozen=True)
class JointGroup:
    group_id: str
    entity_type: int
    target_type: int
    target_id: int
    entity_ids: tuple[int, ...]


@dataclass(frozen=True)
class LSearchGroup:
    group_id: str
    cells: tuple[int, ...]
    entity_ids: tuple[int, ...]


@dataclass(frozen=True)
class LResponseGroup:
    group_id: str
    response_group_index: int
    entity_ids: tuple[int, ...]


@dataclass(frozen=True)
class MStrikeGroup:
    group_id: str
    target_type: int
    target_id: int
    planned_release_steps: tuple[int, ...]
    entity_ids: tuple[int, ...]


def default_profile_path() -> Path:
    return Path(__file__).resolve().parent / "profiles/default.json"


def _field(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _xyz(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    try:
        return (
            float(_field(value, "x")),
            float(_field(value, "y")),
            float(_field(value, "z")),
        )
    except (TypeError, ValueError):
        return None


def _lla(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    try:
        return (
            float(_field(value, "lon", "longitude", "x")),
            float(_field(value, "lat", "latitude", "y")),
            float(_field(value, "alt", "altitude", "z", default=0.0)),
        )
    except (TypeError, ValueError):
        return None


def _local_basis(lon_degrees: float, lat_degrees: float) -> tuple[tuple[float, ...], ...]:
    lon = math.radians(lon_degrees)
    lat = math.radians(lat_degrees)
    east = (-math.sin(lon), math.cos(lon), 0.0)
    north = (-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat))
    return east, north


def _project(vector: tuple[float, float, float], basis: tuple[tuple[float, ...], ...]) -> Vector2:
    east, north = basis
    return Vector2(
        sum(vector[index] * east[index] for index in range(3)),
        sum(vector[index] * north[index] for index in range(3)),
    )


def _subtract(
    first: tuple[float, float, float], second: tuple[float, float, float]
) -> tuple[float, float, float]:
    return tuple(first[index] - second[index] for index in range(3))  # type: ignore[return-value]


def _goal_vector(
    own_lon: float, own_lat: float, goal_lon: float, goal_lat: float
) -> Vector2:
    delta_lon = (goal_lon - own_lon + 180.0) % 360.0 - 180.0
    return Vector2(
        delta_lon * EARTH_EAST_METRES_PER_DEGREE * math.cos(math.radians(own_lat)),
        (goal_lat - own_lat) * EARTH_NORTH_METRES_PER_DEGREE,
    )


def _track_values(container: Any) -> Iterable[Any]:
    if container is None:
        return ()
    if isinstance(container, Mapping):
        return container.values()
    if isinstance(container, (list, tuple)):
        return container
    return (container,)


class HSingleManeuverRuntimePolicy:
    """Adapter consumed by ``UnifiedAttackMissileAgent._evasion_motion``."""

    algorithm_name = "h_single_maneuver_rule_v1"
    gate_mode = "cpa_rule_v1"
    is_evasion_skill = True
    requires_target_context = True
    requires_environment_hooks = False
    records_evaluation = False
    motion_learning_enabled = False
    instances: list["HSingleManeuverRuntimePolicy"] = []

    def __init__(self, config: PlanMotionConfig | Any | None = None, *, max_steps: int | None = None) -> None:
        self.max_steps = int(max_steps or getattr(config, "max_steps", 3000))
        self.rule_config = HSingleManeuverConfig()
        self.controllers: dict[int, HSingleManeuverSkill] = {}
        self._own_history: dict[int, tuple[int, tuple[float, float, float]]] = {}
        self._threat_history: dict[tuple[int, int], tuple[int, tuple[float, float, float]]] = {}
        self._trace_handle = None
        self.training = False
        self.instances.append(self)

    def load(self, profile: str | Path) -> None:
        path = Path(profile)
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("H maneuver profile must be a JSON object")
        self.rule_config = HSingleManeuverConfig.from_mapping(values)
        self.reset_episode()

    def set_training(self, training: bool) -> None:
        if training:
            raise ValueError("The H single-maneuver rule is evaluation-only")
        self.training = False

    def reset_episode(self) -> None:
        for controller in self.controllers.values():
            controller.reset()
        self.controllers.clear()
        self._own_history.clear()
        self._threat_history.clear()

    def encode_motion_observation(
        self,
        observation: Mapping[str, Any],
        *,
        agent_id: int,
        entity_id: int,
        launch_step: int,
        maneuver_state: int,
        goal_position: Mapping[str, Any],
    ) -> EncodedManeuver:
        del agent_id, launch_step, maneuver_state
        step = int(observation.get("step", 0))
        own = observation.get("self") or {}
        entity_type = int(_field(own, "type", "entity_type", "entityType", default=-1))
        if entity_type != H_MISSILE_TYPE:
            return EncodedManeuver(1, entity_id, step, None, "non-H missile")

        position = _lla(_field(own, "position", "lla"))
        own_ecf = _xyz(_field(own, "pos_ecf", "posEcf"))
        goal_lon = _field(goal_position, "lon", "longitude")
        goal_lat = _field(goal_position, "lat", "latitude")
        if position is None or own_ecf is None or goal_lon is None or goal_lat is None:
            return EncodedManeuver(1, entity_id, step, None, "incomplete own/goal kinematics")

        own_lon, own_lat, _own_alt = position
        basis = _local_basis(own_lon, own_lat)
        goal = _goal_vector(own_lon, own_lat, float(goal_lon), float(goal_lat))
        own_velocity = self._own_velocity(entity_id, step, own_ecf, goal, basis)
        threats = self._threats(entity_id, step, own_ecf, basis, own)
        snapshot = HMissileSnapshot(
            step=step,
            missile_id=entity_id,
            position=Vector2(0.0, 0.0),
            velocity=own_velocity,
            goal_position=goal,
            threats=threats,
        )
        controller = self.controllers.setdefault(
            entity_id, HSingleManeuverSkill(self.rule_config)
        )
        decision = controller.step(snapshot)
        self._write_trace(
            entity_id,
            step,
            decision,
            len(threats),
            snapshot,
            own_lon,
            own_lat,
            float(goal_lon),
            float(goal_lat),
        )
        return EncodedManeuver(
            int(decision.action), entity_id, step, decision, decision.reason
        )

    def prepare_action_mask(
        self, observation: EncodedManeuver, action_mask: np.ndarray
    ) -> np.ndarray:
        supplied = np.asarray(action_mask, dtype=np.bool_)
        if supplied.shape != (3,):
            raise ValueError(f"Expected three maneuver actions, got {supplied.shape}")
        action = observation.action if bool(supplied[observation.action]) else 1
        effective = np.zeros(3, dtype=np.bool_)
        effective[action] = True
        return effective

    def select_action(
        self, observation: EncodedManeuver, action_mask: np.ndarray
    ) -> int:
        if not bool(np.asarray(action_mask, dtype=np.bool_)[observation.action]):
            return 1
        return int(observation.action)

    def close(self) -> None:
        if self._trace_handle is not None:
            self._trace_handle.close()
            self._trace_handle = None

    def _own_velocity(
        self,
        entity_id: int,
        step: int,
        current: tuple[float, float, float],
        goal: Vector2,
        basis: tuple[tuple[float, ...], ...],
    ) -> Vector2:
        previous = self._own_history.get(entity_id)
        self._own_history[entity_id] = (step, current)
        if previous is not None and step > previous[0]:
            elapsed = float(step - previous[0])
            delta = _subtract(current, previous[1])
            return _project(tuple(value / elapsed for value in delta), basis)
        norm = math.hypot(goal.east, goal.north)
        if norm <= 1e-9:
            return Vector2(0.0, 0.0)
        return Vector2(2000.0 * goal.east / norm, 2000.0 * goal.north / norm)

    def _threats(
        self,
        entity_id: int,
        step: int,
        own_ecf: tuple[float, float, float],
        basis: tuple[tuple[float, ...], ...],
        own: Mapping[str, Any] | Any,
    ) -> tuple[ThreatTrack, ...]:
        selected: dict[int, ThreatTrack] = {}
        containers = (
            _field(own, "detectInfo", "detect_info", default={}),
            _field(own, "commRangeInfo", "comm_range_info", default={}),
        )
        for container in containers:
            for raw in _track_values(container):
                try:
                    track_type = int(_field(raw, "entity_type", "entityType", "type", default=-1))
                    threat_id = int(_field(raw, "entity_id", "entityId", "id"))
                except (TypeError, ValueError):
                    continue
                if track_type != INTERCEPTOR_TYPE or threat_id in selected:
                    continue
                position = _xyz(_field(raw, "pos_ecf", "posEcf"))
                velocity_ecf = _xyz(_field(raw, "vel_ecf", "velEcf", "velocity"))
                if position is None:
                    continue
                relative = _project(_subtract(position, own_ecf), basis)
                velocity = self._track_velocity(
                    entity_id, threat_id, step, position, velocity_ecf, basis
                )
                if math.hypot(velocity.east, velocity.north) < 25.0:
                    if math.hypot(relative.east, relative.north) > self.rule_config.emergency_range_metres:
                        continue
                selected[threat_id] = ThreatTrack(threat_id, relative, velocity)
        return tuple(selected[key] for key in sorted(selected))

    def _track_velocity(
        self,
        entity_id: int,
        threat_id: int,
        step: int,
        position: tuple[float, float, float],
        supplied: tuple[float, float, float] | None,
        basis: tuple[tuple[float, ...], ...],
    ) -> Vector2:
        key = (entity_id, threat_id)
        previous = self._threat_history.get(key)
        self._threat_history[key] = (step, position)
        if supplied is not None and math.sqrt(sum(value * value for value in supplied)) >= 25.0:
            return _project(supplied, basis)
        if previous is not None and step > previous[0]:
            elapsed = float(step - previous[0])
            delta = _subtract(position, previous[1])
            return _project(tuple(value / elapsed for value in delta), basis)
        return Vector2(0.0, 0.0)

    def _write_trace(
        self,
        entity_id: int,
        step: int,
        decision: HSingleManeuverDecision,
        visible_threats: int,
        snapshot: HMissileSnapshot,
        own_lon: float,
        own_lat: float,
        goal_lon: float,
        goal_lat: float,
    ) -> None:
        path_value = os.getenv("H_SINGLE_MANEUVER_TRACE")
        if not path_value:
            return
        if self._trace_handle is None:
            path = Path(path_value)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._trace_handle = path.open("a", encoding="utf-8")
        threat = decision.threat
        record = {
            "step": step,
            "entity_id": entity_id,
            "action": int(decision.action),
            "engine_acc_z": decision.engine_acc_z,
            "phase": decision.phase.value,
            "reason": decision.reason,
            "goal_error_degrees": decision.goal_error_degrees,
            "goal_distance_metres": math.hypot(
                snapshot.goal_position.east, snapshot.goal_position.north
            ),
            "own_speed_mps": math.hypot(
                snapshot.velocity.east, snapshot.velocity.north
            ),
            "own_lon": own_lon,
            "own_lat": own_lat,
            "goal_lon": goal_lon,
            "goal_lat": goal_lat,
            "visible_threats": visible_threats,
            "threat_id": threat.threat_id if threat else None,
            "threat_range_metres": threat.range_metres if threat else None,
            "time_to_cpa_seconds": threat.time_to_cpa_seconds if threat else None,
            "cpa_distance_metres": threat.cpa_distance_metres if threat else None,
            "closing_speed_mps": threat.closing_speed_mps if threat else None,
            "emergency": threat.emergency if threat else False,
        }
        self._trace_handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._trace_handle.flush()


class MissileTypeDispatchRuntimePolicy:
    """Route H independently while retaining the original local policy for M/L."""

    is_evasion_skill = True
    requires_target_context = True
    requires_environment_hooks = True
    allows_satellite = True
    records_evaluation = False
    motion_learning_enabled = False
    gate_mode = "mission_v12"
    instances: list["MissileTypeDispatchRuntimePolicy"] = []

    def __init__(
        self,
        *,
        mode: str,
        max_steps: int,
        h_profile: str | Path,
        local_factory: Callable[..., Any],
        local_pulse_steps: int = 10,
        joint_rule_profile: str = "safe",
        joint_release_window_steps: int = 10,
        joint_max_members: int = 6,
        mission_replan_mode: str = "off",
    ) -> None:
        if mode not in {
            "hybrid",
            "local-all",
            "joint-h-v13",
            "joint-l-local",
            "joint-hl-local",
            "joint-m-local",
            "joint-hml-local",
        }:
            raise ValueError("unsupported local-policy dispatch mode")
        if type(local_pulse_steps) is not int or local_pulse_steps < 1:
            raise ValueError("local_pulse_steps must be a positive integer")
        self.mode = mode
        self.max_steps = int(max_steps)
        self.h_profile = Path(h_profile)
        self.h_policy = HSingleManeuverRuntimePolicy(max_steps=self.max_steps)
        self.local_policy = local_factory(max_steps=self.max_steps)
        self.local_pulse_steps = local_pulse_steps
        if joint_rule_profile not in {"safe", "experimental", "mission-aware"}:
            raise ValueError(
                "joint_rule_profile must be safe, experimental, or mission-aware"
            )
        self.joint_rule_profile = joint_rule_profile
        if joint_release_window_steps < 0:
            raise ValueError("joint release window must be non-negative")
        if not 2 <= joint_max_members <= 6:
            raise ValueError("joint max members must be between 2 and 6")
        self.joint_release_window_steps = int(joint_release_window_steps)
        self.joint_max_members = int(joint_max_members)
        if mission_replan_mode not in {"off", "l-response", "all-packages"}:
            raise ValueError(
                "mission_replan_mode must be off, l-response, or all-packages"
            )
        self.mission_replan_mode = mission_replan_mode
        self._local_pulses: dict[int, tuple[int, int]] = {}
        self._joint_groups: tuple[JointGroup, ...] = ()
        self._joint_controllers: dict[str, HStrikePackageSkill] = {}
        self._prepared_step: int | None = None
        self._prepared_actions: dict[int, int] = {}
        self._l_search_groups: tuple[LSearchGroup, ...] = ()
        self._l_response_groups: tuple[LResponseGroup, ...] = ()
        self._l_search_controllers: dict[str, LSearchCoordinationSkill] = {}
        self._l_response_controllers: dict[str, LResponsePackageSkill] = {}
        self._mission_replan: MissionReplanSkill | None = None
        self._mission_released_entities: set[int] = set()
        self._mission_cancelled_entities: set[int] = set()
        self._mission_m_stopped_entities: set[int] = set()
        self._m_groups: tuple[MStrikeGroup, ...] = ()
        self._m_controllers: dict[str, MStrikePackageSkill] = {}
        self._damage_belief = TargetDamageBeliefSkill()
        self._belief_registered_targets: set[int] = set()
        self._mission_retarget_by_entity: dict[int, int] = {}
        self._mission_retargeted_packages: set[str] = set()
        self._mission_target_last_assignment_step: dict[int, int] = {}
        self._mission_retarget_history: list[tuple[int, int]] = []
        self.training = False
        self.algorithm_name = (
            "hybrid_h_rule_ml_local_v13"
            if mode == "hybrid"
            else (
                "joint_h_three_missile_v13"
                if mode == "joint-h-v13"
                else (
                    "joint_l_local"
                    if mode == "joint-l-local"
                    else (
                        "joint_hl_local"
                        if mode == "joint-hl-local"
                        else (
                            "joint_m_local"
                            if mode == "joint-m-local"
                            else (
                                "joint_hml_local"
                                if mode == "joint-hml-local"
                                else "local_v13_all"
                            )
                        )
                    )
                )
            )
        )
        self.instances.append(self)

    def load(self, checkpoint: str | Path) -> None:
        self.h_policy.load(self.h_profile)
        self.local_policy.load(str(checkpoint))
        self.reset_episode()

    def encode_motion_observation(
        self, observation: Mapping[str, Any], **context: Any
    ) -> DispatchedObservation:
        agent_id = int(context["agent_id"])
        own = observation.get("self") or {}
        try:
            entity_type = int(
                _field(own, "type", "entity_type", "entityType", default=-1)
            )
        except (TypeError, ValueError):
            entity_type = -1
        if self.mode == "joint-h-v13" and self._prepared_step == int(
            observation.get("step", -1)
        ):
            prepared = self._prepared_actions.get(int(context["entity_id"]))
            if prepared is not None:
                return DispatchedObservation(
                    None, None, agent_id, forced_action=int(prepared)
                )
        delegate: Any | None
        if entity_type == H_MISSILE_TYPE and self.mode == "hybrid":
            delegate = self.h_policy
        elif entity_type in {21000, 21001, 21002}:
            delegate = self.local_policy
        else:
            return DispatchedObservation(None, None, agent_id, forced_action=1)

        if delegate is self.local_policy:
            pulse = self._local_pulses.get(agent_id)
            if pulse is not None:
                action, remaining = pulse
                if remaining > 0:
                    self._local_pulses[agent_id] = (action, remaining - 1)
                    return DispatchedObservation(
                        delegate, None, agent_id, forced_action=action
                    )
                self._local_pulses.pop(agent_id, None)
                return DispatchedObservation(delegate, None, agent_id, forced_action=1)

        payload = delegate.encode_motion_observation(observation, **context)
        return DispatchedObservation(delegate, payload, agent_id)

    def configure_plan_bindings(
        self, bindings: Any, entity_types: Mapping[int, int]
    ) -> None:
        if self.mode in {"joint-h-v13", "joint-hl-local", "joint-hml-local"} and not self._joint_groups:
            self._configure_h_bindings(bindings, entity_types)
        if self.mode in {"joint-l-local", "joint-hl-local", "joint-hml-local"} and not (
            self._l_search_groups or self._l_response_groups
        ):
            self._configure_l_bindings(bindings, entity_types)
        if self.mode in {"joint-m-local", "joint-hml-local"} and not self._m_groups:
            self._configure_m_bindings(bindings, entity_types)

    def _configure_h_bindings(
        self, bindings: Any, entity_types: Mapping[int, int]
    ) -> None:
        buckets: dict[tuple[str, int, int, int], list[tuple[int, int]]] = {}
        for binding in bindings:
            role = getattr(binding.role, "value", str(binding.role))
            goal = binding.goal
            goal_kind = getattr(getattr(goal, "kind", None), "value", "")
            entity_id = int(binding.entity_id)
            entity_type = int(entity_types.get(entity_id, -1))
            if (
                role != "H"
                or entity_type != H_MISSILE_TYPE
                or goal_kind != "public_target"
                or goal.target_id is None
                or goal.target_type is None
            ):
                continue
            key = (role, entity_type, int(goal.target_type), int(goal.target_id))
            release_step = int(getattr(binding, "planned_release_step", 0) or 0)
            buckets.setdefault(key, []).append((release_step, entity_id))

        groups: list[JointGroup] = []
        for (role, entity_type, target_type, target_id), members in sorted(
            buckets.items()
        ):
            ordered = sorted(members)
            cohorts: list[list[tuple[int, int]]] = []
            cohort: list[tuple[int, int]] = []
            for release_step, entity_id in ordered:
                if (
                    cohort
                    and (
                        release_step - cohort[0][0] > self.joint_release_window_steps
                        or len(cohort) >= self.joint_max_members
                    )
                ):
                    cohorts.append(cohort)
                    cohort = []
                cohort.append((release_step, entity_id))
            if cohort:
                cohorts.append(cohort)
            for cohort in cohorts:
                if len(cohort) < 2:
                    continue
                # The pure package skill returns actions in entity-id order.
                # Freeze the runtime group in the same order before zipping
                # final actions back to simulator entities.
                entity_ids = tuple(sorted(entity_id for _step, entity_id in cohort))
                release_min = min(step for step, _entity in cohort)
                release_max = max(step for step, _entity in cohort)
                group_id = (
                    f"{role}:{entity_type}:{target_type}:{target_id}:"
                    f"r{release_min}-{release_max}:"
                    + "-".join(str(value) for value in entity_ids)
                )
                groups.append(
                    JointGroup(
                        group_id=group_id,
                        entity_type=entity_type,
                        target_type=target_type,
                        target_id=target_id,
                        entity_ids=entity_ids,
                    )
                )
                if self.joint_rule_profile == "experimental":
                    config = ThreeMissileJointConfig.experimental()
                elif self.joint_rule_profile == "mission-aware":
                    config = ThreeMissileJointConfig.mission_aware()
                else:
                    config = ThreeMissileJointConfig()
                self._joint_controllers[group_id] = HStrikePackageSkill(config)
        owned = [entity for group in groups for entity in group.entity_ids]
        if len(owned) != len(set(owned)):
            raise ValueError("PlanSpec joint groups contain duplicate entities")
        self._joint_groups = tuple(groups)

    def _configure_l_bindings(
        self, bindings: Any, entity_types: Mapping[int, int]
    ) -> None:
        search_by_cell: dict[int, list[int]] = {}
        response_by_index: dict[int, list[int]] = {}
        for binding in bindings:
            entity_id = int(binding.entity_id)
            if int(entity_types.get(entity_id, -1)) != 21002:
                continue
            role = getattr(binding.role, "value", str(binding.role))
            if role != "L":
                continue
            goal_kind = getattr(
                getattr(getattr(binding, "goal", None), "kind", None),
                "value",
                "",
            )
            if (
                goal_kind == "hidden_search"
                and not bool(getattr(binding, "reserve_response", False))
                and getattr(binding, "search_cell", None) is not None
            ):
                search_by_cell.setdefault(int(binding.search_cell), []).append(entity_id)
            elif (
                bool(getattr(binding, "reserve_response", False))
                and getattr(binding, "response_group_index", None) is not None
            ):
                response_by_index.setdefault(
                    int(binding.response_group_index), []
                ).append(entity_id)

        if set(search_by_cell) == set(range(32)):
            search_groups = []
            max_rank = max(len(values) for values in search_by_cell.values())
            for rank in range(max_rank):
                layer = [
                    (cell, sorted(search_by_cell[cell])[rank])
                    for cell in range(32)
                    if rank < len(search_by_cell[cell])
                ]
                for start in range(0, len(layer), 4):
                    chunk = layer[start : start + 4]
                    if len(chunk) < 2:
                        continue
                    cells = tuple(cell for cell, _entity in chunk)
                    entity_ids = tuple(entity for _cell, entity in chunk)
                    group_id = (
                        f"L-search:r{rank}:c{cells[0]}-{cells[-1]}:"
                        + "-".join(str(value) for value in entity_ids)
                    )
                    search_groups.append(LSearchGroup(group_id, cells, entity_ids))
                    self._l_search_controllers[group_id] = LSearchCoordinationSkill()
            self._l_search_groups = tuple(search_groups)

        response_groups = []
        hard_scale = sum(len(values) for values in search_by_cell.values()) > 32
        nominal_size = 4 if hard_scale else 3
        for response_index, members in sorted(response_by_index.items()):
            remaining = sorted(members)
            sizes = (
                self._partition_response_members(len(remaining), nominal_size)
                if hard_scale
                else self._partition_easy_response_members(len(remaining))
            )
            offset = 0
            for package_index, size in enumerate(sizes):
                entity_ids = tuple(remaining[offset : offset + size])
                offset += size
                group_id = (
                    f"L-response:g{response_index}:p{package_index}:"
                    + "-".join(str(value) for value in entity_ids)
                )
                response_groups.append(
                    LResponseGroup(group_id, response_index, entity_ids)
                )
                self._l_response_controllers[group_id] = LResponsePackageSkill()
        self._l_response_groups = tuple(response_groups)

    def _configure_m_bindings(
        self, bindings: Any, entity_types: Mapping[int, int]
    ) -> None:
        buckets: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for binding in bindings:
            entity_id = int(binding.entity_id)
            if int(entity_types.get(entity_id, -1)) != 21001:
                continue
            role = getattr(binding.role, "value", str(binding.role))
            goal = getattr(binding, "goal", None)
            goal_kind = getattr(getattr(goal, "kind", None), "value", "")
            if (
                role != "M"
                or goal_kind != "public_target"
                or goal.target_id is None
                or int(goal.target_type) not in (9400, 9600)
            ):
                continue
            key = (int(goal.target_type), int(goal.target_id))
            buckets.setdefault(key, []).append(
                (int(getattr(binding, "planned_release_step", 0) or 0), entity_id)
            )
        groups = []
        for (target_type, target_id), values in sorted(buckets.items()):
            ordered = sorted(values)
            cohorts: list[list[tuple[int, int]]] = []
            cohort: list[tuple[int, int]] = []
            for release_step, entity_id in ordered:
                if cohort and release_step - cohort[0][0] > 10:
                    cohorts.append(cohort)
                    cohort = []
                cohort.append((release_step, entity_id))
            if cohort:
                cohorts.append(cohort)
            for cohort_index, cohort in enumerate(cohorts):
                for package_index, package in enumerate(
                    self._partition_m_cohort(cohort)
                ):
                    release_steps = tuple(step for step, _entity in package)
                    entity_ids = tuple(sorted(entity for _step, entity in package))
                    group_id = (
                        f"M:{target_type}:{target_id}:c{cohort_index}:p{package_index}:"
                        + "-".join(str(value) for value in entity_ids)
                    )
                    groups.append(
                        MStrikeGroup(
                            group_id,
                            target_type,
                            target_id,
                            release_steps,
                            entity_ids,
                        )
                    )
                    self._m_controllers[group_id] = MStrikePackageSkill()
        self._m_groups = tuple(groups)

    @staticmethod
    def _partition_m_cohort(
        cohort: list[tuple[int, int]],
    ) -> tuple[tuple[tuple[int, int], ...], ...]:
        remaining = list(cohort)
        packages = []
        while len(remaining) >= 2:
            size = min(6, len(remaining))
            if len(remaining) - size == 1:
                size -= 1
            packages.append(tuple(remaining[:size]))
            del remaining[:size]
        return tuple(packages)

    @staticmethod
    def _partition_response_members(count: int, nominal_size: int) -> tuple[int, ...]:
        if count < 2:
            return ()
        sizes = []
        remaining = int(count)
        while remaining > 0:
            size = min(nominal_size, remaining)
            if remaining - size == 1:
                size -= 1
            sizes.append(size)
            remaining -= size
        return tuple(size for size in sizes if 2 <= size <= 4)

    @staticmethod
    def _partition_easy_response_members(count: int) -> tuple[int, ...]:
        if count < 2:
            return ()
        sizes = [4]
        remaining = count - 4
        while remaining > 0:
            size = min(3, remaining)
            if remaining - size == 1:
                size -= 1
            if size < 2:
                sizes[-1] += remaining
                break
            sizes.append(size)
            remaining -= size
        return tuple(sizes)

    def prepare_decision_step(
        self,
        *,
        step: int,
        decision_order: Any,
        agent_observations: Mapping[int, Mapping[str, Any]],
        agents_by_id: Mapping[int, Any],
        bindings: Any,
        entity_types: Mapping[int, int],
        mission_context_by_group: Mapping[str, JointMissionContext] | None = None,
        plan_policy: Any | None = None,
    ) -> None:
        del decision_order
        if self.mode not in {
            "joint-h-v13",
            "joint-l-local",
            "joint-hl-local",
            "joint-m-local",
            "joint-hml-local",
        }:
            return
        self.configure_plan_bindings(bindings, entity_types)
        self._prepared_step = int(step)
        self._prepared_actions.clear()
        if self.mission_replan_mode in {"l-response", "all-packages"}:
            self._prepare_mission_replan(int(step), plan_policy)
        agents_by_entity = {
            int(agent.entity_id): agent
            for agent in agents_by_id.values()
            if hasattr(agent, "entity_id")
        }
        for group in self._joint_groups:
            agents = [agents_by_entity.get(entity_id) for entity_id in group.entity_ids]
            if any(agent is None for agent in agents):
                continue
            observations = [
                agent_observations.get(int(agent.agent_id)) for agent in agents
            ]
            if any(observation is None for observation in observations):
                continue
            if any(
                int(observation.get("step", -1)) != int(step)
                for observation in observations
            ):
                continue
            if any(
                int(agent.launch_step) < 0
                or int(getattr(agent, "_last_stage", 0)) >= 5
                or int(getattr(agent, "current_goal_id", -1)) != group.target_id
                for agent in agents
            ):
                continue

            baselines: list[int] = []
            reason = "joint_error_v13_fallback"
            metrics: dict[str, Any] = {}
            try:
                shadow_snapshots = self._joint_snapshots(
                    int(step),
                    group,
                    agents,
                    observations,
                    tuple(1 for _entity in group.entity_ids),
                )
                controller = self._joint_controllers[group.group_id]
                linear_separation = self._linear_minimum_separation(
                    shadow_snapshots,
                    horizon_seconds=controller.config.peer_horizon_seconds,
                )
                if linear_separation >= controller.config.peer_trigger_metres:
                    self._write_joint_trace(
                        int(step),
                        group,
                        (),
                        (),
                        "shadow_peer_safe",
                        {
                            "linear_min_separation_metres": linear_separation,
                            "eta_spread_seconds": self._joint_eta_spread(
                                shadow_snapshots
                            ),
                            "eta_actuation_enabled": False,
                        },
                    )
                    continue
                for agent, observation in zip(agents, observations, strict=True):
                    goal = agent._current_goal_position()
                    if goal is None:
                        raise ValueError("joint member has no current goal")
                    context = {
                        "agent_id": int(agent.agent_id),
                        "entity_id": int(agent.entity_id),
                        "launch_step": int(agent.launch_step),
                        "maneuver_state": int(agent.maneuver_state),
                        "goal_position": {
                            "lon": float(goal[0]),
                            "lat": float(goal[1]),
                        },
                    }
                    baselines.append(
                        self._evaluate_local_action(observation, context)
                    )
                snapshots = [
                    replace(snapshot, baseline_action=int(action))
                    for snapshot, action in zip(
                        shadow_snapshots, baselines, strict=True
                    )
                ]
                decision = self._joint_controllers[group.group_id].decide(
                    int(step),
                    tuple(snapshots),
                    None if mission_context_by_group is None else (
                        mission_context_by_group.get(group.group_id)
                    ),
                )
                final_actions = tuple(int(action) for action in decision.actions)
                reason = decision.reason
                metrics = {
                    "threat_owners": [list(item) for item in decision.threat_owners],
                    "baseline_min_separation_metres": (
                        decision.baseline_min_separation_metres
                    ),
                    "selected_min_separation_metres": (
                        decision.selected_min_separation_metres
                    ),
                    "eta_spread_seconds": decision.eta_spread_seconds,
                    "overridden_entity_id": decision.overridden_entity_id,
                }
            except Exception:
                final_actions = tuple(baselines)

            if len(final_actions) != len(group.entity_ids) or any(
                action not in (0, 1, 2) for action in final_actions
            ):
                final_actions = tuple(baselines)
                reason = "invalid_joint_output_v13_fallback"
            if len(final_actions) != len(group.entity_ids):
                continue
            self._prepared_actions.update(
                zip(group.entity_ids, final_actions, strict=True)
            )
            self._write_joint_trace(
                int(step), group, tuple(baselines), final_actions, reason, metrics
            )
        if self.mode in {"joint-l-local", "joint-hl-local", "joint-hml-local"}:
            self._prepare_l_coordination(
                step=int(step),
                agent_observations=agent_observations,
                agents_by_id=agents_by_id,
                plan_policy=plan_policy,
            )
        if self.mode in {"joint-m-local", "joint-hml-local"}:
            self._prepare_m_packages(
                step=int(step),
                agent_observations=agent_observations,
                agents_by_id=agents_by_id,
                plan_policy=plan_policy,
            )
        if self.mission_replan_mode == "all-packages":
            self._prepare_package_retargets(
                step=int(step),
                agents_by_id=agents_by_id,
                plan_policy=plan_policy,
            )

    def _prepare_mission_replan(self, step: int, plan_policy: Any | None) -> None:
        if plan_policy is None or not self._l_response_groups:
            return
        info_center = getattr(plan_policy, "info_center", None)
        spec = getattr(plan_policy, "spec", None)
        hidden_response = getattr(spec, "hidden_response", None)
        if info_center is None or hidden_response is None:
            return
        self._sync_belief_targets(info_center, step)
        if self._mission_replan is None:
            package_plans = []
            by_index: dict[int, list[LResponseGroup]] = {}
            for group in self._l_response_groups:
                by_index.setdefault(group.response_group_index, []).append(group)
            for response_index, groups in sorted(by_index.items()):
                for sequence, group in enumerate(sorted(groups, key=lambda item: item.group_id)):
                    package_plans.append(
                        ResponsePackagePlan(
                            package_id=group.group_id,
                            response_group_index=response_index,
                            sequence_index=sequence,
                            member_ids=group.entity_ids,
                        )
                    )
            hard_scale = len(by_index) > 4
            timeout = 800 if hard_scale else 600
            self._mission_replan = MissionReplanSkill(
                MissionReplanConfig(
                    required_discoveries=int(
                        getattr(hidden_response, "required_discoveries", 0)
                    ),
                    release_delay_steps=int(
                        getattr(hidden_response, "release_delay_steps", 20)
                    ),
                    delivery_timeout_steps=timeout,
                    subsequent_release_interval_steps=(
                        300 if hard_scale else 250
                    ),
                    hard_deadline_steps=900,
                    initial_release_fraction=0.50,
                    minimum_initial_members=3,
                ),
                tuple(package_plans),
            )

        unique_events = []
        seen = set()
        for event in info_center.discovery_events:
            target_id = int(event["target_id"])
            if target_id in seen:
                continue
            seen.add(target_id)
            unique_events.append(event)
        if unique_events:
            self._mission_replan.update_discoveries(
                step=max(int(item["step"]) for item in unique_events),
                target_ids=tuple(int(item["target_id"]) for item in unique_events),
            )

        beliefs = []
        for response_index, event in enumerate(unique_events):
            target_id = int(event["target_id"])
            belief = self._belief_for(target_id, info_center, step)
            if belief is None:
                continue
            beliefs.append(
                MissionTargetBelief(
                    response_group_index=response_index,
                    target_id=target_id,
                    expected_hp=belief.expected_hp,
                    projected_expected_hp=belief.projected_expected_hp,
                    confidence=belief.confidence,
                    confirmed_destroyed=belief.confirmed_destroyed,
                    initial_health=belief.initial_health,
                    delivered_expected_damage=(
                        belief.delivered_expected_damage
                    ),
                    confidence_source=belief.confidence_source,
                )
            )
        decision = self._mission_replan.decide(
            step=step,
            beliefs=tuple(beliefs),
        )
        self._mission_released_entities.update(decision.launch_member_ids)
        self._mission_cancelled_entities.update(decision.cancelled_member_ids)
        self._mission_released_entities.difference_update(
            self._mission_cancelled_entities
        )
        self._write_status_trace(
            "MISSION_REPLAN_TRACE",
            {
                "step": step,
                "launch_package_ids": list(decision.launch_package_ids),
                "launch_member_ids": list(decision.launch_member_ids),
                "held_package_ids": list(decision.held_package_ids),
                "cancelled_package_ids": list(decision.cancelled_package_ids),
                "target_ids_by_response_index": [
                    list(item) for item in decision.target_ids_by_response_index
                ],
                "beliefs": [
                    {
                        "response_group_index": item.response_group_index,
                        "target_id": item.target_id,
                        "initial_health": item.initial_health,
                        "expected_hp": item.expected_hp,
                        "projected_expected_hp": item.projected_expected_hp,
                        "delivered_expected_damage": (
                            item.delivered_expected_damage
                        ),
                        "confidence": item.confidence,
                        "confidence_source": item.confidence_source,
                        "confirmed_destroyed": item.confirmed_destroyed,
                    }
                    for item in beliefs
                ],
            },
        )

    def _sync_belief_targets(self, info_center: Any, step: int) -> None:
        for target in getattr(info_center.fusion, "targets", ()):
            target_id = int(target.entity_id)
            if target_id in self._belief_registered_targets:
                continue
            initial_health = float(
                info_center.initial_health.get(
                    target_id,
                    1.2 if int(target.entity_type) == 9500 else 0.0,
                )
            )
            if initial_health <= 0.0:
                continue
            self._damage_belief.register_target(
                TargetPriorEvidence(
                    target_id=target_id,
                    target_type=int(target.entity_type),
                    initial_health=initial_health,
                    discovered_step=int(step),
                    source="legal_info_center_track",
                )
            )
            self._belief_registered_targets.add(target_id)

    def _belief_for(
        self, target_id: int, info_center: Any, step: int
    ) -> TargetHealthBelief | None:
        self._sync_belief_targets(info_center, step)
        try:
            belief = self._damage_belief.estimate(int(target_id))
            if (
                belief.delivered_expected_damage
                >= belief.initial_health * 1.25
                and belief.confidence < 0.95
            ):
                belief = replace(
                    belief,
                    confidence=0.95,
                    confidence_source="heuristic_overkill_margin",
                )
            return belief
        except ValueError:
            return None

    def observe_command_batch(self, records: Any, plan_policy: Any) -> None:
        info_center = plan_policy.info_center
        self._sync_belief_targets(info_center, int(getattr(plan_policy, "_current_step", 0)))
        agents_by_entity = {
            int(agent.entity_id): agent for agent in getattr(plan_policy, "_agents", ())
        }
        for record in records:
            for command in np.asarray(record.commands).reshape(-1, 4):
                action_type = int(command[0])
                if action_type not in (1, 2):
                    continue
                entity_id = int(command[1])
                agent = agents_by_entity.get(entity_id)
                if agent is None or agent.current_goal_id is None:
                    continue
                target_id = int(agent.current_goal_id)
                try:
                    if action_type == 1:
                        self._damage_belief.commit_missile(
                            missile_id=entity_id,
                            missile_type=int(agent.missile_type),
                            target_id=target_id,
                            satellite_expected=False,
                        )
                    else:
                        try:
                            self._damage_belief.transfer_missile(entity_id, target_id)
                        except ValueError:
                            self._damage_belief.commit_missile(
                                missile_id=entity_id,
                                missile_type=int(agent.missile_type),
                                target_id=target_id,
                                satellite_expected=False,
                            )
                except ValueError:
                    continue

    def observe_terminal(self, agent: Any) -> None:
        if agent.death_reason == "delivery" and agent.current_goal_id is not None:
            self._damage_belief.report_delivery(
                DeliveryEvidence(
                    missile_id=int(agent.entity_id),
                    missile_type=int(agent.missile_type),
                    target_id=int(agent.current_goal_id),
                    step=int(getattr(agent.policy, "_current_step", 0)),
                    last_stage=int(getattr(agent, "_last_stage", 0)),
                    minimum_target_distance_metres=(
                        None
                        if getattr(agent, "_min_goal_dist", None) is None
                        else float(agent._min_goal_dist) * 1000.0
                    ),
                    disappeared=True,
                    satellite_active=bool(getattr(agent, "sat_used", False)),
                    event_id=f"terminal:{agent.entity_id}:{getattr(agent.policy, '_current_step', 0)}",
                )
            )
        else:
            self._damage_belief.release_missile(int(agent.entity_id))

    def filter_plan_decision(
        self,
        *,
        plan_policy: Any,
        agent_id: int,
        decision: Mapping[str, Any],
    ) -> dict:
        if self.mission_replan_mode == "off":
            return dict(decision)
        agent = plan_policy._agent_by_id.get(int(agent_id))
        if agent is None:
            return dict(decision)
        entity_id = int(agent.entity_id)
        binding = plan_policy._binding_by_entity.get(entity_id)
        if binding is None:
            return dict(decision)
        filtered = dict(decision)
        retarget_id = self._mission_retarget_by_entity.get(entity_id)
        if retarget_id is not None and int(agent.launch_step) >= 0:
            slot = next(
                (
                    index
                    for index, target in enumerate(
                        getattr(agent, "_sorted_targets", ())
                    )
                    if int(target.entity_id) == int(retarget_id)
                ),
                None,
            )
            if slot is not None:
                filtered["goal"] = int(slot)
                approved = getattr(
                    plan_policy, "_mission_approved_retargets", None
                )
                if approved is None:
                    approved = {}
                    plan_policy._mission_approved_retargets = approved
                approved[entity_id] = (
                    int(agent.current_goal_id),
                    int(retarget_id),
                )
                self._mission_retarget_by_entity.pop(entity_id, None)
        if (
            self.mission_replan_mode == "all-packages"
            and entity_id in self._mission_m_stopped_entities
        ):
            filtered["launch"] = None
            filtered["goal"] = None
            return filtered
        if not bool(binding.reserve_response):
            return filtered
        if entity_id in self._mission_cancelled_entities:
            filtered["launch"] = None
            filtered["goal"] = None
            return filtered
        if decision.get("launch") is not None and (
            entity_id not in self._mission_released_entities
        ):
            filtered["launch"] = None
            filtered["goal"] = None
        return filtered

    def _prepare_package_retargets(
        self,
        *,
        step: int,
        agents_by_id: Mapping[int, Any],
        plan_policy: Any | None,
    ) -> None:
        info_center = getattr(plan_policy, "info_center", None)
        if info_center is None:
            return
        self._sync_belief_targets(info_center, step)
        agents_by_entity = {
            int(agent.entity_id): agent
            for agent in agents_by_id.values()
            if hasattr(agent, "entity_id")
        }
        self._mission_retarget_history = [
            item for item in self._mission_retarget_history
            if step - item[0] < 100
        ]
        live_launched = sum(
            int(getattr(agent, "launch_step", -1)) >= 0
            and getattr(agent, "death_reason", None) is None
            for agent in agents_by_entity.values()
        )
        retarget_cap = max(4, math.ceil(live_launched * 0.25))
        used = sum(count for _event_step, count in self._mission_retarget_history)

        packages = [
            (group.group_id, 21000, group.entity_ids)
            for group in self._joint_groups
        ] + [
            (group.group_id, 21001, group.entity_ids)
            for group in self._m_groups
        ] + [
            (group.group_id, 21002, group.entity_ids)
            for group in self._l_response_groups
        ]
        targets = tuple(info_center.targets())
        for package_id, missile_type, entity_ids in packages:
            if package_id in self._mission_retargeted_packages:
                continue
            agents = [agents_by_entity.get(entity_id) for entity_id in entity_ids]
            eligible = [
                agent for agent in agents
                if agent is not None
                and int(getattr(agent, "launch_step", -1)) >= 0
                and getattr(agent, "death_reason", None) is None
                and int(getattr(agent, "_last_stage", 0)) < 4
                and getattr(agent, "current_goal_id", None) is not None
            ]
            if not eligible or used + len(eligible) > retarget_cap:
                continue
            source_ids = {int(agent.current_goal_id) for agent in eligible}
            if len(source_ids) != 1:
                continue
            source_id = next(iter(source_ids))
            source_belief = self._belief_for(source_id, info_center, step)
            if source_belief is None:
                continue
            source_done = source_belief.confirmed_destroyed or (
                source_belief.expected_hp <= 0.0
                and source_belief.confidence >= 0.95
                and source_belief.delivered_expected_damage
                >= source_belief.initial_health * 1.25
            )
            if not source_done:
                continue

            representative = eligible[0]
            observation = getattr(representative, "latest_observation", None) or {}
            own_position = _lla(
                _field(observation.get("self") or {}, "position")
            )
            if own_position is None:
                continue
            source_target = info_center.target_by_id(source_id)
            source_type = (
                None if source_target is None else int(source_target.entity_type)
            )
            candidates = []
            for target in targets:
                target_id = int(target.entity_id)
                target_type = int(target.entity_type)
                if target_id == source_id:
                    continue
                if missile_type in (21000, 21001) and target_type not in (9400, 9600):
                    continue
                if missile_type == 21002 and target_type != 9500:
                    continue
                if (
                    step - self._mission_target_last_assignment_step.get(
                        target_id, -10_000
                    )
                    < 60
                ):
                    continue
                belief = self._belief_for(target_id, info_center, step)
                if belief is None or belief.confirmed_destroyed:
                    continue
                if belief.expected_hp <= 0.0 or belief.projected_expected_hp <= 0.0:
                    continue
                coordinate_scale = float(
                    getattr(info_center, "coordinate_scale", 1.0)
                )
                distance_km = math.hypot(
                    (target.position.lon - own_position[0])
                    * 96.0
                    * coordinate_scale,
                    (target.position.lat - own_position[1])
                    * 110.0
                    * coordinate_scale,
                )
                speed = {21000: 2.0, 21001: 1.2, 21002: 0.3}[missile_type]
                eta = distance_km / speed
                type_penalty = 0 if source_type == target_type else 1
                weight = {9400: 5, 9600: 2, 9500: 1}[target_type]
                candidates.append(
                    (type_penalty, eta, -weight, target_id)
                )
            if not candidates:
                continue
            _type_penalty, _eta, _weight, target_id = min(candidates)
            for agent in eligible:
                self._mission_retarget_by_entity[int(agent.entity_id)] = target_id
            self._mission_retargeted_packages.add(package_id)
            self._mission_target_last_assignment_step[target_id] = step
            self._mission_retarget_history.append((step, len(eligible)))
            used += len(eligible)
            self._write_status_trace(
                "MISSION_REPLAN_TRACE",
                {
                    "step": step,
                    "retarget_package_id": package_id,
                    "retarget_member_ids": [
                        int(agent.entity_id) for agent in eligible
                    ],
                    "from_target_id": source_id,
                    "to_target_id": target_id,
                    "reason": "completed_target_nearest_legal_undercommitted",
                },
            )

    def _prepare_l_coordination(
        self,
        *,
        step: int,
        agent_observations: Mapping[int, Mapping[str, Any]],
        agents_by_id: Mapping[int, Any],
        plan_policy: Any | None,
    ) -> None:
        agents_by_entity = {
            int(agent.entity_id): agent
            for agent in agents_by_id.values()
            if hasattr(agent, "entity_id")
        }
        info_center = getattr(plan_policy, "info_center", None)
        plan_spec = getattr(plan_policy, "spec", None)
        hidden_response = getattr(plan_spec, "hidden_response", None)
        required_discoveries = int(
            getattr(hidden_response, "required_discoveries", 0) or 0
        )
        discovered_ids = tuple(
            sorted({
                int(item["target_id"])
                for item in getattr(info_center, "discovery_events", ())
            })
        )

        for group in self._l_search_groups:
            members = []
            for entity_id, cell in zip(group.entity_ids, group.cells, strict=True):
                agent = agents_by_entity.get(entity_id)
                if agent is None:
                    continue
                observation = agent_observations.get(int(agent.agent_id))
                if observation is None:
                    continue
                own = observation.get("self") or {}
                members.append(
                    SearchMemberSnapshot(
                        entity_id=entity_id,
                        search_cell=cell,
                        alive=float(_field(own, "health", default=0.0)) > 0.0
                        and bool(_field(own, "isVisible", default=True)),
                        launched=int(agent.launch_step) >= 0,
                        report_step=int(observation.get("step", -1)),
                        discovered_target_ids=discovered_ids,
                        baseline_action=1,
                    )
                )
            if len(members) < 2:
                continue
            decision = self._l_search_controllers[group.group_id].decide(
                step=step,
                members=tuple(members),
                required_discoveries=required_discoveries,
            )
            self._write_status_trace(
                "L_SEARCH_COORDINATION_TRACE",
                {
                    "step": step,
                    "group_id": group.group_id,
                    "cells": list(group.cells),
                    "entity_ids": list(group.entity_ids),
                    "discovered_target_ids": list(
                        decision.status.discovered_target_ids
                    ),
                    "response_ready": decision.status.response_ready,
                    "uncovered_cells": list(decision.status.uncovered_cells),
                    "reassignment_requests": [
                        request.__dict__ for request in decision.reassignment_requests
                    ],
                    "satellite_request": (
                        None
                        if decision.satellite_request is None
                        else decision.satellite_request.__dict__
                    ),
                },
            )

        if info_center is None:
            return
        for group in self._l_response_groups:
            agents = [agents_by_entity.get(entity_id) for entity_id in group.entity_ids]
            if any(agent is None for agent in agents):
                continue
            target_ids = {
                int(agent.current_goal_id)
                for agent in agents
                if getattr(agent, "current_goal_id", None) is not None
            }
            if len(target_ids) != 1:
                continue
            target_id = next(iter(target_ids))
            target = info_center.target_by_id(target_id)
            confirmed_dead = target_id in info_center.dead_target_ids
            if target is None and not confirmed_dead:
                continue
            if target is not None and int(target.entity_type) != 9500:
                continue
            belief = self._belief_for(target_id, info_center, step)
            if belief is None:
                continue
            members = []
            for agent in agents:
                observation = agent_observations.get(int(agent.agent_id))
                if observation is None:
                    continue
                own = observation.get("self") or {}
                members.append(
                    ResponseMemberSnapshot(
                        entity_id=int(agent.entity_id),
                        target_id=target_id,
                        alive=float(_field(own, "health", default=0.0)) > 0.0
                        and bool(_field(own, "isVisible", default=True)),
                        launched=int(agent.launch_step) >= 0,
                        delivered=int(getattr(agent, "_last_stage", 0)) >= 4
                        or (
                            getattr(agent, "_min_goal_dist", None) is not None
                            and float(agent._min_goal_dist) < 1.2
                        ),
                        launch_step=int(agent.launch_step),
                        estimated_eta_seconds=None,
                        baseline_action=1,
                    )
                )
            if len(members) < 2:
                continue
            decision = self._l_response_controllers[group.group_id].decide(
                members=tuple(members),
                belief=belief,
            )
            self._write_status_trace(
                "L_RESPONSE_PACKAGE_TRACE",
                {
                    "step": step,
                    "group_id": group.group_id,
                    "response_group_index": group.response_group_index,
                    "target_id": target_id,
                    "entity_ids": list(group.entity_ids),
                    "expected_hp": decision.status.expected_hp,
                    "projected_expected_hp": (
                        decision.status.projected_expected_hp
                    ),
                    "roles": [
                        [entity_id, role.value]
                        for entity_id, role in decision.status.roles
                    ],
                    "stop_commitment_request": (
                        None
                        if decision.stop_commitment_request is None
                        else decision.stop_commitment_request.__dict__
                    ),
                    "supplement_request": (
                        None
                        if decision.supplement_request is None
                        else decision.supplement_request.__dict__
                    ),
                    "retarget_request": (
                        None
                        if decision.retarget_request is None
                        else decision.retarget_request.__dict__
                    ),
                },
            )

    def _prepare_m_packages(
        self,
        *,
        step: int,
        agent_observations: Mapping[int, Mapping[str, Any]],
        agents_by_id: Mapping[int, Any],
        plan_policy: Any | None,
    ) -> None:
        info_center = getattr(plan_policy, "info_center", None)
        if info_center is None:
            return
        agents_by_entity = {
            int(agent.entity_id): agent
            for agent in agents_by_id.values()
            if hasattr(agent, "entity_id")
        }
        hard_scale = len(getattr(plan_policy, "entity_types", {})) > 200
        timeout = 1000 if hard_scale else 700
        for group in self._m_groups:
            target = info_center.target_by_id(group.target_id)
            confirmed_dead = group.target_id in info_center.dead_target_ids
            if target is None and not confirmed_dead:
                continue
            belief = self._belief_for(group.target_id, info_center, step)
            if belief is None:
                continue
            members = []
            for entity_id, release_step in zip(
                group.entity_ids, group.planned_release_steps, strict=True
            ):
                agent = agents_by_entity.get(entity_id)
                if agent is None:
                    continue
                observation = agent_observations.get(int(agent.agent_id))
                if observation is None:
                    continue
                own = observation.get("self") or {}
                members.append(
                    MMemberSnapshot(
                        entity_id=entity_id,
                        missile_type=21001,
                        target_id=group.target_id,
                        target_type=group.target_type,
                        planned_release_step=release_step,
                        alive=float(_field(own, "health", default=0.0)) > 0.0
                        and bool(_field(own, "isVisible", default=True)),
                        launched=int(agent.launch_step) >= 0,
                        delivered=int(getattr(agent, "_last_stage", 0)) >= 4
                        or (
                            getattr(agent, "_min_goal_dist", None) is not None
                            and float(agent._min_goal_dist) < 1.2
                        ),
                        baseline_action=1,
                    )
                )
            if len(members) < 2:
                continue
            context = MStrikeMissionContext(
                belief=belief,
                current_step=step,
                last_accepted_release_step=min(group.planned_release_steps),
                delivery_timeout_steps=timeout,
            )
            decision = self._m_controllers[group.group_id].decide(
                members=tuple(members),
                context=context,
            )
            if (
                self.mission_replan_mode == "all-packages"
                and decision.stop_request is not None
            ):
                self._mission_m_stopped_entities.update(
                    decision.stop_request.member_ids
                )
            self._write_status_trace(
                "M_STRIKE_PACKAGE_TRACE",
                {
                    "step": step,
                    "group_id": group.group_id,
                    "target_id": group.target_id,
                    "target_type": group.target_type,
                    "entity_ids": list(group.entity_ids),
                    "roles": [
                        [entity_id, role.value]
                        for entity_id, role in decision.status.roles
                    ],
                    "expected_hp": decision.status.expected_hp,
                    "projected_expected_hp": (
                        decision.status.projected_expected_hp
                    ),
                    "motion_overrides": list(decision.motion_overrides),
                    "stop_request": (
                        None
                        if decision.stop_request is None
                        else decision.stop_request.__dict__
                    ),
                    "supplement_request": (
                        None
                        if decision.supplement_request is None
                        else decision.supplement_request.__dict__
                    ),
                    "retarget_request": (
                        None
                        if decision.retarget_request is None
                        else decision.retarget_request.__dict__
                    ),
                },
            )

    @staticmethod
    def _write_status_trace(environment_name: str, record: Mapping[str, Any]) -> None:
        path_value = os.getenv(environment_name)
        if not path_value:
            return
        path = Path(path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(record), sort_keys=True) + "\n")

    def _evaluate_local_action(
        self, observation: Mapping[str, Any], context: Mapping[str, Any]
    ) -> int:
        agent_id = int(context["agent_id"])
        pulse = self._local_pulses.get(agent_id)
        if pulse is not None:
            action, remaining = pulse
            if remaining > 0:
                self._local_pulses[agent_id] = (action, remaining - 1)
                return int(action)
            self._local_pulses.pop(agent_id, None)
            return 1
        payload = self.local_policy.encode_motion_observation(
            observation, **dict(context)
        )
        mask = self.local_policy.prepare_action_mask(
            payload, np.ones(3, dtype=np.bool_)
        )
        if bool(mask[1]) and int(np.asarray(mask).sum()) == 1:
            return 1
        action = int(self.local_policy.select_action(payload, mask))
        if action != 1:
            self._local_pulses[agent_id] = (
                action,
                self.local_pulse_steps - 1,
            )
        return action

    def _joint_snapshots(
        self,
        step: int,
        group: JointGroup,
        agents: list[Any],
        observations: list[Mapping[str, Any]],
        baselines: tuple[int, ...],
    ) -> list[JointMemberSnapshot]:
        first_ecf = _xyz(_field(observations[0].get("self") or {}, "pos_ecf"))
        first_lla = _lla(_field(observations[0].get("self") or {}, "position"))
        if first_ecf is None or first_lla is None:
            raise ValueError("joint group lacks a coordinate origin")
        basis = _local_basis(first_lla[0], first_lla[1])
        snapshots = []
        for agent, observation, baseline in zip(
            agents, observations, baselines, strict=True
        ):
            own = observation.get("self") or {}
            ecf = _xyz(_field(own, "pos_ecf"))
            lla = _lla(_field(own, "position"))
            goal = agent._current_goal_position()
            if ecf is None or lla is None or goal is None:
                raise ValueError("joint member kinematics are incomplete")
            position = _project(_subtract(ecf, first_ecf), basis)
            goal_delta = _goal_vector(
                lla[0], lla[1], float(goal[0]), float(goal[1])
            )
            velocity = self.h_policy._own_velocity(
                int(agent.entity_id), int(step), ecf, goal_delta, basis
            )
            threats = []
            seen = set()
            for raw in _track_values(
                _field(own, "detectInfo", "detect_info", default={})
            ):
                try:
                    threat_id = int(
                        _field(raw, "entity_id", "entityId", "id")
                    )
                    threat_type = int(
                        _field(
                            raw,
                            "entity_type",
                            "entityType",
                            "type",
                            default=-1,
                        )
                    )
                except (TypeError, ValueError):
                    continue
                if threat_type != INTERCEPTOR_TYPE or threat_id in seen:
                    continue
                threat_ecf = _xyz(_field(raw, "pos_ecf", "posEcf"))
                threat_velocity = _xyz(
                    _field(raw, "vel_ecf", "velEcf", "velocity")
                )
                if threat_ecf is None or threat_velocity is None:
                    continue
                seen.add(threat_id)
                threat_position = _project(
                    _subtract(threat_ecf, first_ecf), basis
                )
                projected_velocity = _project(threat_velocity, basis)
                threats.append(
                    JointThreat(
                        threat_id=threat_id,
                        position=JointVector2(
                            threat_position.east, threat_position.north
                        ),
                        velocity=JointVector2(
                            projected_velocity.east,
                            projected_velocity.north,
                        ),
                    )
                )
            snapshots.append(
                JointMemberSnapshot(
                    entity_id=int(agent.entity_id),
                    missile_type=group.entity_type,
                    target_id=group.target_id,
                    position=JointVector2(position.east, position.north),
                    velocity=JointVector2(velocity.east, velocity.north),
                    goal_position=JointVector2(
                        position.east + goal_delta.east,
                        position.north + goal_delta.north,
                    ),
                    threats=tuple(threats),
                    baseline_action=int(baseline),
                    launch_step=int(agent.launch_step),
                )
            )
        return snapshots

    @staticmethod
    def _linear_minimum_separation(
        snapshots: list[JointMemberSnapshot], *, horizon_seconds: float
    ) -> float:
        minimum = float("inf")
        for sample in range(13):
            time_s = horizon_seconds * sample / 12.0
            positions = [
                (
                    item.position.east + item.velocity.east * time_s,
                    item.position.north + item.velocity.north * time_s,
                )
                for item in snapshots
            ]
            for first in range(len(snapshots)):
                for second in range(first + 1, len(snapshots)):
                    minimum = min(
                        minimum,
                        math.hypot(
                            positions[first][0] - positions[second][0],
                            positions[first][1] - positions[second][1],
                        ),
                    )
        return minimum

    @staticmethod
    def _joint_eta_spread(snapshots: list[JointMemberSnapshot]) -> float:
        values = []
        for item in snapshots:
            gx = item.goal_position.east - item.position.east
            gy = item.goal_position.north - item.position.north
            distance = math.hypot(gx, gy)
            radial = 0.0 if distance <= 1e-9 else (
                item.velocity.east * gx + item.velocity.north * gy
            ) / distance
            values.append(distance / max(50.0, radial))
        return max(values) - min(values)

    @staticmethod
    def _write_joint_trace(
        step: int,
        group: JointGroup,
        baselines: tuple[int, ...],
        actions: tuple[int, ...],
        reason: str,
        metrics: Mapping[str, Any],
    ) -> None:
        path_value = os.getenv("THREE_MISSILE_JOINT_TRACE")
        if not path_value:
            return
        path = Path(path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "step": step,
            "group_id": group.group_id,
            "entity_ids": list(group.entity_ids),
            "target_id": group.target_id,
            "baseline_actions": list(baselines),
            "actions": list(actions),
            "reason": reason,
            **dict(metrics),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def prepare_action_mask(
        self, observation: DispatchedObservation, action_mask: np.ndarray
    ) -> np.ndarray:
        supplied = np.asarray(action_mask, dtype=np.bool_)
        if supplied.shape != (3,):
            raise ValueError(f"Expected three maneuver actions, got {supplied.shape}")
        if observation.forced_action is not None or observation.delegate is None:
            action = observation.forced_action if observation.forced_action is not None else 1
            if not bool(supplied[action]):
                action = 1
            effective = np.zeros(3, dtype=np.bool_)
            effective[action] = True
            return effective
        return observation.delegate.prepare_action_mask(observation.payload, supplied)

    def select_action(
        self, observation: DispatchedObservation, action_mask: np.ndarray
    ) -> int:
        if observation.forced_action is not None or observation.delegate is None:
            selected = observation.forced_action if observation.forced_action is not None else 1
            if not bool(np.asarray(action_mask, dtype=np.bool_)[selected]):
                return 1
            return int(selected)
        selected = int(
            observation.delegate.select_action(observation.payload, action_mask)
        )
        if observation.delegate is self.local_policy and selected != 1:
            self._local_pulses[observation.agent_id] = (
                selected,
                self.local_pulse_steps - 1,
            )
        return selected

    def set_training(self, training: bool) -> None:
        if training:
            raise ValueError("The comparison dispatcher is evaluation-only")
        self.h_policy.set_training(False)
        self.local_policy.set_training(False)
        self.training = False

    def reset_episode(self) -> None:
        self.h_policy.reset_episode()
        self.local_policy.reset_episode()
        self._local_pulses.clear()
        self._prepared_step = None
        self._prepared_actions.clear()
        for controller in self._joint_controllers.values():
            controller.reset()
        for controller in self._l_search_controllers.values():
            controller.reset()
        for controller in self._l_response_controllers.values():
            controller.reset()
        for controller in self._m_controllers.values():
            controller.reset()
        if self._mission_replan is not None:
            self._mission_replan.reset()
        self._mission_released_entities.clear()
        self._mission_cancelled_entities.clear()
        self._mission_m_stopped_entities.clear()
        self._mission_retarget_by_entity.clear()
        self._mission_retargeted_packages.clear()
        self._mission_target_last_assignment_step.clear()
        self._mission_retarget_history.clear()

    def close(self) -> None:
        self.h_policy.close()
        close = getattr(self.local_policy, "close", None)
        if callable(close):
            close()


__all__ = [
    "EncodedManeuver",
    "DispatchedObservation",
    "HSingleManeuverRuntimePolicy",
    "INTERCEPTOR_TYPE",
    "JointGroup",
    "LResponseGroup",
    "LSearchGroup",
    "MStrikeGroup",
    "MissileTypeDispatchRuntimePolicy",
    "PlanMotionConfig",
    "default_profile_path",
]
