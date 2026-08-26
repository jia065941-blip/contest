"""Blue defense commander wired to the selectable baseline policies."""

import logging
import math
import os
from typing import Callable

from envengine.sdk.Util.UtilsPy import Vector3D

from envengine.common import SimmerCommandType, InterceptorLaunchCmd, PosVelAcc
from envengine.sdk.base_struct.Basic import Vector3d, DetectInfo
from envengine.sdk.base_struct.Entity import EntityExt
from envengine.sdk.base_struct.Message import Command
from envengine.simulator.decorator import Simulator
from envengine.simulator.interfaces import ISimulator
from envengine.simulator.simulator_factory import SimulatorFactory
from policies.blue import (
    BlueObservation,
    DefendedAsset,
    InterceptorAssignment,
    InterceptorState,
    ThreatTrack,
    Vector3 as BlueVector3,
    build_blue_policy,
)


@Simulator.register("DefendCommanderS")
class DefendCommanderModelSimulator(ISimulator):
    """Translate engine observations into assignments from ``BLUE_POLICY``."""

    def __init__(
        self,
        entity_ext: EntityExt,
        send_commands: Callable[[list[Command]], None],
        send_events: Callable[[dict], None],
        simulator_factory: SimulatorFactory = None,
    ):
        super().__init__(entity_ext, send_commands, send_events, simulator_factory)
        self.launched_list: dict[int, list[int]] = {}
        self._last_policy_decision_time = float("-inf")
        self.blue_policy = build_blue_policy(
            name=os.getenv("BLUE_POLICY", "b0_fixed_ratio_random"),
            # Keep the engine's historic 2-on-1 fire scale unless the caller
            # explicitly requests a different salvo size.
            max_shots_per_target=self._read_int_env("BLUE_INTERCEPTOR_RATIO", 2),
            seed=self._read_optional_int_env("BLUE_POLICY_SEED"),
        )
        logging.info("Blue defense policy activated: %s", self.blue_policy.name)

    @property
    def simulator_sim_step(self) -> float:
        return 1000

    def update(self) -> None:
        pass

    def set_lla(self, lla: Vector3d) -> None:
        pass

    def set_speed(self, speed: float) -> None:
        pass

    def handel_detect_info(self, new_detect_info: dict[int, DetectInfo]) -> None:
        """Store newer radar tracks and dispatch the selected policy."""
        target_dict = self._entity_ext.entity.detectInfo
        for key, new_value in new_detect_info.items():
            old_value = target_dict.get(key)
            if old_value is None or new_value.time > old_value.time:
                target_dict[key] = new_value

        detect_info_list = list(new_detect_info.values())
        if detect_info_list:
            self.launch_intercept_missile(detect_info_list)

    def launch_intercept_missile(self, detect_info_list: list[DetectInfo]) -> None:
        if (
            self.blue_policy.name == "b3_min_cost_assignment"
            and self.sim_time - self._last_policy_decision_time
            < self._read_int_env("BLUE_MIN_COST_DECISION_INTERVAL", 5)
        ):
            return
        self._last_policy_decision_time = self.sim_time
        observation = self._build_blue_observation(detect_info_list)
        assignments = [
            item for item in self.blue_policy.decide(observation)
            if self._assignment_is_launch_eligible(item)
        ]
        command_list = [self._assignment_to_command(item) for item in assignments]

        for item in assignments:
            launched = self.launched_list.setdefault(item.target_id, [])
            if item.interceptor_id not in launched:
                launched.append(item.interceptor_id)

        if command_list:
            self._send_commands(command_list)

    def _build_blue_observation(self, detect_info_list: list[DetectInfo]) -> BlueObservation:
        targets = [self._detect_info_to_threat_track(item) for item in detect_info_list]
        targets = [item for item in targets if item is not None]
        launched_interceptor_ids = {item for items in self.launched_list.values() for item in items}
        candidate_entities = [
            simulator.entity_ext.entity
            for simulator in self._simulator_factory.get_simulators_by_type(24000)
            if simulator.entity_ext.entity.survivePoints > 0
            and simulator.entity_ext.entity.id not in launched_interceptor_ids
        ]
        # A target with no currently eligible interceptor cannot receive an
        # assignment this tick. Excluding it keeps min-cost matching bounded
        # while allowing it to re-enter on a later radar update.
        targets = [
            target for target in targets
            if any(
                self._interceptor_is_launch_eligible(entity, target.pos_ecf, target.vel_ecf)
                for entity in candidate_entities
            )
        ]

        interceptors = []
        for simulator in self._simulator_factory.get_simulators_by_type(24000):
            entity = simulator.entity_ext.entity
            interceptors.append(
                InterceptorState(
                    interceptor_id=entity.id,
                    lla=self._copy_vector(entity.lla),
                    pos_ecf=self._copy_vector(entity.posEcf),
                    vel_ecf=self._copy_vector(entity.velEcf),
                    # Keep candidates that can engage at least one current
                    # track. This prevents the optimization policy from
                    # evaluating every stored interceptor on every radar tick.
                    available=(
                        entity.survivePoints > 0
                        and entity.id not in launched_interceptor_ids
                        and self._interceptor_has_launch_window(entity, targets)
                    ),
                )
            )

        assets = []
        for entity_type in (9400, 9500, 9600):
            for simulator in self._simulator_factory.get_simulators_by_type(entity_type):
                entity = simulator.entity_ext.entity
                if entity.survivePoints <= 0 or not entity.isVisible:
                    continue
                assets.append(
                    DefendedAsset(
                        asset_id=entity.id,
                        name=entity.nameChn,
                        lla=self._copy_vector(entity.lla),
                        pos_ecf=self._copy_vector(entity.posEcf),
                        health=entity.survivePoints,
                        value=self._asset_value(entity.nameChn),
                    )
                )

        return BlueObservation(
            sim_time=self.sim_time,
            targets=targets,
            interceptors=interceptors,
            assets=assets,
            launched_map={key: value.copy() for key, value in self.launched_list.items()},
        )

    def _detect_info_to_threat_track(self, detect_info: DetectInfo) -> ThreatTrack | None:
        target_simulator = self._simulator_factory.get_simulator_by_id(detect_info.entity_id)
        if target_simulator is not None:
            entity = target_simulator.entity_ext.entity
            if entity.survivePoints <= 0 or not entity.isVisible:
                return None
            target_type = entity.entityType
            target_name = entity.nameChn
        else:
            target_type = 0
            target_name = detect_info.nameChn

        return ThreatTrack(
            target_id=detect_info.entity_id,
            name=target_name,
            type=target_type,
            lla=self._copy_vector(detect_info.lla),
            pos_ecf=self._copy_vector(detect_info.pos_ecf),
            vel_ecf=self._copy_vector(detect_info.vel_ecf),
            detect_time=detect_info.time,
        )

    def _assignment_to_command(self, assignment: InterceptorAssignment) -> Command:
        target_pos = assignment.target_pos_ecf
        target_vel = assignment.target_vel_ecf
        return Command(
            executorId=assignment.interceptor_id,
            commandTypeId=SimmerCommandType.INTERCEPTOR_LAUNCH,
            commandAttributes=InterceptorLaunchCmd(
                weaponId=assignment.interceptor_id,
                targetId=assignment.target_id,
                targetPosLLA=PosVelAcc(
                    cPos=Vector3D(target_pos.x, target_pos.y, target_pos.z),
                    cVel=Vector3D(target_vel.x, target_vel.y, target_vel.z),
                ),
            ),
        )

    def _assignment_is_launch_eligible(self, assignment: InterceptorAssignment) -> bool:
        """Preserve the original engine's platform angle/range launch gate."""
        interceptor = self._simulator_factory.get_simulator_by_id(assignment.interceptor_id)
        if interceptor is None:
            return False
        return self._interceptor_is_launch_eligible(
            interceptor.entity_ext.entity,
            assignment.target_pos_ecf,
            assignment.target_vel_ecf,
        )

    def _interceptor_has_launch_window(
        self,
        interceptor_entity,
        targets: list[ThreatTrack],
    ) -> bool:
        return any(
            self._interceptor_is_launch_eligible(
                interceptor_entity,
                target.pos_ecf,
                target.vel_ecf,
            )
            for target in targets
        )

    def _interceptor_is_launch_eligible(self, interceptor_entity, target_pos, target_vel) -> bool:
        parent_id = interceptor_entity.parentId
        for entity_type, max_angle in ((9500, 30.0), (9600, 20.0)):
            for platform in self._simulator_factory.get_simulators_by_type(entity_type):
                entity = platform.entity_ext.entity
                if entity.id != parent_id:
                    continue
                relative = self._subtract(entity.posEcf, target_pos)
                if self._vector_length(relative) >= 600_000:
                    return False
                return self._vector_angle(relative, target_vel) < max_angle
        return False

    def reset(self) -> None:
        super().reset()
        self.launched_list = {}
        self._last_policy_decision_time = float("-inf")

    def init_model(self):
        pass

    def command_received(self, command: Command) -> None:
        if command.commandTypeId == SimmerCommandType.RADAR_DETECT_STATUS_UPDATE:
            self.handel_detect_info(command.commandAttributes)
        else:
            super().command_received(command)

    @staticmethod
    def _asset_value(name: str) -> float:
        if name.startswith("目标"):
            return 10.0
        if name.startswith("拦截阵地"):
            return 6.0
        if name.startswith("无人船"):
            return 3.0
        return 1.0

    @staticmethod
    def _copy_vector(value) -> BlueVector3:
        return BlueVector3(
            float(DefendCommanderModelSimulator._get_coord(value, "x")),
            float(DefendCommanderModelSimulator._get_coord(value, "y")),
            float(DefendCommanderModelSimulator._get_coord(value, "z")),
        )

    @staticmethod
    def _get_coord(value, name: str) -> float:
        coord = getattr(value, name, 0.0)
        return coord() if callable(coord) else coord

    @staticmethod
    def _subtract(left, right) -> BlueVector3:
        return BlueVector3(
            DefendCommanderModelSimulator._get_coord(left, "x") - right.x,
            DefendCommanderModelSimulator._get_coord(left, "y") - right.y,
            DefendCommanderModelSimulator._get_coord(left, "z") - right.z,
        )

    @staticmethod
    def _vector_length(value) -> float:
        return math.sqrt(value.x ** 2 + value.y ** 2 + value.z ** 2)

    @staticmethod
    def _vector_angle(left, right) -> float:
        left_length = DefendCommanderModelSimulator._vector_length(left)
        right_length = DefendCommanderModelSimulator._vector_length(right)
        if left_length == 0 or right_length == 0:
            return 0.0
        cosine = (left.x * right.x + left.y * right.y + left.z * right.z) / (left_length * right_length)
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    @staticmethod
    def _read_int_env(name: str, default: int) -> int:
        raw_value = os.getenv(name)
        if raw_value is None:
            return default
        try:
            return int(raw_value)
        except ValueError:
            return default

    @staticmethod
    def _read_optional_int_env(name: str) -> int | None:
        raw_value = os.getenv(name)
        if raw_value is None or raw_value == "":
            return None
        try:
            return int(raw_value)
        except ValueError:
            return None
