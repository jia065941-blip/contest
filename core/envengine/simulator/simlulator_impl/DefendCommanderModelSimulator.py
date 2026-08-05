# -*- coding: utf-8 -*-
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
from user_agents.blue_strategies import (
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
    """
    Defense command simulator.
    """

    def __init__(self, entity_ext: EntityExt,
                 send_commands: Callable[[list[Command]], None],
                 send_events: Callable[[dict], None],
                 simulator_factory: SimulatorFactory = None):
        super().__init__(entity_ext, send_commands, send_events, simulator_factory)
        self.launched_list: dict[int, list[int]] = {}
        self.blue_policy = build_blue_policy(
            name=os.getenv("BLUE_POLICY", "fixed_ratio_random"),
            max_shots_per_target=self._read_int_env("BLUE_INTERCEPTOR_RATIO", 5),
            seed=self._read_optional_int_env("BLUE_POLICY_SEED"),
        )

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
        """
        Receive radar detection tracks and ask the selected blue policy for launch assignments.
        """
        target_dict = self._entity_ext.entity.detectInfo
        for key, new_value in new_detect_info.items():
            old_value = target_dict.get(key)
            if old_value is None or new_value.time > old_value.time:
                target_dict[key] = new_value

        detect_info_list = list(new_detect_info.values())
        if detect_info_list:
            self.launch_intercept_missile(detect_info_list)

    def launch_intercept_missile(self, detect_info_list: list[DetectInfo]) -> None:
        if not detect_info_list:
            return

        observation = self._build_blue_observation(detect_info_list)
        assignments = self.blue_policy.decide(observation)
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

        interceptors = []
        for simulator in self._simulator_factory.get_simulators_by_type(24000):
            entity = simulator.entity_ext.entity
            interceptors.append(
                InterceptorState(
                    interceptor_id=entity.id,
                    lla=self._copy_vector(entity.lla),
                    pos_ecf=self._copy_vector(entity.posEcf),
                    vel_ecf=self._copy_vector(entity.velEcf),
                    available=entity.survivePoints > 0 and entity.id not in launched_interceptor_ids,
                )
            )

        assets = []
        for simulator in self._simulator_factory.get_simulators_by_type(9400):
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

    def reset(self) -> None:
        super().reset()
        self.launched_list = {}

    def command_received(self, command: Command) -> None:
        if command.commandTypeId == SimmerCommandType.RADAR_DETECT_STATUS_UPDATE:
            detect_info = command.commandAttributes
            self.handel_detect_info(detect_info)
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
        if callable(coord):
            return coord()
        return coord

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
