# -*-coding:utf-8 -*-

import json
import math
from typing import Callable

from envengine.sdk.Util.UtilsPy import Vector3D

from envengine.common import SimmerCommandType
from envengine.sdk.Util import UtilsPy
from envengine.sdk.base_struct.Basic import Vector3d, DetectInfo
from envengine.sdk.base_struct.Entity import EntityExt
from envengine.sdk.base_struct.Message import Command

from envengine.simulator.decorator import Simulator
from envengine.simulator.interfaces import ISimulator
from envengine.simulator.simulator_factory import SimulatorFactory


@Simulator.register("NormalRadarS")
class RadarModelSimulator(ISimulator):
    """
    轨道模型仿真器
    """

    def __init__(self, entity_ext: EntityExt,
                 send_commands: Callable[[list[Command]], None],
                 send_events: Callable[[dict], None],
                 simulator_factory: SimulatorFactory = None):
        super().__init__(entity_ext, send_commands, send_events, simulator_factory)
        self._detection_range_m = self._read_detection_range(self.entity_ext)
        self.entity_ext.entity.stage = 1

    TRACK_TTL_MS = 30_000

    @staticmethod
    def _read_detection_range(entity_ext: EntityExt) -> float:
        """Read and validate the radar range declared by the scenario."""
        radar_id = entity_ext.entity.id
        try:
            external = json.loads(entity_ext.external)
            distance = external["raderExternal"]["distance"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(
                f"雷达实体 {radar_id} 缺少有效的 external.raderExternal.distance 配置"
            ) from error

        if (
            isinstance(distance, bool)
            or not isinstance(distance, (int, float))
            or not math.isfinite(distance)
            or distance <= 0
        ):
            raise ValueError(
                f"雷达实体 {radar_id} 的 external.raderExternal.distance 必须是正数（米）"
            )
        return float(distance)

    @property
    def simulator_sim_step(self) -> float:
        """
        仿真器内部步长（毫秒）
        :return: 内部步长（毫秒）
        """
        return 1000

    def update(self) -> None:
        """
        执行仿真步进
        """

        # 执行探测
        self.execute_detection()

        # Tracks are finite-lived sensor snapshots, not live truth references.
        self._expire_stale_tracks()

        # 发送探测信息
        self.send_detect_info()

    def set_lla(self, lla: Vector3d) -> None:
        """
        初始设置lla
        """
        pass

    def set_speed(self, speed: float) -> None:
        """
        初始设置速度
        """
        pass

    def send_detect_info(self):
        """
        发送探测信息
        """
        if self.entity_ext.entity.detectInfo:
            # 获取拦截指控
            can_communication_simulators: list[ISimulator] = self._simulator_factory.get_simulators_by_type(35000)
            # 构建探测信息指令
            command_list: list[Command] = [Command(
                executorId=i.entity_ext.entity.id,
                commandTypeId=SimmerCommandType.RADAR_DETECT_STATUS_UPDATE,
                commandAttributes=self.entity_ext.entity.detectInfo
            ) for i in can_communication_simulators]
            # 指令转发
            self._send_commands(command_list)

    def execute_detection(self):
        """
        执行探测
        :return:
        """

        # 获取所有被探测到的实体
        detected_candidate_simulators: list[ISimulator] = self._simulator_factory.get_simulators_by_side(
            1 if self.entity_ext.entity.sideId == 0 else 0)
        detected: list[ISimulator] = self._get_targets_within_self_range(detected_candidate_simulators,
                                                                         max_range=self._detection_range_m)
        # 只有蓝方有雷达，仅探测 stage >= 3 的实体（滑翔段及以后）
        detected = [i for i in detected if i.entity_ext.entity.stage >= 3]

        if not detected:
            return

        # 组装探测信息
        detect_info: dict[int, DetectInfo] = {
            target.entity_ext.entity.id: DetectInfo(
                detect_from=self.entity_ext.entity.id,
                time=int(self.sim_time),
                entity_id=target.entity_ext.entity.id,
                entity_type=target.entity_ext.entity.entityType,
                nameChn=target.entity_ext.entity.nameChn,
                lla=self._copy_vector(target.entity_ext.entity.lla),
                pos_ecf=self._copy_vector(target.entity_ext.entity.posEcf),
                vel_ecf=self._copy_vector(target.entity_ext.entity.velEcf)
            ) for target in detected}
        # 更新自身探测信息
        self.handel_detect_info(detect_info)

    def _expire_stale_tracks(self) -> None:
        tracks = self.entity_ext.entity.detectInfo
        expired_ids = [
            target_id
            for target_id, track in tracks.items()
            if self.sim_time - track.time > self.TRACK_TTL_MS
        ]
        for target_id in expired_ids:
            del tracks[target_id]

    @staticmethod
    def _copy_vector(vector: Vector3d) -> Vector3d:
        return Vector3d(vector.x, vector.y, vector.z)

    def reset(self) -> None:
        """重置到初始状态"""
        super().reset()
        posEcf: Vector3D = UtilsPy.CoordinateHelper.llaToEcef_py(
            Vector3D(self.entity_ext.entity.lla.x, self.entity_ext.entity.lla.y, self.entity_ext.entity.lla.z))

        self.entity_ext.entity.posEcf = Vector3d(
            posEcf.x(),
            posEcf.y(),
            posEcf.z()
        )

    def init_model(self) -> None:
        pass
