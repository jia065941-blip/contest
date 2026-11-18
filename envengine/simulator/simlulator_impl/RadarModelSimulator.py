# -*-coding:utf-8 -*-

from typing import Callable

from envengine.sdk.Util.UtilsPy import Vector3D

from envengine.common import SimmerCommandType
from envengine.sdk.Util import UtilsPy
from envengine.sdk.base_struct.Basic import DetectInfo
from envengine.sdk.base_struct.Entity import EntityExt
from envengine.sdk.base_struct.Message import Command

from envengine.simulator.decorator import Simulator
from envengine.simulator.interfaces import ISimulator
from envengine.simulator.simulator_factory import SimulatorFactory
from envengine.common.BaseStruct import Vector3d


@Simulator.register("NormalRadarS")
class RadarModelSimulator(ISimulator):
    """
    视野塔探测仿真器
    """

    def __init__(self, entity_ext: EntityExt,
                 send_commands: Callable[[list[Command]], None],
                 send_events: Callable[[dict], None],
                 simulator_factory: SimulatorFactory = None):
        super().__init__(entity_ext, send_commands, send_events, simulator_factory)
        self.entity_ext.entity.stage = 1

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

        # 发送探测信息
        self.send_detect_info()

    def set_position(self, pos: Vector3d) -> None:
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
            # 获取虫族控制
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
        按20千米三维距离探测对方存活、可见且已进入机动或进攻阶段的实体
        场景external.raderExternal.distance字段不参与距离判断
        :return:
        """

        # 获取所有被探测到的实体
        detected_candidate_simulators: list[ISimulator] = self._simulator_factory.get_simulators_by_side(
            1 if self.entity_ext.entity.sideId == 0 else 0)
        detected: list[ISimulator] = self._get_targets_within_self_range(detected_candidate_simulators,
                                                                         max_range=20 * 1000)
        # 原生巡航阶段3现映射为公开阶段2；保留巡航及以后可探测的规则。
        detected = [i for i in detected if i.entity_ext.entity.stage >= 2]

        if not detected:
            return

        # 组装探测信息
        detect_info: dict[int, DetectInfo] = {
            target.entity_ext.entity.id: DetectInfo(
                detect_from=self.entity_ext.entity.id,
                step=self.current_step,
                entity_id=target.entity_ext.entity.id,
                entity_type=target.entity_ext.entity.entityType,
                nameChn=target.entity_ext.entity.nameChn,
                pos=Vector3d(target.entity_ext.entity.posEcf.x, target.entity_ext.entity.posEcf.y, target.entity_ext.entity.posEcf.z),
                vel=Vector3d(target.entity_ext.entity.velEcf.x, target.entity_ext.entity.velEcf.y, target.entity_ext.entity.velEcf.z)
            ) for target in detected}
        # 更新自身探测信息
        self.handel_detect_info(detect_info)

    def reset(self) -> None:
        """重置到初始状态"""
        super().reset()

    def init_model(self) -> None:
        pass
