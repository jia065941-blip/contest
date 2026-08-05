import datetime
import time
from typing import Callable

from envengine.common import SimmerCommandType
from envengine.sdk.base_struct.Basic import Vector3d, DetectInfo
from envengine.sdk.base_struct.Entity import EntityExt
from envengine.sdk.base_struct.Message import Command
from envengine.sdk.Util import UtilsPy
from envengine.sdk.Util import State_py
from envengine.simulator.decorator import Simulator, timer_decorator
from envengine.simulator.interfaces import ISimulator
from envengine.simulator.models.OrbitModel.OrbitModelPy import JTC_OrbitModel
from envengine.simulator.simulator_factory import SimulatorFactory

EARTH_RADIUS = 6371000.0


@Simulator.register("TleSatelliteS")
class OrbitModelSimulator(ISimulator):
    """
    轨道模型仿真器
    """

    def __init__(self, entity_ext: EntityExt,
                 send_commands: Callable[[list[Command]], None],
                 send_events: Callable[[dict], None],
                 simulator_factory: SimulatorFactory = None):
        super().__init__(entity_ext, send_commands, send_events, simulator_factory)

        self.model = JTC_OrbitModel("1 68310U 26057S   26147.90258910  .00262022  00000-0  17035-2 0  9994",
                                    "2 68310  97.2843 274.2645 0001211  70.9562 289.1834 15.75199990")

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

        # 卫星轨道解算
        # dt = datetime.datetime.fromtimestamp(self.sim_time / 1000)
        # self.model.update(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond / 1000. / 1000.,
        #                   50 / 1000)
        # state: State_py = self.model.getState()
        #
        # pos = state.posEcf()
        # vel = state.velEcf()
        # lla = UtilsPy.CoordinateHelper.ecefToLla_py(pos)
        # entity = self.entity_ext.entity
        # entity.lla.x, entity.lla.y, entity.lla.z = lla.x(), lla.y(), lla.z()
        # entity.posEcf.x, entity.posEcf.y, entity.posEcf.z = pos.x(), pos.y(), pos.z()
        # entity.velEcf.x, entity.velEcf.y, entity.velEcf.z = vel.x(), vel.y(), vel.z()

        # 执行探测
        self.execute_detection(Vector3d(124.625763, 27.482257, 0))

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

    # @timer_decorator
    def send_detect_info(self):
        """
        发送探测信息
        """
        if self.entity_ext.entity.detectInfo:
            t0 = time.perf_counter()
            # 基于通信距离触发
            communication_candidate_simulators: list[ISimulator] = self._simulator_factory.get_simulators_by_side(
                self.entity_ext.entity.sideId)
            t1 = time.perf_counter()

            can_communication_simulators: list[ISimulator] = self._get_targets_within_self_range(
                communication_candidate_simulators,
                max_range=500 * 1000)
            t2 = time.perf_counter()
            # 如果已经发送过同样的探测信息，则不再发送
            can_communication_send_simulators = []
            for i in can_communication_simulators:
                if i.entity_ext.entity.id not in self._delivered_to:
                    can_communication_send_simulators.append(i)
                    self._delivered_to.add(i.entity_ext.entity.id)
            # 构建探测信息指令
            command_list: list[Command] = [Command(
                executorId=i.entity_ext.entity.id,
                commandTypeId=SimmerCommandType.DETECT_STATUS_UPDATE,
                commandAttributes=self.entity_ext.entity.detectInfo
            ) for i in can_communication_send_simulators]
            t3 = time.perf_counter()
            # print(f"获取己方实体时间：{t1-t0},获取通信距离内实体时间：{t2 - t1}, 构建探测指令时间{t3 - t2}")
            # 指令转发
            self._send_commands(command_list)

    def execute_detection(self, detect_point: Vector3d = None):
        """
        执行探测
        :param detect_point: 探测点
        :return:
        """
        detected_candidate_simulators: list[ISimulator] = self._simulator_factory.get_simulators_by_side(
            1 if self.entity_ext.entity.sideId == 0 else 0)
        detected: list[ISimulator] = self.detect_targets(detected_candidate_simulators, max_range=500 * 1000,
                                                         detect_radius=200 * 1000, detect_point=detect_point)
        if not detected:
            return
        # 组装探测信息
        detect_info: dict[int, DetectInfo] = {
            target.entity_ext.entity.id: DetectInfo(
                detect_from=self.entity_ext.entity.id,
                time=int(self.sim_time),
                entity_id=target.entity_ext.entity.id,
                nameChn=target.entity_ext.entity.nameChn,
                lla=target.entity_ext.entity.lla,
                pos_ecf=target.entity_ext.entity.posEcf,
                vel_ecf=target.entity_ext.entity.velEcf
            ) for target in detected}
        # 更新自身探测信息
        self.handel_detect_info(detect_info)

    def reset(self) -> None:
        """重置到初始状态"""
        super().reset()
        self.model = JTC_OrbitModel("1 68310U 26057S   26147.90258910  .00262022  00000-0  17035-2 0  9994",
                                    "2 68310  97.2843 274.2645 0001211  70.9562 289.1834 15.75199990")
