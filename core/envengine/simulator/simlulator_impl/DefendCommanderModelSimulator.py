import math
import random
from typing import Callable

from envengine.sdk.Util.UtilsPy import Vector3D

from envengine.common import SimmerCommandType, InterceptorLaunchCmd, PosVelAcc
from envengine.sdk.base_struct.Basic import Vector3d, DetectInfo
from envengine.sdk.base_struct.Entity import EntityExt
from envengine.sdk.base_struct.Message import Command
from envengine.simulator.decorator import Simulator
from envengine.simulator.interfaces import ISimulator
from envengine.simulator.simulator_factory import SimulatorFactory


@Simulator.register("DefendCommanderS")
class DefendCommanderModelSimulator(ISimulator):
    """
    轨道模型仿真器
    """

    def __init__(self, entity_ext: EntityExt,
                 send_commands: Callable[[list[Command]], None],
                 send_events: Callable[[dict], None],
                 simulator_factory: SimulatorFactory = None):
        super().__init__(entity_ext, send_commands, send_events, simulator_factory)
        # 已发射拦截弹列表，key为被拦截目标id，value为发射的拦截弹id列表
        self.launched_list: dict[int, list[int]] = {}

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
        pass

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

    def handel_detect_info(self, new_detect_info: dict[int, DetectInfo]) -> None:
        """
        接收雷达的探测信息, 控制拦截弹发射
        :param new_detect_info: 新探测信息
        """

        target_dict = self._entity_ext.entity.detectInfo
        for key, new_value in new_detect_info.items():
            old_value = target_dict.get(key)
            # 如果不存在 或 新数据时间更新，则更新
            if old_value is None or new_value.time > old_value.time:
                target_dict[key] = new_value

        # 汇总目标信息
        detect_info_list = list(new_detect_info.values())
        if not detect_info_list:
            return

        # 发射拦截弹
        self.launch_intercept_for_target(detect_info_list)

    def launch_intercept_for_target(self, detect_info_list: list[DetectInfo]):
        """
        发射拦截弹 - 2拦1策略
        :param detect_info_list: 目标列表
        """

        INTERCEPTOR_RATIO = 2

        all_interceptor_list = self._simulator_factory.get_simulators_by_type(24000)
        launched_interceptor_list = [num for sublist in self.launched_list.values() for num in sublist]
        un_launched_interceptor_list = [interceptor for interceptor in all_interceptor_list if
                                        interceptor.entity_ext.entity.id not in launched_interceptor_list]

        command_list = []

        for target in detect_info_list:
            # 获取可用拦截弹
            interceptor_list = self.filter_interceptor_by_angle(target, un_launched_interceptor_list)

            if not interceptor_list:
                continue

            # 计算目标需要的拦截弹数量
            target_need = 0
            target_id = target.entity_id
            if target_id in self.launched_list:
                target_need = max(0, INTERCEPTOR_RATIO - len(self.launched_list[target_id]))
            else:
                target_need = INTERCEPTOR_RATIO

            if target_need <= 0:
                continue

            # 分配拦截弹
            available_interceptors = interceptor_list.copy()

            # 分配拦截弹
            allocated = min(target_need, len(available_interceptors))

            # 生成指令
            for _ in range(allocated):
                interceptor = random.choice(available_interceptors)
                available_interceptors.remove(interceptor)
                command_list.append(Command(
                    executorId=interceptor.entity_ext.entity.id,
                    commandTypeId=SimmerCommandType.INTERCEPTOR_LAUNCH,
                    commandAttributes=InterceptorLaunchCmd(
                        weaponId=interceptor.entity_ext.entity.id,
                        targetId=target_id,
                        targetPosLLA=PosVelAcc(
                            cPos=Vector3D(
                                target.pos_ecf.x,
                                target.pos_ecf.y,
                                target.pos_ecf.z
                            ),
                            cVel=Vector3D(
                                target.vel_ecf.x,
                                target.vel_ecf.y,
                                target.vel_ecf.z
                            )
                        )
                    )
                ))

                # 更新已发射列表
                if target_id not in self.launched_list:
                    self.launched_list[target_id] = []
                self.launched_list[target_id].append(interceptor.entity_ext.entity.id)

        if command_list:
            self._send_commands(command_list)

    def filter_interceptor_by_angle(self, target: DetectInfo, un_launched_interceptor_list: list[ISimulator])->list[ISimulator]:
        """
        可用拦截弹：满足发射角度
        :param target:
        :return:
        """

        # return un_launched_interceptor_list

        interceptor_angle_test = []
        # 按照拦截阵地和无人船的位置进行判断，如果角度满足，则其上挂载的拦截弹可以发射
        ship_id = []
        for wr in self._simulator_factory.get_simulators_by_type(9500):
            re_pos = wr.entity_ext.entity.posEcf - target.pos_ecf
            re_vel = target.vel_ecf

            dis = math.sqrt(re_pos.x ** 2 + re_pos.y ** 2 + re_pos.z ** 2)
            if self.get_vector_angle(re_pos, re_vel) < 30 and dis < 600_000:
                ship_id.append(wr.entity_ext.entity.id)

        for wr in self._simulator_factory.get_simulators_by_type(9600):
            re_pos = wr.entity_ext.entity.posEcf - target.pos_ecf
            re_vel = target.vel_ecf

            dis = math.sqrt(re_pos.x ** 2 + re_pos.y ** 2 + re_pos.z ** 2)
            if self.get_vector_angle(re_pos, re_vel) < 20 and dis < 600_000:
                ship_id.append(wr.entity_ext.entity.id)

        for interceptor in un_launched_interceptor_list:
            if interceptor.entity_ext.entity.parentId in ship_id:
                interceptor_angle_test.append(interceptor)
        return interceptor_angle_test


    @staticmethod
    def get_vector_angle(vector1: Vector3d, vector2: Vector3d) -> float:
        """
        计算两个向量的夹角
        :param vector1: 向量1
        :param vector2: 向量2
        :return: 夹角（度）
        """
        # 点积
        dot = vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z
        # 模长
        norm1 = math.sqrt(vector1.x ** 2 + vector1.y ** 2 + vector1.z ** 2)
        norm2 = math.sqrt(vector2.x ** 2 + vector2.y ** 2 + vector2.z ** 2)
        # 零向量时夹角视为 0 度
        if norm1 == 0 or norm2 == 0:
            return 0.0
        # 余弦值（截断到 [-1, 1] 防止浮点误差）
        cos_theta = max(-1.0, min(1.0, dot / (norm1 * norm2)))
        return math.degrees(math.acos(cos_theta))

    def reset(self) -> None:
        """重置到初始状态"""
        super().reset()
        self.launched_list = {}

    def init_model(self):
        pass

    def command_received(self, command: Command) -> None:
        """
        自定义指令接收
        :param command:
        :return:
        """
        if command.commandTypeId == SimmerCommandType.RADAR_DETECT_STATUS_UPDATE:
            detect_info = command.commandAttributes
            self.handel_detect_info(detect_info)
        else:
            super().command_received(command)
