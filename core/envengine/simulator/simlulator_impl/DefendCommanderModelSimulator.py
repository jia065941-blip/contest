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
        self.launch_intercept_missile(detect_info_list)

    def launch_intercept_missile(self, detect_info_list: [DetectInfo]) -> None:
        """
        发射拦截弹 - 2拦1策略
        :param detect_info_list: 目标列表
        """
        INTERCEPTOR_RATIO = 5

        # 获取可用拦截弹
        interceptor_list = self._simulator_factory.get_simulators_by_type(24000)
        launched_interceptor_list = [num for sublist in self.launched_list.values() for num in sublist]
        un_launched_interceptor_list = [interceptor for interceptor in interceptor_list if
                                        interceptor.entity_ext.entity.id not in launched_interceptor_list]

        if not un_launched_interceptor_list or not detect_info_list:
            return

        # 计算每个目标需要的拦截弹数量
        target_needs = {}
        for detect_info in detect_info_list:
            target_id = detect_info.entity_id
            if target_id in self.launched_list:
                need = max(0, INTERCEPTOR_RATIO - len(self.launched_list[target_id]))
            else:
                need = INTERCEPTOR_RATIO
            if need > 0:
                target_needs[target_id] = need

        if not target_needs:
            return

        # 按需求排序（优先拦截需求多的目标）
        sorted_targets = sorted(target_needs.items(), key=lambda x: x[1], reverse=True)

        # 分配拦截弹
        command_list = []
        available_interceptors = un_launched_interceptor_list.copy()

        for target_id, need in sorted_targets:
            if not available_interceptors:
                break

            # 找到对应的目标信息
            target_info = next((t for t in detect_info_list if t.entity_id == target_id), None)
            if not target_info:
                continue

            # 分配拦截弹
            allocated = min(need, len(available_interceptors))
            # selected = available_interceptors[:allocated]
            # available_interceptors = available_interceptors[allocated:]

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
                                target_info.pos_ecf.x,
                                target_info.pos_ecf.y,
                                target_info.pos_ecf.z
                            ),
                            cVel=Vector3D(
                                target_info.vel_ecf.x,
                                target_info.vel_ecf.y,
                                target_info.vel_ecf.z
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
