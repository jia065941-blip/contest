import logging
import math
from typing import Callable

from blue_defense import ThreatTrack
from envengine.common import SimmerCommandType, DamageC, DamageCmd
from envengine.sdk.base_struct.Entity import EntityExt
from envengine.sdk.base_struct.Message import Command
from envengine.sdk.Util import UtilsPy
from envengine.simulator.decorator import Simulator
from envengine.simulator.interfaces import ISimulator
from envengine.simulator.models.SM6_1BMissileModel import Missile
from envengine.simulator.simulator_factory import SimulatorFactory
from envengine.simulator.util.scaled_coordinates import unscaled_ecef_velocity
from envengine.simulator.util.scaled_coordinates import xyz_to_lla_position, xyz_to_ecf_position, xyz_to_ecf_velocity
from envengine.common.BaseStruct import Vector3d


@Simulator.register("InterceptorS")
class InterceptorSimulator(ISimulator):
    """
    自爆虫仿真器
    """

    def __init__(self, entity_ext: EntityExt,
                 send_commands: Callable[[list[Command]], None],
                 send_events: Callable[[dict], None],
                 simulator_factory: SimulatorFactory = None):
        super().__init__(entity_ext, send_commands, send_events, simulator_factory)
        self.ret = -1
        self.launched = -1
        self.damage_point = 5000
        self.target_id = None

    @property
    def simulator_sim_step(self) -> float:
        """
        仿真器内部步长（毫秒）
        :return: 内部步长（毫秒），返回0表示使用引擎步长
        """
        return 50

    def update(self) -> None:
        """
        执行仿真步进
        """
        if self.launched == -1 or self.entity_ext.entity.survivePoints <= 0:
            return

        if self.ret < 0:
            self.ret = self.model.Update()
            state: UtilsPy.State_py = self.model.getState()
            self._update_state(state)

            self.set_target_ecf()

            if self.ret > 0 and self.entity_ext.entity.isVisible:
                # 自爆
                self.entity_ext.entity.isVisible = False
                self.entity_ext.entity.survivePoints = 0
                # 构建自爆掉血指令
                self_command_list: list[Command] = [Command(
                    prevTriggerId=self.entity_ext.entity.id,
                    executorId=self.entity_ext.entity.id,
                    commandTypeId=SimmerCommandType.DAMAGE,
                    commandAttributes=DamageC(
                        DamageCmd(
                            damagePoint=self.damage_point
                        )
                    ).to_json()
                )]

                # 指令转发
                self._send_commands(self_command_list)

                # 单位触发范围效果后，筛选周围实体并结算范围内的伤害
                damage_candidate_simulators: list[ISimulator] = self._simulator_factory.get_simulators_by_side(
                    1 if self.entity_ext.entity.sideId == 0 else 0)
                can_damage_simulators: list[ISimulator] = self._get_targets_within_self_range(
                    damage_candidate_simulators,
                    max_range=100)
                if not can_damage_simulators:
                    return

                # 构建掉血指令
                command_list: list[Command] = [Command(
                    prevTriggerId=self.entity_ext.entity.id,
                    executorId=i.entity_ext.entity.id,
                    commandTypeId=SimmerCommandType.DAMAGE,
                    commandAttributes=DamageC(
                        DamageCmd(
                            damagePoint=self.damage_point
                        )
                    ).to_json()
                ) for i in can_damage_simulators]

                # 指令转发
                self._send_commands(command_list)

    def launch(self, target_id: int, target_pos: Vector3d, target_vel: Vector3d):
        """
        释放自爆虫
        :param target_id: 目标id
        :param target_pos: 目标位置
        :param target_vel: 目标速度
        :return:
        """

        if self._simulator_factory is None:
            return

        if self.launched != -1:
            return

        self.target_id = target_id

        target:ThreatTrack|None = self._simulator_factory.get_blue_track(self.target_id, self.current_step)
        if target is None:
            return

        # 将信息库预测的引擎坐标转换到自爆虫模型使用的尺度。
        pos = Vector3d(target.pos.x, target.pos.y, target.pos.z)
        target_lla = xyz_to_lla_position(pos * ISimulator.MODEL_SCALE)

        self.leave_parent()
        self.launched = 1
        self.model.Launch(UtilsPy.Vector3D(target_lla.x, target_lla.y, target_lla.z))

    def set_target_ecf(self) -> None:
        """
        设置目标位置
        :return:
        """

        if self._simulator_factory is None:
            return

        target:ThreatTrack|None = self._simulator_factory.get_blue_track(self.target_id, self.current_step)
        if target is None:
            return

        # 将信息库预测的引擎坐标转换到自爆虫模型使用的尺度。
        pos = Vector3d(target.pos.x, target.pos.y, target.pos.z)
        vel = Vector3d(target.vel.x, target.vel.y, target.vel.z)

        pos = xyz_to_ecf_position(pos * ISimulator.MODEL_SCALE)
        vel = xyz_to_ecf_velocity(vel * ISimulator.MODEL_SCALE)

        self.model.SetTargetEcf(UtilsPy.Vector3D(pos.x, pos.y, pos.z), UtilsPy.Vector3D(vel.x, vel.y, vel.z))

    def set_speed(self, speed: float) -> None:
        """
        初始设置速度
        """
        pass

    def reset(self) -> None:
        """重置到初始状态"""
        super().reset()

        self.model = None

        self.ret = -1
        self.launched = -1
        self.target_id = None

    def init_model(self):
        self.model = Missile()
        self.set_position(self.entity_ext.entity.posEcf)

    def set_position(self, pos: Vector3d) -> None:
        """
        初始设置lla
        """

        if self.launched == 1:
            return

        self.entity_ext.entity.posEcf = pos

        missile_lla = xyz_to_lla_position(self.entity_ext.entity.posEcf * ISimulator.MODEL_SCALE)

        self.model.Init(self.simulator_sim_step / 1000, UtilsPy.Vector3D(missile_lla.x, missile_lla.y, missile_lla.z) , 20)
        self.model.Save(False, self.entity_ext.entity.id)

    def command_received(self, command: Command) -> None:
        """
        自定义指令接收
        :param command:
        :return:
        """
        if command.commandTypeId == SimmerCommandType.INTERCEPTOR_LAUNCH:
            self.launch(command.commandAttributes.targetId, command.commandAttributes.targetPosInfo.cPos,
                        command.commandAttributes.targetPosInfo.cVel)
        else:
            super().command_received(command)
