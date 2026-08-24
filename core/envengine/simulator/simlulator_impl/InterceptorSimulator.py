import logging
import math
from typing import Callable

from envengine.sdk.Util.UtilsPy import Vector3D, CoordinateHelper, State_py

from envengine.common import SimmerCommandType, DamageC, DamageCmd
from envengine.sdk.base_struct.Basic import Vector3d
from envengine.sdk.base_struct.Entity import EntityExt
from envengine.sdk.base_struct.Message import Command
from envengine.sdk.Util import UtilsPy
from envengine.simulator.decorator import Simulator
from envengine.simulator.interfaces import ISimulator
from envengine.simulator.models.SM6_1BMissileModel import Missile
from envengine.simulator.simulator_factory import SimulatorFactory


@Simulator.register("InterceptorS")
class InterceptorSimulator(ISimulator):
    """
    固定翼飞行器仿真器
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
            state: State_py = self.model.getState()

            pos = state.posEcf()
            vel = state.velEcf()
            lla = UtilsPy.CoordinateHelper.ecefToLla_py(pos)
            entity = self.entity_ext.entity
            if math.isnan(lla.x()):
                print(1)
            entity.lla.x, entity.lla.y, entity.lla.z = lla.x(), lla.y(), lla.z()
            entity.posEcf.x, entity.posEcf.y, entity.posEcf.z = pos.x(), pos.y(), pos.z()
            entity.velEcf.x, entity.velEcf.y, entity.velEcf.z = vel.x(), vel.y(), vel.z()
            entity.stage = state.stage()

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

                # 弹爆炸了，计算与周围仿真器的距离，如果距离小于100米，则通知仿真器进行爆炸
                damage_candidate_simulators: list[ISimulator] = self._simulator_factory.get_simulators_by_side(
                    1 if self.entity_ext.entity.sideId == 0 else 0)
                can_damage_simulators: list[ISimulator] = self._get_targets_within_self_range(
                    damage_candidate_simulators,
                    max_range=1000)
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

    def launch(self, target_id: int, target_pos_ecf: Vector3D, target_vel_ecf: Vector3D):
        """
        发射
        :param target_lla: 目标位置
        :return:
        """
        if self.launched != -1:
            return
        self.target_id = target_id

        target_lla = CoordinateHelper.ecefToLla_py(target_pos_ecf)
        # print(f"target_lla:{target_lla.x(), target_lla.y(), target_lla.z()}")
        # print(f"target_vel_ecf:{target_vel_ecf.x(), target_vel_ecf.y(), target_vel_ecf.z()}")

        # targetPosNue = CoordinateHelper.ecefToNuePosition_py(target_pos_ecf, self.entity_ext.entity.lla.x,
        #                                                      self.entity_ext.entity.lla.y)
        # theta_f = CoordinateHelper.getTheta_py(targetPosNue) * 57.3
        # psi_f = CoordinateHelper.getPsi_py(targetPosNue) * 57.3
        # self.model.SetTargetEcf(target_pos_ecf, target_vel_ecf)

        self.model.Launch(target_lla)
        self.leave_parent()
        self.launched = 1

    def set_target_ecf(self) -> None:
        """
        设置目标位置
        :return:
        """
        target: ISimulator = self._simulator_factory.get_simulator_by_id(self.target_id)

        if target.entity_ext.entity.survivePoints <= 0:
            return
        self.model.SetTargetEcf(Vector3D(
                target.entity_ext.entity.posEcf.x,
                target.entity_ext.entity.posEcf.y,
                target.entity_ext.entity.posEcf.z),
            Vector3D(
                target.entity_ext.entity.velEcf.x,
                target.entity_ext.entity.velEcf.y,
                target.entity_ext.entity.velEcf.z))

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

    def reset(self) -> None:
        """重置到初始状态"""
        super().reset()

        self.model = None

        self.ret = -1
        self.launched = -1
        self.target_id = None

    def init_model(self):
        self.model = Missile()
        missile_lla = UtilsPy.Vector3D(self.entity_ext.entity.lla.x, self.entity_ext.entity.lla.y,
                                       self.entity_ext.entity.lla.z + 0.1)

        self.model.Init(self.simulator_sim_step / 1000, missile_lla, 20)
        self.model.Save(False, self.entity_ext.entity.id)

    def command_received(self, command: Command) -> None:
        """
        自定义指令接收
        :param command:
        :return:
        """
        if command.commandTypeId == SimmerCommandType.INTERCEPTOR_LAUNCH:
            self.launch(command.commandAttributes.targetId, command.commandAttributes.targetPosLLA.cPos,
                        command.commandAttributes.targetPosLLA.cVel)
        else:
            super().command_received(command)
