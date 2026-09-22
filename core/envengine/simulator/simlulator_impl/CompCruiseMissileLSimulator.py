# -*-coding:utf-8 -*-
import json
import logging
import math
import os
import time
from copy import deepcopy
from typing import Callable

from envengine.common import SimmerCommandType, DamageC, DamageCmd
from envengine.sdk.Util.UtilsPy import Vector3D
from envengine.sdk.base_struct.Basic import Vector3d, DetectInfo
from envengine.sdk.base_struct.Entity import EntityExt
from envengine.sdk.base_struct.Message import Command
from envengine.sdk.Util import UtilsPy
from envengine.sdk.Util import State_py
from envengine.simulator.decorator import Simulator, timer_decorator
from envengine.simulator.interfaces import ISimulator
from envengine.simulator.models.ACMMissileModel import Missile
from envengine.simulator.simulator_factory import SimulatorFactory

logger = logging.getLogger(__name__)


@Simulator.register("CruiseMissileLS")
class CompCruiseMissileLSimulator(ISimulator):
    """
    无人机

    Attributes:
        launch: 发射时间
    """

    def __init__(self, entity_ext: EntityExt,
                 send_commands: Callable[[list[Command]], None],
                 send_events: Callable[[dict], None],
                 simulator_factory: SimulatorFactory = None):
        super().__init__(entity_ext, send_commands, send_events, simulator_factory)
        self.ret = -1
        self.launch = -1
        self.damage_point = 1
        # Training/evaluation diagnostic only.  This is never copied into an
        # entity observation or communication message.
        self.direct_9500_first_detection_step: dict[int, int] = {}
        self.direct_9500_first_distance_m: dict[int, float] = {}
        self.direct_9500_min_distance_m: dict[int, float] = {}
        self.direct_9500_min_alive_distance_m: dict[int, float] = {}
        self.direct_9500_detection_sample_count = 0
        self.direct_9500_last_detection_step = -1
        # Diagnostic only: distinguish route completion/timeout from an
        # external kill.  None of these fields enters an observation.
        self.search_termination_reason: str | None = None
        self.search_termination_step = -1
        self.search_destroyed_by_entity_id: int | None = None
        self.search_destroyed_by_entity_type: int | None = None
        self.search_commanded_target: dict[str, float] | None = None
        self.search_termination_distance_m: float | None = None
        monitor_entity_id = os.getenv("RED_SEARCH_MONITOR_ENTITY_ID")
        try:
            self.direct_9500_monitor_enabled = (
                monitor_entity_id is not None
                and int(monitor_entity_id) == int(entity_ext.entity.id)
            )
        except ValueError:
            self.direct_9500_monitor_enabled = False
        self.direct_9500_monitor_samples: list[dict] = []
        # Diagnostic-only sensor footprints for team coverage credit. These
        # samples contain only the L position and never objective coordinates.
        self.sensor_footprint_samples: list[dict] = []

    @property
    def simulator_sim_step(self) -> float:
        """
        仿真器内部步长（毫秒）
        :return: 内部步长（毫秒）
        """
        return 50

    def update(self) -> None:
        """
        执行仿真步进
        """
        if self.launch < 0:
            return

        if self.ret < 0 and (self.sim_time - self.launch <= 1800_000):
            # 更新探测信息
            if self.sim_time % 1000 == 0:
                self.execute_detection()

            self.ret = self.model.Update()
            state: State_py = self.model.getState()

            pos = state.posEcf()
            vel = state.velEcf()
            lla = UtilsPy.CoordinateHelper.ecefToLla_py(pos)
            entity = self.entity_ext.entity
            entity.lla.x, entity.lla.y, entity.lla.z = lla.x(), lla.y(), lla.z()
            entity.posEcf.x, entity.posEcf.y, entity.posEcf.z = pos.x(), pos.y(), pos.z()
            entity.velEcf.x, entity.velEcf.y, entity.velEcf.z = vel.x(), vel.y(), vel.z()
            entity.stage = state.stage()
        else:
            if self.entity_ext.entity.isVisible:
                if self.ret < 0:
                    self.search_termination_reason = "timeout"
                else:
                    target = self.search_commanded_target
                    if target is None:
                        self.search_termination_reason = "model_completion"
                    else:
                        entity = self.entity_ext.entity
                        mean_latitude = math.radians(
                            (float(entity.lla.y) + target["lat"]) / 2.0
                        )
                        east_m = (
                            target["lon"] - float(entity.lla.x)
                        ) * 111_320.0 * math.cos(mean_latitude)
                        north_m = (
                            target["lat"] - float(entity.lla.y)
                        ) * 110_570.0
                        self.search_termination_distance_m = math.hypot(
                            east_m, north_m
                        )
                        self.search_termination_reason = (
                            "waypoint_completion"
                            if self.search_termination_distance_m <= 5_000.0
                            else "model_completion_away_from_waypoint"
                        )
                self.search_termination_step = int(getattr(
                    self._simulator_factory,
                    "_event_step",
                    self.sim_time // 1000,
                ))
                if self.ret < 0:
                    logger.warning(f"无人机 '{self.entity_ext.entity.nameChn}' 飞行时间超时自爆")

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

    def send_detect_info(self):
        """
        发送探测信息
        """
        if self.entity_ext.entity.detectInfo:
            # 基于通信距离触发
            communication_candidate_simulators: list[ISimulator] = self._simulator_factory.get_simulators_by_side(
                self.entity_ext.entity.sideId)
            can_communication_simulators: list[ISimulator] = self._get_targets_within_self_range(
                communication_candidate_simulators,
                max_range=100 * 1000)

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

            # 指令转发
            self._send_commands(command_list)

    def execute_detection(self):
        """
        执行探测，低性能弹可以探测30KM范围内的无人船，
        在使用卫星期间，可以获取全部拦截弹
        :return:
        """

        """
        9500:  无人船
        24000: 拦截弹
        """

        ships: list[ISimulator] = self._simulator_factory.get_simulators_by_type(9500)

        logical_step = int(getattr(
            self._simulator_factory,
            "_event_step",
            self.sim_time // 1000,
        ))
        self.direct_9500_detection_sample_count += 1
        self.direct_9500_last_detection_step = logical_step
        sensor_entity = self.entity_ext.entity
        if bool(sensor_entity.isVisible and sensor_entity.survivePoints > 0):
            self.sensor_footprint_samples.append({
                "step": logical_step,
                "lon": float(sensor_entity.lla.x),
                "lat": float(sensor_entity.lla.y),
                "alt": float(sensor_entity.lla.z),
            })
        detected: list[tuple[ISimulator, bool]] = []
        monitor_targets: list[dict] = []
        for sim in ships:
            dx = (
                sim.entity_ext.entity.posEcf.x
                - self.entity_ext.entity.posEcf.x
            )
            dy = (
                sim.entity_ext.entity.posEcf.y
                - self.entity_ext.entity.posEcf.y
            )
            dz = (
                sim.entity_ext.entity.posEcf.z
                - self.entity_ext.entity.posEcf.z
            )
            distance_m = math.sqrt(dx * dx + dy * dy + dz * dz)
            target_id = int(sim.entity_ext.entity.id)
            self.direct_9500_first_distance_m.setdefault(
                target_id,
                distance_m,
            )
            self.direct_9500_min_distance_m[target_id] = min(
                distance_m,
                self.direct_9500_min_distance_m.get(target_id, math.inf),
            )
            target_alive = bool(
                sim.entity_ext.entity.isVisible
                and sim.entity_ext.entity.survivePoints > 0
            )
            if target_alive:
                self.direct_9500_min_alive_distance_m[target_id] = min(
                    distance_m,
                    self.direct_9500_min_alive_distance_m.get(
                        target_id, math.inf
                    ),
                )
            within_normal_range = self._is_geometrically_visible(
                sim.entity_ext.entity.posEcf,
                self.entity_ext.entity.posEcf,
                30 * 1000,
            )
            if self.direct_9500_monitor_enabled:
                monitor_targets.append({
                    "id": target_id,
                    "distance_m": distance_m,
                    "alive": target_alive,
                    "health": float(sim.entity_ext.entity.survivePoints),
                    "within_30km": bool(within_normal_range),
                })
            if target_alive and within_normal_range:
                detected.append((sim, False))

        if self.direct_9500_monitor_enabled:
            entity = self.entity_ext.entity
            self.direct_9500_monitor_samples.append({
                "step": logical_step,
                "sim_time": int(self.sim_time),
                "l_alive": bool(entity.isVisible and entity.survivePoints > 0),
                "l_health": float(entity.survivePoints),
                "l_lla": {
                    "lon": float(entity.lla.x),
                    "lat": float(entity.lla.y),
                    "alt": float(entity.lla.z),
                },
                "targets": monitor_targets,
                "direct_detection_ids": [
                    int(target.entity_ext.entity.id)
                    for target, _ in detected
                ],
            })

        if not detected:
            return
        for target, _ in detected:
            self.direct_9500_first_detection_step.setdefault(
                int(target.entity_ext.entity.id),
                logical_step,
            )
        # print("巡航弹探测到的目标：", detected)
        # 组装探测信息
        detect_info: dict[int, DetectInfo] = {
            target.entity_ext.entity.id: DetectInfo(
                detect_from=self.entity_ext.entity.id,
                time=int(self.sim_time),
                entity_id=target.entity_ext.entity.id,
                entity_type=target.entity_ext.entity.entityType,
                nameChn=target.entity_ext.entity.nameChn,
                lla=target.entity_ext.entity.lla,
                pos_ecf=target.entity_ext.entity.posEcf,
                vel_ecf=target.entity_ext.entity.velEcf,
                health_remaining=float(target.entity_ext.entity.survivePoints),
                health_max=float(target.entity_ext.entity.maxSurvivePoints),
                health_observed=target.entity_ext.entity.entityType in {9400, 9500, 9600},
                via_satellite=via_satellite,
            ) for target, via_satellite in detected}
        # 更新自身探测信息
        self.handel_detect_info(detect_info)

    def reset(self) -> None:
        """重置到初始状态"""
        super().reset()

        self.model = None

        self.ret = -1
        self.launch = -1
        self.direct_9500_first_detection_step.clear()
        self.direct_9500_first_distance_m.clear()
        self.direct_9500_min_distance_m.clear()
        self.direct_9500_min_alive_distance_m.clear()
        self.direct_9500_detection_sample_count = 0
        self.direct_9500_last_detection_step = -1
        self.direct_9500_monitor_samples.clear()
        self.sensor_footprint_samples.clear()
        self.search_termination_reason = None
        self.search_termination_step = -1
        self.search_destroyed_by_entity_id = None
        self.search_destroyed_by_entity_type = None
        self.search_commanded_target = None
        self.search_termination_distance_m = None

    def init_model(self):
        self.model = Missile()
        self.set_lla(self.entity_ext.entity.lla)

    def set_lla(self, lla: Vector3d) -> None:
        """
        初始设置lla
        """
        self.entity_ext.entity.lla = lla
        missile_lla = UtilsPy.Vector3D(self.entity_ext.entity.lla.x, self.entity_ext.entity.lla.y,
                                       self.entity_ext.entity.lla.z + 0.1)

        self.model.Init(self.simulator_sim_step / 1000, missile_lla, 5)
        self.model.SetDesiredHeight(10000)
        self.model.Save(False, self.entity_ext.entity.id)
        self.set_speed(300)

    def set_speed(self, speed: float) -> None:
        """
        初始设置速度
        """
        self.model.SetDesiredSpeed(300)

    def command_received(self, command: Command) -> None:
        """
        自定义指令接收
        :param command:
        :return:
        """
        if command.commandTypeId == SimmerCommandType.ATTACK_COMMANDER_START:
            pass
        elif command.commandTypeId == SimmerCommandType.MISSILE_LAUNCH:
            if self.launch >= 0:
                print(f"entity_id:{self.entity_ext.entity.id}, name:{self.entity_ext.entity.nameChn} 重复发射")
                return

            self.model.Launch(Vector3D(  # type:ignore
                command.commandAttributes["target"]["x"],
                command.commandAttributes["target"]["y"],
                command.commandAttributes["target"]["z"]
            ))
            self.search_commanded_target = {
                "lon": float(command.commandAttributes["target"]["x"]),
                "lat": float(command.commandAttributes["target"]["y"]),
            }

            self.launch = self.sim_time
        elif command.commandTypeId == SimmerCommandType.SET_DESIRED_ACC_Z:
            if command.commandAttributes["acc_z"] == 0:
                self.model.ClearDesiredAccZ()
            else:
                self.model.SetDesiredAccZ(command.commandAttributes["acc_z"] * 20 * 9.8)
        elif command.commandTypeId == SimmerCommandType.SET_DESIRED_VEL_X:
            self.model.SetDesiredSpeed(command.commandAttributes["vel_x"])
        elif command.commandTypeId == SimmerCommandType.CHANGE_MISSILE_TARGET:
            pos = UtilsPy.CoordinateHelper.llaToEcef_py(Vector3D(  # type:ignore
                command.commandAttributes["target"]["x"],
                command.commandAttributes["target"]["y"],
                command.commandAttributes["target"]["z"]
            ))
            self.model.SetTargetEcf(Vector3D(pos.x(), pos.y(), pos.z()), Vector3D(0, 0, 0))
            self.search_commanded_target = {
                "lon": float(command.commandAttributes["target"]["x"]),
                "lat": float(command.commandAttributes["target"]["y"]),
            }
        else:
            health_before = float(self.entity_ext.entity.survivePoints)
            super().command_received(command)
            if (
                command.commandTypeId in {
                    SimmerCommandType.DAMAGE,
                    SimmerCommandType.DESTROY,
                }
                and health_before > 0.0
                and float(self.entity_ext.entity.survivePoints) <= 0.0
                and self.search_termination_reason is None
            ):
                source_id = int(command.prevTriggerId)
                source = self._simulator_factory.get_simulator_by_id(source_id)
                self.search_termination_reason = "external_damage"
                self.search_termination_step = int(getattr(
                    self._simulator_factory,
                    "_event_step",
                    self.sim_time // 1000,
                ))
                self.search_destroyed_by_entity_id = source_id
                self.search_destroyed_by_entity_type = (
                    int(source.entity_ext.entity.entityType)
                    if source is not None else None
                )
