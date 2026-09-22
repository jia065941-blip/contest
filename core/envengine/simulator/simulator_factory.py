# -*-coding:utf-8 -*-
from copy import deepcopy
import json
import logging
import math
from typing import Optional

from envengine.sdk.base_struct.Message import Command
from envengine.sdk.base_struct.profile.profile import Profile
from envengine.sdk.base_struct.Entity.EntityExt import EntityExt
from .decorator.simulator_decorator import Simulator
from .interfaces.isimulator import ISimulator
from ..common import SimmerCommandType
from ..sdk.base_struct.Basic import Vector3d
from ..sdk.writer import write_event

logger = logging.getLogger(__name__)

class SimulatorFactory:
    """
    仿真器工厂
    负责创建、管理和查询仿真器实例

    Attributes:
        red_sat_max_use_count: 卫星最大使用次数
        red_sat_use_count: 卫星已使用次数
        sat_use_minutes: 卫星使用时间（分钟）
        _satellite_use_end_time: 卫星当本次使用结束时间
    """

    def __init__(self, profile: Profile):
        self._profile = profile
        self._simulators: dict[int, ISimulator] = {}  # 按ID索引
        self._simulators_by_type: dict[int, list[ISimulator]] = {}  # 按类型索引
        self._simulators_by_side: dict[int, list[ISimulator]] = {}  # 按作战方索引
        self._command_queue: list[Command] = []  # 存储待处理的指令
        self.sim_step = profile.imagineProfile.simStep  # 引擎仿真步长
        self.sim_time = 0  # 当前仿真时间

        # 初始化所有仿真器
        self._initialize_simulators()

        # 当前仿真轮次
        self.current_round = 0

        # 目标命中关系，记录击中每个目标的飞行器信息
        self.target_hit_relation:dict[int, list] = {}

        # 卫星使用情况
        self.red_sat_use_count = 0
        self._satellite_use_end_time = 0
        self._satellite_use_count_by_entity: dict[int, int] = {}
        self._satellite_use_end_time_by_entity: dict[int, float] = {}
        if self.profile and self.profile.imagineProfile:
            self.red_sat_max_use_count = max(0, self.profile.imagineProfile.satelliteMaxUseCount)
            self.sat_use_minutes = max(0, self.profile.imagineProfile.satelliteUseMinutes)

        # 因果事件账本在回合内持久保留，消费游标仅影响增量读取。
        self._causal_event_ledger: list[dict] = []
        self._causal_event_sequence = 0
        self._causal_event_consume_cursor = 0
        self._event_step = 0
        self._event_sim_time = float(profile.imagineProfile.simTime)

    @property
    def profile(self)->Profile:
        """当前想定"""
        return self._profile

    @property
    def command_queue(self):
        return self._command_queue

    @property
    def causal_event_ledger(self) -> tuple[dict, ...]:
        """返回当前回合的完整因果事件账本快照。"""
        return tuple(deepcopy(self._causal_event_ledger))

    def consume_causal_events(self) -> tuple[dict, ...]:
        """返回上次消费后新增的事件，并推进本消费者游标。"""
        start = self._causal_event_consume_cursor
        events = tuple(deepcopy(self._causal_event_ledger[start:]))
        self._causal_event_consume_cursor = len(self._causal_event_ledger)
        return events

    def set_event_context(self, *, step: int, sim_time: float) -> None:
        """设置随后分发命令所属的外层仿真步和逻辑时间。"""
        self._event_step = int(step)
        self._event_sim_time = float(sim_time)

    def _append_causal_event(self, event_type: str, **payload) -> str:
        self._causal_event_sequence += 1
        event_id = f"{self.current_round}:{self._causal_event_sequence}"
        event = {
            "event_id": event_id,
            "sequence": self._causal_event_sequence,
            "round": self.current_round,
            "step": self._event_step,
            "sim_time": self._event_sim_time,
            "event_type": event_type,
        }
        event.update(payload)
        self._causal_event_ledger.append(event)
        return event_id

    @staticmethod
    def _vector_snapshot(vector) -> dict[str, float] | None:
        if vector is None:
            return None

        def component(name: str) -> float:
            value = getattr(vector, name, 0.0)
            return float(value() if callable(value) else value)

        return {
            "x": component("x"),
            "y": component("y"),
            "z": component("z"),
        }

    @staticmethod
    def _command_attribute(command: Command, name: str):
        attributes = command.commandAttributes
        if isinstance(attributes, dict):
            return attributes.get(name)
        return getattr(attributes, name, None)

    def _record_direct_hit(
        self,
        *,
        attacking_simulator: ISimulator,
        target_simulator: ISimulator,
        actual_damage: float,
        health_before: float,
        health_after: float,
    ) -> str | None:
        if actual_damage <= 0.0:
            return None

        attacker = attacking_simulator.entity_ext.entity
        target = target_simulator.entity_ext.entity
        satellite_hit_rate_effect_entity_id = None
        if (
            int(attacker.entityType) == 21000
            and int(target.entityType) in (9400, 9600)
            and self.is_using_satellite(int(attacker.id))
        ):
            satellite_hit_rate_effect_entity_id = int(attacker.id)
        return self._append_causal_event(
            "direct_hit",
            attacking_entity_id=int(attacker.id),
            attacking_entity_type=int(attacker.entityType),
            target_entity_id=int(target.id),
            target_entity_type=int(target.entityType),
            actual_damage=float(actual_damage),
            target_health_before=float(health_before),
            target_health_after=float(health_after),
            source_sim_time=float(
                getattr(attacking_simulator, "sim_time", self._event_sim_time)
            ),
            source_position=self._vector_snapshot(attacker.posEcf),
            source_velocity=self._vector_snapshot(attacker.velEcf),
            satellite_hit_rate_effect_entity_id=(
                satellite_hit_rate_effect_entity_id
            ),
        )

    def record_satellite_request_accepted(
        self,
        simulator: ISimulator,
    ) -> str:
        """记录一次已成功生效的卫星请求。"""
        entity = simulator.entity_ext.entity
        return self._append_causal_event(
            "satellite_request_accepted",
            entity_id=int(entity.id),
            satellite_use_count=int(self.satellite_use_count(int(entity.id))),
            satellite_use_end_time=float(
                self.satellite_use_end_time(int(entity.id))
            ),
        )

    def _deliver_interceptor_launch(
        self,
        interceptor_simulator: ISimulator,
        command: Command,
    ) -> None:
        interceptor = interceptor_simulator.entity_ext.entity
        launched_before = getattr(interceptor_simulator, "launched", None)
        parent_before = getattr(interceptor, "parentId", -1)

        interceptor_simulator.command_received(command)

        launched_after = getattr(interceptor_simulator, "launched", None)
        if launched_before != -1 or launched_after == -1:
            return

        intercepted_id = self._command_attribute(command, "targetId")
        if intercepted_id is None:
            intercepted_id = getattr(interceptor_simulator, "target_id", None)
        if intercepted_id is None:
            return

        intercepted_id = int(intercepted_id)
        intercepted_simulator = self.get_simulator_by_id(intercepted_id)
        intercepted_type = None
        if intercepted_simulator is not None:
            intercepted_type = int(
                intercepted_simulator.entity_ext.entity.entityType
            )

        self._append_causal_event(
            "interceptor_launch",
            interceptor_entity_id=int(interceptor.id),
            interceptor_entity_type=int(interceptor.entityType),
            intercepted_red_entity_id=intercepted_id,
            intercepted_red_entity_type=intercepted_type,
            attacking_entity_id=intercepted_id,
            target_entity_id=intercepted_id,
            defending_parent_entity_id=int(parent_before),
            actual_damage=0.0,
            source_sim_time=float(
                getattr(interceptor_simulator, "sim_time", self._event_sim_time)
            ),
        )

    def _initialize_simulators(self):
        """初始化所有仿真器"""
        for entity_ext in self._profile.imagineProfile.entityList:
            # 创建仿真器实例（可以根据entityType创建不同类型的仿真器）
            entity_type: int = entity_ext.entity.entityType
            class_name: str = self._get_simulator_class_name_by_entity_type(entity_type)
            try:
                simulator = self._create_simulator(class_name, entity_ext)
                # 注册到工厂
                self._register_simulator(simulator)
            except Exception as e:
                logging.error(f"[仿真器工厂] 创建仿真器 {class_name} 失败: {e}")

        logging.info(
            f"[仿真器工厂] 初始化完成: 共{len(self._simulators)}个实体, "
            f"分别是{[i.entity_ext.entity.nameChn for i in self._simulators.values()]}"
        )

    def _get_simulator_class_name_by_entity_type(self, entity_type: int) -> str:
        """
        根据实体类型获取对应的仿真器类名
        :param entity_type: 实体类型
        :return: 仿真器类名
        """
        environment_profile: Profile.environmentProfile = self._profile.environmentProfile
        dynamic_library_configs: list[Profile.environmentProfile.dynamicLibraryConfigs] = environment_profile.dynamicLibraryConfigs
        for dynamic_library_config in dynamic_library_configs:
            if dynamic_library_config.entityType == entity_type:
                class_name = dynamic_library_config.dynamicLibraryPath.rsplit("/")[-1].rsplit(".", 1)[0].rsplit("lib")[
                    -1]
                return class_name

    def _create_simulator(self, class_name: str, entity_ext: EntityExt) -> ISimulator:
        """
        根据实体类型创建对应的仿真器
        :param entity_ext: 实体扩展信息
        :return: 仿真器实例
        """
        return Simulator.create(class_name, entity_ext, self._send_command_callback, self._send_events_callback,
                                simulator_factory=self)

    def _register_simulator(self, simulator: ISimulator):
        """注册仿真器到索引"""
        id = simulator.entity_ext.entity.id
        entity_type = simulator.entity_ext.entity.entityType
        side_id = simulator.entity_ext.entity.sideId

        # 按ID存储
        self._simulators[id] = simulator

        # 按类型分组
        if entity_type not in self._simulators_by_type:
            self._simulators_by_type[entity_type] = []
        self._simulators_by_type[entity_type].append(simulator)

        # 按作战方分组
        if side_id not in self._simulators_by_side:
            self._simulators_by_side[side_id] = []
        self._simulators_by_side[side_id].append(simulator)

    def _send_command_callback(self, commands: list[Command]) -> None:
        """
        仿真器发送指令的回调，用于将指令加入待处理的指令队列
        :param commands:
        :return:
        """
        if commands:
            self._command_queue.extend(commands)

    def process_commands(self):
        """
        引擎调用： 分发并执行队列中的所有指令
        :return:
        """
        if not self._command_queue:
            return
        for command in self._command_queue:
            target_sim = self.get_simulator_by_id(command.executorId)
            if target_sim:
                if command.commandTypeId == SimmerCommandType.INTERCEPTOR_LAUNCH:
                    self._deliver_interceptor_launch(target_sim, command)
                    continue
                if command.commandTypeId == SimmerCommandType.DAMAGE:
                    command = self.process_hit(command)

                    prev_trigger = self.get_simulator_by_id(command.prevTriggerId)
                    if prev_trigger:
                        health_before = float(
                            target_sim.entity_ext.entity.survivePoints
                        )
                        target_sim.command_received(command)
                        health_after = float(
                            target_sim.entity_ext.entity.survivePoints
                        )
                        damage_point = max(0.0, health_before - health_after)
                        event_id = self._record_direct_hit(
                            attacking_simulator=prev_trigger,
                            target_simulator=target_sim,
                            actual_damage=damage_point,
                            health_before=health_before,
                            health_after=health_after,
                        )
                        hit_event = {
                            "id":prev_trigger.entity_ext.entity.id,
                            "entity_type":prev_trigger.entity_ext.entity.entityType,
                            "sim_time":prev_trigger.sim_time,
                            "damage_point":damage_point,
                            "velEcf":prev_trigger.entity_ext.entity.velEcf,
                            "posEcf":prev_trigger.entity_ext.entity.posEcf,
                        }
                        if event_id is not None:
                            hit_event["event_id"] = event_id
                        self.target_hit_relation.setdefault(
                            command.executorId, []
                        ).append(hit_event)
                        continue
                target_sim.command_received(command)
                # if command.commandTypeId == SimmerCommandType.DAMAGE:
                #     print(command)
        self._command_queue.clear()

    def process_hit(self, command):
        """
        命中率映射表：后序需要提取成配置文件，新增不同entityType模型不同模型
        :return:
        """

        """
        21000 高性能飞行器
        21001 低性能飞行器
        21002 无人机
        
        9400 目标
        9500 无人船
        9600 拦截阵地
        """

        # 红方毁伤数值表
        damage_point_table = {
            21000:{
                9400:20,
                9600:20,
                9500:20
            },
            21001:{
                9400:5,
                9600:5,
                9500:5
            },
            21002:{
                9400:0,
                9600:0,
                9500:1
            }
        }

        # 红方命中率表
        hit_rate_table = {
            # 高性能飞行器
            21000: {
                9400: 0.8,
                9600: 0.6,
                9500: 0
            },
            # 低性能飞行器
            21001: {
                9400: 0.8,
                9600: 0.6,
                9500: 0
            },
            # 无人机
            21002: {
                9400: 0.05,
                9600: 0.05,
                9500: 0.8
            }
        }

        # 伤害来源
        prev_trigger_id = command.prevTriggerId
        prev_simulator = self.get_simulator_by_id(prev_trigger_id)
        if not prev_simulator:
            return command

        prev_simulator_type = prev_simulator.entity_ext.entity.entityType

        # 被击中的对象
        executor_id = command.executorId
        executor_simulator = self.get_simulator_by_id(executor_id)
        if not executor_simulator:
            return command
        executor_simulator_type = executor_simulator.entity_ext.entity.entityType

        # 获取命中率
        if prev_simulator_type not in hit_rate_table:
            return command
        if executor_simulator_type not in hit_rate_table[prev_simulator_type]:
            return command
        hit_rate = hit_rate_table[prev_simulator_type][executor_simulator_type]
        hit_rate = self.recalculate_rate(prev_simulator, executor_simulator, hit_rate)
        # damage_point = json.loads(command.commandAttributes)["cmd"]["damagePoint"]
        damage_point = damage_point_table[prev_simulator_type][executor_simulator_type]
        command.commandAttributes = json.dumps({"cmd": {"damagePoint": damage_point * hit_rate}})
        return command

    def recalculate_rate(self, prev_simulator: ISimulator | None, executor_simulator: ISimulator | None, hit_rate:float):
        """
        命中率调整
        :param prev_simulator:      进攻方
        :param executor_simulator:  被集中目标
        :param hit_rate:            原始命中率
        :return:
        """

        if not executor_simulator or not prev_simulator:
            return hit_rate

        # 使用卫星期间，高性能飞行器对目标和阵地的命中率提升至 100%
        if (prev_simulator.entity_ext.entity.entityType == 21000
                and executor_simulator.entity_ext.entity.entityType in [9400, 9600]
                and self.is_using_satellite(
                    int(prev_simulator.entity_ext.entity.id)
                )):
            return 1

        # 命中率提升条件
        hit_rate = self.hit_increase(prev_simulator, executor_simulator, hit_rate)
        hit_rate = self.hit_decrease(prev_simulator, executor_simulator, hit_rate)

        return hit_rate

    def hit_increase(self, prev_simulator: ISimulator, executor_simulator: ISimulator, hit_rate:float)->float:
        """
        命中率提升

        当任意两枚高/低性能飞行器在一定时间内命中同一目标，且入射夹角大于30°时，
        命中率最大提升20%，按照时间间隔线性递减，时间间隔0为20%

        :param prev_simulator: 进攻方
        :param executor_simulator: 被击中目标
        :param hit_rate: 原始命中率
        :return: 新的命中率
        """

        if hit_rate == 0:
            return hit_rate

        if prev_simulator.entity_ext.entity.entityType not in [21000, 21001]:
            return hit_rate

        if executor_simulator.entity_ext.entity.entityType != 9400:
            return hit_rate

        if executor_simulator.entity_ext.entity.id not in self.target_hit_relation:
            return hit_rate

        # 时间间隔
        time_interval = self.profile.imagineProfile.missileRateIncreaseTimeIntervalMinutes *60*1000
        min_angle = self.profile.imagineProfile.missileRateIncreaseMinAngle # 最小入射夹角（度）
        max_increase = self.profile.imagineProfile.missileRateIncreaseMaxValue # 最大提升比例

        last_hit_time = 0
        for item in self.target_hit_relation[executor_simulator.entity_ext.entity.id]:
            if (item["entity_type"] in [21000, 21001] and
                    prev_simulator.sim_time - item["sim_time"] <= time_interval and
                    self.get_vector_angle(prev_simulator.entity_ext.entity.velEcf, item["velEcf"]) > min_angle):
                # 满足提升条件，记录时间间隔最小的一次命中
                last_hit_time = max(last_hit_time, item["sim_time"])

        if last_hit_time == 0:
            return hit_rate

        # 计算命中率提升比例
        increase_rate = max_increase * (1-(prev_simulator.sim_time - last_hit_time) / time_interval)
        return hit_rate * (1+increase_rate)

    def hit_decrease(self, prev_simulator: ISimulator, executor_simulator: ISimulator, hit_rate:float)->float:
        """
        命中率降低

        当任意两枚高/低性能飞行器在一定时间内命中同一目标，且入射夹角小于10°时，
        命中率最大降低40%，按照时间间隔线性递增，时间间隔0为40%

        :param prev_simulator: 进攻方
        :param executor_simulator: 被击中目标
        :param hit_rate: 原始命中率
        :return: 新的命中率
        """

        if hit_rate == 0:
            return hit_rate

        if prev_simulator.entity_ext.entity.entityType not in [21000, 21001]:
            return hit_rate

        if executor_simulator.entity_ext.entity.entityType != 9400:
            return hit_rate

        if executor_simulator.entity_ext.entity.id not in self.target_hit_relation:
            return hit_rate

        # 时间间隔
        time_interval = self.profile.imagineProfile.missileRateDecreaseTimeIntervalMinutes *60*1000
        max_angle = self.profile.imagineProfile.missileRateDecreaseMaxAngle # 最大入射夹角（度）
        max_decrease = self.profile.imagineProfile.missileRateDecreaseMaxValue

        last_hit_time = 0
        for item in self.target_hit_relation[executor_simulator.entity_ext.entity.id]:
            if (item["entity_type"] in [21000, 21001] and
                    prev_simulator.sim_time - item["sim_time"] <= time_interval and
                    self.get_vector_angle(prev_simulator.entity_ext.entity.velEcf, executor_simulator.entity_ext.entity.velEcf) < max_angle):
                # 满足条件，记录时间间隔最小的一次命中
                last_hit_time = max(last_hit_time, item["sim_time"])

        if last_hit_time == 0:
            return hit_rate

        # 计算命中率提升比例
        decrease_rate = max_decrease * (1-(prev_simulator.sim_time - last_hit_time) / time_interval)
        return hit_rate * (1-decrease_rate)

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

    def process_ai_commands(self, ai_commands: list[Command]):
        """
        引擎调用： 收集并分发执行AI的所有指令
        :return:
        """
        for ai_command in ai_commands:
            target_sim = self.get_simulator_by_id(ai_command.executorId)
            if not target_sim:
                continue

            if ai_command.commandTypeId == SimmerCommandType.EXECUTE_SATELLITE_DETECTION:
                # 每个实体拥有独立的申请次数和生效窗口；不设并发连接上限。
                self._process_missile_use_satellite(ai_command, target_sim)
            elif ai_command.commandTypeId == SimmerCommandType.INTERCEPTOR_LAUNCH:
                self._deliver_interceptor_launch(target_sim, ai_command)
            else:
                target_sim.command_received(ai_command)

    def _process_missile_use_satellite(self, command: Command, target_simulator: ISimulator):
        """
        处理使用卫星的指令
        :param command:
        :return:
        """

        entity_id = int(target_simulator.entity_ext.entity.id)
        used_count = self.satellite_use_count(entity_id)
        if used_count >= self.red_sat_max_use_count:
            logger.warning(f"使用卫星次数已经超出最大使用次数，来自【{target_simulator.entity_ext.entity.nameChn}】的卫星使用指令无效")
            return
        if self.is_using_satellite(entity_id):
            logger.warning(
                f"entity_id:{entity_id} 卫星仍在生效，重复申请指令无效"
            )
            return

        used_count += 1
        end_time = self.sim_time + self.sat_use_minutes * 60 * 1000
        self._satellite_use_count_by_entity[entity_id] = used_count
        self._satellite_use_end_time_by_entity[entity_id] = end_time
        self.red_sat_use_count = sum(self._satellite_use_count_by_entity.values())
        self._satellite_use_end_time = max(
            self._satellite_use_end_time_by_entity.values(), default=0
        )
        logger.info(
            f"entity_id:{entity_id}, name:{target_simulator.entity_ext.entity.nameChn} "
            f"使用卫星，该实体剩余次数：{self.red_sat_max_use_count - used_count}"
        )
        self.record_satellite_request_accepted(target_simulator)

    def satellite_use_count(self, entity_id: int | None = None) -> int:
        """返回实体级卫星已用次数；无 entity_id 时返回兼容的总数。"""

        if entity_id is None:
            return int(sum(self._satellite_use_count_by_entity.values()))
        return int(self._satellite_use_count_by_entity.get(int(entity_id), 0))

    def satellite_use_end_time(self, entity_id: int | None = None) -> float:
        """返回实体级卫星生效截止时间。"""

        if entity_id is None:
            return float(max(
                self._satellite_use_end_time_by_entity.values(), default=0.0
            ))
        return float(
            self._satellite_use_end_time_by_entity.get(int(entity_id), 0.0)
        )

    def satellite_remaining_uses(self, entity_id: int) -> int:
        """返回实体可用的卫星申请次数。"""

        return max(
            0,
            int(self.red_sat_max_use_count) - self.satellite_use_count(entity_id),
        )

    def is_using_satellite(self, entity_id: int | None = None) -> bool:
        """查询实体级或兼容的全局卫星生效状态。"""

        return self.sim_time < self.satellite_use_end_time(entity_id)

    def _send_events_callback(self, event: dict):
        """
        即时发送
        :param event: 事件
        :return:
        """
        write_event(event, str(self.current_round) + ".json")
        # logging.info(f"[仿真器工厂] 事件显示: {event}")

    def get_simulator_by_id(self, id: int) -> Optional[ISimulator]:
        """
        按ID获取仿真器
        :param id: 实体ID
        :return: 仿真器实例，不存在返回None
        """
        return self._simulators.get(id)

    def get_simulators_by_type(self, entity_type: int) -> list[ISimulator]:
        """
        按类型获取仿真器列表
        :param entity_type: 实体类型
        :return: 仿真器列表
        """
        return self._simulators_by_type.get(entity_type, [])

    def get_lived_simulators_by_type(self, entity_type: int) -> list[ISimulator]:
        """
        按类型获取有生命的仿真器列表
        :param entity_type: 实体类型
        :return: 仿真器列表
        """
        simulators: list[ISimulator] = self._simulators_by_type.get(entity_type, [])

        return [simulator for simulator in simulators if simulator.entity_ext.entity.survivePoints > 0]

    def get_simulators_by_side(self, side_id: int) -> list[ISimulator]:
        """
        按作战方获取仿真器列表
        :param side_id: 作战方ID
        :return: 仿真器列表
        """
        return self._simulators_by_side.get(side_id, [])

    def get_all_simulators(self) -> list[ISimulator]:
        """获取所有仿真器"""
        return list(self._simulators.values())

    def get_all_lived_simulators(self) -> list[ISimulator]:
        """获取所有有生命的仿真器"""
        simulators: list[ISimulator] = list(self._simulators.values())
        return [simulator for simulator in simulators if simulator.entity_ext.entity.survivePoints > 0]

    def get_simulator_count(self) -> int:
        """获取仿真器总数"""
        return len(self._simulators)

    def get_profile_entity_count_by_type(self, entity_type: int):
        """获取想定中的某种类型实体个数"""
        return sum(1 for e in self.profile.imagineProfile.entityList if e.entity.entityType == entity_type)

    def update_sim_time(self, sim_time:float):
        self.sim_time = sim_time
        for simulator in self._simulators.values():
            simulator.sim_time = sim_time

    def reset_all(self):
        """重置所有仿真器"""

        # 必须先重置所有仿真器，再统一进行初始化
        for simulator in self._simulators.values():
            simulator.reset()

        for simulator in self._simulators.values():
            simulator.init_model()

        self._command_queue.clear()
        self.current_round += 1
        self.target_hit_relation.clear()
        self._satellite_use_end_time = 0
        self.red_sat_use_count = 0
        getattr(self, "_satellite_use_count_by_entity", {}).clear()
        getattr(self, "_satellite_use_end_time_by_entity", {}).clear()
        self._causal_event_ledger.clear()
        self._causal_event_sequence = 0
        self._causal_event_consume_cursor = 0
        self._event_step = 0
        self._event_sim_time = float(self.profile.imagineProfile.simTime)
        logging.info("[仿真器工厂] 所有仿真器已重置")

    def remove_simulator(self, id: int) -> bool:
        """
        移除仿真器
        :param id: 实体ID
        :return: 是否成功移除
        """
        if id not in self._simulators:
            return False

        simulator = self._simulators.pop(id)

        # 从类型索引中移除
        if simulator.entity_ext.entity.entityType in self._simulators_by_type:
            self._simulators_by_type[simulator.entity_ext.entity.entityType] = [
                s for s in self._simulators_by_type[simulator.entity_ext.entity.entityType]
                if s.entity_ext.entity.id != id
            ]

        # 从作战方索引中移除
        if simulator.entity_ext.entity.sideId in self._simulators_by_side:
            self._simulators_by_side[simulator.entity_ext.entity.sideId] = [
                s for s in self._simulators_by_side[simulator.entity_ext.entity.sideId]
                if s.entity_ext.entity.id != id
            ]

        return True

    def modify_simulator_position(self, id: int, lla: dict) -> bool:
        """
        修改仿真器位置
        :param id: 模型id
        :param lla: 新的模型位置
        :return:
        """
        if id not in self._simulators:
            return False
        self._simulators[id].set_lla(Vector3d(lla["x"], lla["y"], lla["z"]))

    def modify_simulator_speed(self, id: int, speed: float) -> bool:
        """
        修改仿真器速度
        :param id: 模型id
        :param speed: 新的速度
        :return:
        """
        if id not in self._simulators:
            return False
        self._simulators[id].set_speed(speed)
