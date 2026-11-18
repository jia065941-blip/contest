# -*- coding: utf-8 -*-#
import logging
from collections.abc import Mapping
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from numbers import Integral
from typing import Dict, Optional, List
from uuid import uuid4

import numpy as np

from envengine.action_validation import IllegalActionError
from envengine.common.BaseStruct import Vector3d
from envengine.sdk.base_struct.Entity import Entity
from envengine.sdk.writer import write_ai_action
from scenarios import DeploymentConstraints
from user_agents.base import BaseAgent
from envengine.environment.scene_adapter import public_detections

logger = logging.getLogger(__name__)


class RedAgentGroup:
    """本局红方实例集合，由运行入口统一释放各实例。"""

    def __init__(self, agents):
        self.agents = tuple(agents)
        self._closed = False

    def reset(self) -> None:
        # 逐个清理实体的局部状态，示例内部资源由实例自身管理。
        if self._closed:
            raise RuntimeError("红方策略组已关闭")
        for agent in self.agents:
            agent.reset()

    def close(self) -> None:
        # 各实体释放一次，任一失败仍继续关闭其余实例。
        if self._closed:
            return
        self._closed = True
        with ExitStack() as stack:
            for agent in reversed(self.agents):
                stack.callback(agent.close)


class AgentManager:
    """接入初始布局、注册红方实例，分发单机观测并收集动作。"""

    def __init__(self, engine=None, scenario_config=None, *, record_logs: bool = False):
        self.engine = engine
        self._record_logs = record_logs
        self._deployment: DeploymentConstraints | None = None
        if scenario_config is not None:
            deployment = scenario_config.deployment
            self._deployment = DeploymentConstraints(
                coordinates=deployment.coordinates, launch_altitude=deployment.launch_altitude,
            )
        elif engine is not None:
            area = engine.profile.imagineProfile.redArea
            self._deployment = DeploymentConstraints(coordinates=area.coordinates)
        self._episode_started = False
        self._deployment_applied = False
        self._run_id: str | None = None
        self._agents: Dict[int, BaseAgent] = {}  # 注册的实体编号 -> agent
        self._red_group: RedAgentGroup | None = None

    def register_agent(self, agent: BaseAgent):
        """
        注册智能体
        :param agent: 智能体实例
        """
        entity_id = agent.get_entity_id()
        if entity_id in self._agents:
            raise ValueError(f"实体重复注册：{entity_id}")

        self._agents[entity_id] = agent

        logger.info(f"[AgentManager] 注册实体 {entity_id} 的智能体")

    def register_red_agents(self, agent_class: type[BaseAgent], entity_ids, init_observations: dict[int, dict]) -> RedAgentGroup:
        """按受控清单创建具体类；构造成功后一次注册本局全部实例。"""
        if self._red_group is not None:
            raise RuntimeError("本环境已经注册红方策略")
        entity_ids = tuple(entity_ids)
        if len(set(entity_ids)) != len(entity_ids) or set(entity_ids).intersection(self._agents):
            raise ValueError("红方受控清单包含重复的实体")

        # 逐实体传递已裁剪的初始化数据，框架只管理各实例的生命周期。
        agents = []
        try:
            for entity_id in entity_ids:
                agent = agent_class(entity_id, deepcopy(init_observations[entity_id]))
                agents.append(agent)
                if agent.get_entity_id() != entity_id:
                    raise ValueError("红方实例必须保留注册时分配的实体编号")
        except BaseException:
            RedAgentGroup(agents).close()
            raise

        group = RedAgentGroup(agents)
        try:
            for agent in agents:
                self.register_agent(agent)
        except BaseException:
            # 注册失败时撤回本批次记录，已有其他实例继续保留。
            for entity_id in entity_ids:
                self._agents.pop(entity_id, None)
            group.close()
            raise
        self._red_group = group
        return group

    def _require_engine(self):
        if self.engine is None:
            raise RuntimeError("AgentManager 尚未绑定 Engine")
        return self.engine

    def _require_deployment_phase(self):
        engine = self._require_engine()
        if not self._episode_started or engine.current_step != 0 or self._deployment_applied:
            raise RuntimeError("每局只能在 reset 后、初始部署完成前执行部署")
        return engine

    def get_deployment_observation(self, *, seed=None) -> dict:
        """为独立部署提供己方初态、公开目标、部署范围和种子。"""
        engine = self._require_deployment_phase()
        deployment = self._deployment
        if len(deployment.coordinates) != 1:
            raise ValueError("Exactly one deployment polygon is required")
        coordinates: list[list[float]] = [list(point) for point in deployment.coordinates[0]]
        entities = {}
        for simulator in engine.simulator_factory.get_all_simulators():
            entity = simulator.entity_ext.entity
            if entity.entityType not in (21000, 21001, 21002):
                continue
            entities[entity.id] = {
                "id": entity.id,
                "type": entity.entityType
            }
        return {
            "step": engine.current_step,
            "entities": entities,
            "public_targets": self._public_targets_observation(),
            "coordinates": coordinates,
            "seed": seed,
        }

    def _public_targets_observation(self) -> dict:
        """部署与策略初始化共用母巢、哨卫巢的公开目录格式。"""
        targets = {}
        for simulator in self._require_engine().simulator_factory.get_all_simulators():
            entity = simulator.entity_ext.entity
            if entity.entityType in (9400, 9600):
                targets[entity.id] = {
                    "nameChn": entity.nameChn,
                    "pos": {"x": entity.posEcf.x, "y": entity.posEcf.y, "z": entity.posEcf.z},
                    "health": entity.survivePoints,
                    "type": entity.entityType,
                    "side": entity.sideId,
                }
        return targets

    def apply_deployment(self, plan: Mapping) -> None:
        """整批校验二维布局，保留各实体原始高度设置位置，再统一启动。"""
        engine = self._require_deployment_phase()
        factory = engine.simulator_factory
        if not isinstance(plan, Mapping):
            raise ValueError("部署方案必须是实体编号到 (x, y) 的映射")
        if any(not isinstance(entity_id, Integral) or isinstance(entity_id, bool) for entity_id in plan):
            raise ValueError("部署方案必须使用整数实体编号")
        expected_ids = {
            simulator.entity_ext.entity.id
            for entity_type in (21000, 21001, 21002)
            for simulator in factory.get_simulators_by_type(entity_type)
        }
        if set(plan) != expected_ids:
            raise ValueError("部署方案必须准确覆盖全部红方受控实体")

        # 先检查整个映射及原始高度，任何非法数据都在位置修改与模型启动之前报错。
        actions = []
        for entity_id, position in plan.items():
            try:
                coordinates = np.asarray(position, dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError(f"实体 {entity_id} 的部署位置必须是数值坐标") from error
            if coordinates.shape != (2,) or not np.isfinite(coordinates).all():
                raise ValueError("部署位置必须是有限数值 (x, y)，高度沿用实体初态")
            x, y = map(float, coordinates)
            entity_id = int(entity_id)
            altitude = self._validate_deployment_position(entity_id, x, y)
            actions.append(
                {
                "executor_id": entity_id,
                "pos": Vector3d(x=x, y=y, z=altitude),
            })

        for action in actions:
            if self._record_logs:
                write_ai_action(action, str(factory.current_round) + ".json")
            factory.modify_simulator_position(action["executor_id"], action["pos"])
        factory.start_red_missions()
        self._deployment_applied = True

    def _validate_deployment_position(self, entity_id: int, x: float, y: float) -> float:
        """检查统一区域与原始高度，返回应用二维布局时使用的实体高度。"""
        simulator = self.engine.simulator_factory.get_simulator_by_id(entity_id)
        if simulator is None:
            raise ValueError(f"部署动作引用了不存在的实体 {entity_id}")
        if self._deployment is None:
            raise ValueError(f"不存在部署方案")

        entity_type = simulator.entity_ext.entity.entityType
        if entity_type not in (21000, 21001, 21002):
            raise ValueError(f"实体 {entity_id} 的类型 {entity_type} 不允许红方部署")
        regions = self._deployment.coordinates
        if len(regions) != 1 or not self._point_in_polygon(x, y, regions[0]):
            raise ValueError(f"实体 {entity_id} 的部署位置 ({x}, {y}) 超出允许区域")
        # 高度来自引擎中的实体，而非部署提交；先验证后才开始应用整批位置。
        try:
            altitude = float(simulator.entity_ext.entity.posEcf.z)
        except (TypeError, ValueError) as error:
            raise ValueError(f"实体 {entity_id} 的原始部署高度必须是有限数值") from error
        if not np.isfinite(altitude):
            raise ValueError(f"实体 {entity_id} 的原始部署高度必须是有限数值")
        # expected_altitude = self._deployment.launch_altitude
        # if expected_altitude is not None and abs(altitude - expected_altitude) > 1e-6:
        #     raise ValueError(f"实体 {entity_id} 的部署高度 {altitude} 不符合要求 {expected_altitude}")
        return altitude

    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon) -> bool:
        if len(polygon) < 3:
            return False
        inside = False
        previous = polygon[-1]
        for current in polygon:
            x1, y1 = float(previous[0]), float(previous[1])
            x2, y2 = float(current[0]), float(current[1])
            cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
            if (
                abs(cross) <= 1e-12
                and min(x1, x2) - 1e-12 <= x <= max(x1, x2) + 1e-12
                and min(y1, y2) - 1e-12 <= y <= max(y1, y2) + 1e-12
            ):
                return True
            if (y1 > y) != (y2 > y):
                intersection = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < intersection:
                    inside = not inside
            previous = current
        return inside

    def register_user_agent(
        self, agent_class: type[BaseAgent], *, seed=None, options: dict | None = None, max_steps=None,
    ) -> RedAgentGroup:
        """部署后按受控清单创建策略实例，每台仅接收本机和公开初态。"""
        engine = self._require_engine()
        if not self._episode_started or engine.current_step != 0 or not self._deployment_applied:
            raise RuntimeError("红方策略需要在本局初始部署完成后注册")
        simulators = engine.simulator_factory.get_all_simulators()
        entity_ids = tuple(
            simulator.entity_ext.entity.id
            for simulator in simulators
            if simulator.entity_ext.entity.entityType in (21000, 21001, 21002)
        )

        # 公开目标只包含母巢与哨卫巢，隐藏暗巢仍通过后续合法探测获知。

        friendly = {}
        for simulator in simulators:
            entity = simulator.entity_ext.entity
            position = {"x": entity.posEcf.x, "y": entity.posEcf.y, "z": entity.posEcf.z}
            if entity.entityType in (21000, 21001, 21002):
                friendly[entity.id] = {
                    "nameChn": entity.nameChn,
                    "pos": position,
                    "vel": {"x": entity.velEcf.x, "y": entity.velEcf.y, "z": entity.velEcf.z},
                    "health": entity.survivePoints,
                    "type": entity.entityType,
                    "side": entity.sideId,
                    "detectInfo": entity.detectInfo.copy(),
                }
        area = engine.profile.imagineProfile.mapArea
        initial = {}
        for entity_id in entity_ids:
            observation = {
                "run_id": self._run_id,
                "entity_id": entity_id,
                "step": engine.current_step,
                "entities": self._public_targets_observation(),
                "self": deepcopy(friendly[entity_id]),
                "map_area": {key: getattr(area, key) for key in ("xMin", "xMax", "yMin", "yMax")},
                "seed": seed,
                "max_steps": engine.total_steps if max_steps is None else max_steps,
                "options": deepcopy(options) if options is not None else {},
            }
            initial[entity_id] = observation
        return self.register_red_agents(agent_class, entity_ids, initial)

    def get_agent(self, entity_id: int) -> Optional[BaseAgent]:
        """按受控实体编号获取智能体。"""
        return self._agents.get(entity_id)

    def get_agent_by_entity(self, entity_id: int) -> Optional[BaseAgent]:
        """通过实体ID获取对应的智能体"""
        return self.get_agent(entity_id)

    def get_registered_agents(self) -> tuple[tuple[int, BaseAgent], ...]:
        """返回注册实体编号与实例的快照，路由不依赖实例中的可变属性。"""
        return tuple(self._agents.items())

    def get_all_agents(self) -> List[BaseAgent]:
        """获取所有智能体"""
        return list(self._agents.values())

    def get_agent_count(self) -> int:
        """获取智能体总数"""
        return len(self._agents)

    def extract_observation_for_agent(self, full_observation: dict, entity_id: int) -> dict:
        """
        从完整观测中提取指定智能体的隔离观测
        :param full_observation: 完整的观测信息
        :param entity_id: 注册的受控实体ID
        :return: 该智能体的隔离观测
        """
        if entity_id not in self._agents:
            logger.warning(f"[智能体工厂] 实体 {entity_id} 未注册智能体")
            return {}

        # 只返回该智能体对应实体的观测
        entities = full_observation.get("entities", {})
        if entity_id in entities:
            entity_info = entities[entity_id]
            if entity_info["health"] > 0 and entity_info["isVisible"]:
                self_info = {
                    "nameChn": entity_info["nameChn"],
                    "pos": entity_info["pos"],
                    "vel": entity_info["vel"],
                    "stage": entity_info["stage"],
                    "health": entity_info["health"],
                    "isVisible": entity_info["isVisible"],
                    "type": entity_info["type"],
                    "side": entity_info["side"],
                    "commRangeInfo": entity_info["commRangeInfo"],
                    "detectInfo": entity_info["detectInfo"]
                }
                return {
                    "step": full_observation.get("step", 0),
                    "entity_id": entity_id,
                    "self": self_info
                }
            return {}
        else:
            # logger.warning(f"[智能体工厂] 实体 {entity_id} 不在观测中")
            return {}

    def collect_actions_from_agents(self, agent_observations: Dict[int, dict]) -> list[np.array]:
        """
        向所有智能体传递观测信息，并收集智能体的动作
        :param agent_observations: {entity_id: observation}
        :return: 各实体的 N x 4 动作数组列表
        """
        actions = []
        for entity_id, observation in agent_observations.items():
            agent = self._agents.get(entity_id)
            if agent is None:
                raise IllegalActionError(f"实体 {entity_id} 未注册智能体")
            # agent.get_obs(deepcopy(observation))
            agent.get_obs(observation)

        for entity_id in agent_observations:
            # Ownership comes from registration, never from the mutable agent.
            raw_action = self._agents[entity_id].get_action()
            try:
                action = np.asarray(raw_action, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise IllegalActionError(f"实体 {entity_id} 的智能体动作必须是数值数组") from exc
            if action.shape in ((0,), (0, 4)):
                actions.append(np.empty((0, 4), dtype=np.float64))
                continue
            if action.ndim != 2 or action.shape[1] != 4 or not np.isfinite(action).all():
                raise IllegalActionError(f"实体 {entity_id} 的智能体动作必须是有限数值 N x 4 数组")
            if np.any(action[:, 1] != entity_id):
                raise IllegalActionError(f"智能体只能控制自身注册实体 {entity_id}")
            actions.append(action.copy())

        return actions

    def reset_all(self):
        """重置所有智能体"""
        self._episode_started = False
        self._deployment_applied = False
        group = self._red_group
        grouped = set()
        if group is not None:
            group.reset()
            grouped = {id(agent) for agent in group.agents}
        for agent in self._agents.values():
            if id(agent) not in grouped:
                agent.reset()
        if self.engine is not None:
            if group is None:
                self._run_id = uuid4().hex
            self._episode_started = True
        logger.info("[AgentManager] 所有智能体已重置")

    def clear(self):
        """清空所有智能体"""
        self._agents.clear()
        self._red_group = None
        self._episode_started = False
        self._deployment_applied = False
        self._run_id = None
