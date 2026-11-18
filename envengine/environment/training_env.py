# -*-coding:utf-8 -*-

import logging
from copy import deepcopy
from typing import Optional

from blue_defense import BaseBlueDefense
from envengine.agent_manager import AgentManager
from envengine.engine import Engine
from envengine.engine.engine import MISSILE_COMMUNICATION_RANGE_M
from envengine.environment.command_adapter import CommandAdapter
from envengine.environment.command_converter import CommandConverter
from envengine.render import Renderer
from envengine.sdk.base_struct.profile.profile import Profile
from envengine.sdk.writer import write_state, write_ai_action, write_config
from scenarios.config import ScenarioConfig
from .scene_adapter import public_detections, public_communications

logger = logging.getLogger(__name__)


class TrainingEnv:
    """
    训练环境 - 封装Engine，提供标准RL接口
    支持多智能体隔离观测
    """

    def __init__(
        self, profile: ScenarioConfig | Profile, render_mode: Optional[str] = None, *, record_logs: bool = True,
    ):
        # 接收场景侧组装的配置，复制初始数据供模型维护本局状态。
        self.config = profile if isinstance(profile, ScenarioConfig) else None
        profile = deepcopy(self.config.profile if self.config is not None else profile)
        self.engine = Engine(profile, 0)
        self.current_round = 0
        self.current_step = 0
        self.max_steps = self.engine.total_steps
        self.render_mode = render_mode
        self._record_logs = record_logs
        self.renderer = None
        if render_mode == "human":
            map_area = profile.imagineProfile.mapArea
            self.renderer = Renderer(map_area.xMin, map_area.xMax, map_area.yMin, map_area.yMax)
            self.renderer.start()

        # 管理器接收可信引擎与场景配置，负责策略注册和初始布局接入。
        self.agent_manager = AgentManager(self.engine, self.config, record_logs=record_logs)

        # 初始化上一帧状态缓存（用于奖励计算）。
        self._last_observation = None
        self._last_actions = None

    def register_blue_policy(self, policy: BaseBlueDefense) -> None:
        """将运行阶段创建的蓝方策略绑定到场景中唯一的控制节点。"""
        if not isinstance(policy, BaseBlueDefense):
            raise TypeError("blue policy must inherit BaseBlueDefense")
        if not isinstance(policy.name, str) or not policy.name.strip():
            raise ValueError("blue policy name must be a nonempty string")
        for attribute in ("plans_launch_time", "preserves_pending_launches"):
            if type(getattr(policy, attribute)) is not bool:
                raise TypeError(f"blue policy {attribute} must be bool")

        commanders = self.engine.simulator_factory.get_simulators_by_type(35000)
        if len(commanders) != 1:
            raise ValueError(
                f"blue policy registration requires exactly one type 35000 commander; found {len(commanders)}"
            )
        commanders[0].set_defense_controller(policy)

    def reset(self) -> dict:
        """
        重置环境
        :return: 初始观测
        """
        # 重置引擎
        self.engine.reset()
        self.current_step = 0
        self.current_round += 1

        # 重置所有智能体
        self.agent_manager.reset_all()

        # 获取初始态势作为观测
        observation = self._get_observation()

        if getattr(self, "_record_logs", True):
            # 将想定配置和初始态势写入文件；预览初始化不接触文件写入器。
            imagine_profile = self.engine.profile.imagineProfile
            red_area = imagine_profile.redArea
            map_area = imagine_profile.mapArea
            write_config({"mapArea": map_area.__dict__, "redArea": red_area.__dict__}, str(self.current_round) + ".json")
            write_state(observation, str(self.current_round) + ".json")

        # 重置上一帧状态缓存
        self._last_observation = observation.copy()
        self._last_actions = None

        logger.info(f"[训练环境] 环境已重置")
        if self.render_mode == 'human' and self.renderer:
            self.renderer.update_data(observation["entities"])
            # 更新红蓝方区域
            self.renderer.update_area(self.engine.profile.imagineProfile.redArea.coordinates)
        return observation

    def step(self) -> tuple[dict, dict, bool, dict]:
        """
        执行一步
        """

        factory = getattr(self.engine, "simulator_factory", None)
        if factory is not None:
            for commander in factory.get_simulators_by_type(35000):
                entity = commander.entity_ext.entity
                if entity.survivePoints > 0 and getattr(commander, "defense_controller", None) is None:
                    raise RuntimeError(f"虫族控制节点 {entity.id} 尚未配置防御控制器，不能开始仿真步进")

        # 获取观测
        observation = self._get_observation()

        action: list[dict] = self._generate_actions_from_agents(observation)
        # 环境负责提交原有过程记录，具体通道开关与采样频率由 SDK writer 配置控制。
        if getattr(self, "_record_logs", True):
            for i in action:
                write_ai_action(i, str(self.current_round) + ".json")

        # 对AI指令到引擎可接收指令进行适配
        action_adapter = CommandAdapter.common_adapter(action)
        # 执行仿真步进
        self.engine.step(action_adapter)
        self.current_step += 1

        if getattr(self, "_record_logs", True):
            write_state(observation, str(self.current_round) + ".json")

        # 计算奖励
        rewards = self._compute_reward()

        # 判断是否结束
        done = self.get_is_done()

        # 额外信息
        info = {
            "step": self.current_step,
            "max_steps": self.max_steps,
            "done": done,
        }

        # 通知每个智能体它们的奖励，并记录历史
        for entity_id, agent in self.agent_manager.get_registered_agents():
            reward = rewards.get(entity_id, 0.0)

            # 为该智能体提取隔离观测
            isolated_obs = self.agent_manager.extract_observation_for_agent(
                observation, entity_id
            )

            # 获取该智能体刚才执行的动作
            agent_action = None
            for cmd in action:
                if isinstance(cmd, dict) and "executor_id" in cmd and cmd["executor_id"] == entity_id:
                    agent_action = cmd
                    break
            # 记录这一步的经验
            agent.record_step(
                observation=isolated_obs,
                action=agent_action,
                reward=reward,
                info=info
            )

        if self.render_mode == 'human' and self.renderer:
            self.renderer.update_data(observation["entities"])

        # 更新上一帧状态
        self._last_observation = observation.copy()
        self._last_actions = action.copy() if action else None
        return observation, rewards, done, info

    def get_is_done(self) -> bool:
        if self.current_step >= self.max_steps:
            return True

        # 所有人族单位及自爆虫结束活动后，仿真结束
        for h in self.engine.simulator_factory.get_simulators_by_type(21000):
            if h.entity_ext.entity.isVisible:
                return False

        for l in self.engine.simulator_factory.get_simulators_by_type(21001):
            if l.entity_ext.entity.isVisible:
                return False

        for m in self.engine.simulator_factory.get_simulators_by_type(21002):
            if m.entity_ext.entity.isVisible:
                return False

        for la in self.engine.simulator_factory.get_simulators_by_type(24000):
            if la.entity_ext.entity.isVisible:
                return False

        return True

    def _generate_actions_from_agents(self, full_observation) -> list[dict]:
        """Deliver isolated observations, collect agent actions, and convert them."""
        self.engine.refresh_red_communications()

        # # 获取当前完整观测
        # full_observation = self._get_observation()

        # 为每个智能体(前提是模型存活)提取隔离观测并收集动作
        agent_observations = {}
        for entity_id, _ in self.agent_manager.get_registered_agents():
            isolated_obs = self.agent_manager.extract_observation_for_agent(
                full_observation,
                entity_id
            )

            if isolated_obs:
                isolated_obs["is_using_satellite"] = self.engine.simulator_factory.is_using_satellite()
                agent_observations[entity_id] = isolated_obs

        # 收集所有智能体的动作
        actions = self.agent_manager.collect_actions_from_agents(agent_observations)
        # 转换所有智能体动作到结构体类型
        return CommandConverter.common_converter(actions)

    def get_observation(self) -> dict:
        """Return a full state snapshot for trusted training and statistics callers.

        Refresh engine-owned communications before copying the state. Platform
        agents continue to receive only their isolated observations through the
        agent manager; this query is not part of the actor interface.
        """
        self.engine.refresh_red_communications()
        observation = self._get_observation()
        # 初态查询同步缓存和界面，使外部完成的模型初始化及时反映到记录中。
        if self.current_step == 0:
            self._last_observation = deepcopy(observation)
            if self.render_mode == "human" and self.renderer:
                self.renderer.update_data(observation["entities"])
        return deepcopy(observation)

    def _get_observation(self) -> dict:
        """导出环境完整态势，供运行记录、组织方查看和渲染使用。"""
        simulators = self.engine.simulator_factory.get_all_simulators()
        by_id = {item.entity_ext.entity.id: item.entity_ext.entity for item in simulators}
        profile = self.engine.profile.imagineProfile
        area = profile.mapArea
        # 场景静态区域用于组织方预览和记录，读取时生成独立列表。
        config = getattr(self, "config", None)
        deployment = config.deployment if config is not None else None
        coordinates = deployment.coordinates if deployment is not None else profile.redArea.coordinates
        if coordinates and coordinates[0] and not isinstance(coordinates[0][0], (list, tuple)):
            coordinates = [coordinates]
        observation = {
            "step": self.current_step,
            "sim_time_ms": self.engine.sim_time,
            "map_area": {key: getattr(area, key) for key in ("xMin", "xMax", "yMin", "yMax")},
            "deployment_area": {
                "coordinates": [[list(point) for point in polygon] for polygon in coordinates],
                "launch_altitude_m": deployment.launch_altitude if deployment is not None else None,
            },
            "entities": {},
        }
        for identifier, entity in by_id.items():
            kind = entity.entityType
            observation["entities"][identifier] = {
                "id": identifier,
                "nameChn": entity.nameChn,
                "type": kind,
                "side": entity.sideId,
                "pos": {
                    "x": entity.posEcf.x,
                    "y": entity.posEcf.y,
                    "z": entity.posEcf.z
                },
                "vel": {
                    "x": entity.velEcf.x,
                    "y": entity.velEcf.y,
                    "z": entity.velEcf.z
                },
                "health": entity.survivePoints,
                "isVisible": entity.isVisible,
                "stage": entity.stage,
                "parent_id": entity.parentId,
                "children_ids": list(entity.childrenId),
                "inventory": {"interceptors": sum(
                    by_id[child].entityType == 24000 for child in entity.childrenId if child in by_id
                )},
                "detection_range_m": 20000.0 if kind == 44000 else None,
                "communication_range_m": MISSILE_COMMUNICATION_RANGE_M.get(kind),
                "detectInfo": entity.detectInfo.copy(),
                # "commRangeInfo": public_communications(entity.commRangeInfo),
                "commRangeInfo": entity.commRangeInfo,
            }
        return observation

    def _compute_reward(self) -> dict[int, float]:
        """
        计算每个智能体的奖励
        :return: {entity_id: reward}
        """
        rewards = {}

        # 获取所有智能体
        for entity_id, _ in self.agent_manager.get_registered_agents():

            # 获取该实体的仿真器
            simulator = self.engine.get_simulator_by_id(entity_id)
            if not simulator:
                rewards[entity_id] = 0.0
                continue

            entity_data = simulator.entity_ext.entity

            # Per-step learning reward; competition scoring lives in scenarios.
            rewards[entity_id] = float(entity_data.survivePoints > 0)

        return rewards

    def close(self):
        """关闭环境"""
        if self.renderer:
            self.renderer.stop()
        self.agent_manager.clear()
        logger.debug("[TrainingEnv] 环境已关闭")

    # 委托方法
    def get_simulator(self, entity_id: int):
        return self.engine.get_simulator_by_id(entity_id)

    def get_simulators_by_type(self, entity_type: int):
        return self.engine.get_simulators_by_type(entity_type)

    def get_simulators_by_side(self, side_id: int):
        return self.engine.get_simulators_by_side(side_id)
