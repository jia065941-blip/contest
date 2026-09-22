# -*- coding: utf-8 -*-#
import logging
from typing import Dict, Optional, List
import numpy as np
from user_agents.base_agent import BaseAgent, AgentType

logger = logging.getLogger(__name__)


class AgentManager:
    """
    智能体工厂
    负责创建、管理和查询智能体实例
    为每个智能体分发对应的隔离观测
    """

    def __init__(self):
        self._agents: Dict[int, BaseAgent] = {}  # agent_id -> agent
        self._entity_to_agent: Dict[int, int] = {}  # entity_id -> agent_id
        self._agent_to_entity: Dict[int, int] = {}  # agent_id -> entity_id

    def register_agent(self, agent: BaseAgent):
        """
        注册智能体
        :param agent: 智能体实例
        """
        agent_id = agent.agent_id
        entity_id = agent.get_entity_id()

        self._agents[agent_id] = agent
        self._entity_to_agent[entity_id] = agent_id
        self._agent_to_entity[agent_id] = entity_id

        logger.info(f"[智能体工厂] 注册智能体 {agent_id} -> 实体 {entity_id}")

    def get_agent(self, agent_id: int) -> Optional[BaseAgent]:
        """获取指定智能体"""
        return self._agents.get(agent_id)

    def get_agent_by_entity(self, entity_id: int) -> Optional[BaseAgent]:
        """通过实体ID获取对应的智能体"""
        agent_id = self._entity_to_agent.get(entity_id)
        if agent_id:
            return self._agents.get(agent_id)
        return None

    def get_agent_by_type(self, agent_type: AgentType):
        """
        通过智能体类型获取智能体
        :param agent_type:
        :return:
        """
        agents: List[BaseAgent] = self.get_all_agents()
        return [agent for agent in agents if agent.agent_type == agent_type]

    def get_all_agents(self) -> List[BaseAgent]:
        """获取所有智能体"""
        return list(self._agents.values())

    def get_agent_count(self) -> int:
        """获取智能体总数"""
        return len(self._agents)

    def extract_observation_for_agent(self, full_observation: dict, agent_id: int) -> dict:
        """
        从完整观测中提取指定智能体的隔离观测
        :param full_observation: 完整的观测信息
        :param agent_id: 智能体ID
        :return: 该智能体的隔离观测
        """
        entity_id = self._agent_to_entity.get(agent_id)
        if entity_id is None:
            logger.warning(f"[智能体工厂] 智能体 {agent_id} 未找到对应实体")
            return {}

        # 只返回该智能体对应实体的观测
        entities = full_observation.get("entities", {})
        if entity_id in entities:
            self_info = entities[entity_id]
            if self_info["health"] > 0 and self_info["isVisible"]:
                return {
                    "step": full_observation.get("step", 0),
                    "sim_time": full_observation.get("sim_time"),
                    "sim_step": full_observation.get("sim_step"),
                    "entity_id": entity_id,
                    "agent_id": agent_id,
                    "self": entities[entity_id]  # 只包含自己的信息
                }
            return {}
        else:
            # logger.warning(f"[智能体工厂] 实体 {entity_id} 不在观测中")
            return {}

    def collect_actions_from_agents(
        self,
        agent_observations: Dict[int, dict],
        *,
        fail_fast: bool = False,
    ) -> list[np.array]:
        """
        向所有智能体传递观测信息，并收集智能体的动作
        :param agent_observations: {agent_id: observation}
        :return: {entity_id: actions} 格式的总动作字典
        """
        actions = []
        for agent_id, observation in agent_observations.items():
            agent = self._agents.get(agent_id)
            if agent:
                try:
                    agent.set_observation(observation)
                except Exception as e:
                    logger.error(f"[智能体工厂] 智能体 {agent_id} 输入观测信息错误: {e}")
                    if fail_fast:
                        raise
            else:
                logger.warning(f"[智能体工厂] 智能体 {agent_id} 不存在")


        for agent_id, observation in agent_observations.items():
            agent = self._agents.get(agent_id)
            if agent:
                try:
                    action = agent.get_action()
                    actions.append(action)
                except Exception as e:
                    logger.error(f"[智能体工厂] 智能体 {agent_id} 获取动作失败: {e}")
                    if fail_fast:
                        raise
            else:
                logger.warning(f"[智能体工厂] 智能体 {agent_id} 不存在")

        return actions

    def collect_actions_from_deploy_agent(self, agent_observations: Dict[int, dict]) -> np.array:
        """
        收集部署智能体的动作
        :param agent_observations: {agent_id: observation}
        :return: {entity_id: actions} 格式的总动作字典
        """
        action = {}
        agent = self._agents.get(-1)
        if agent:
            try:
                agent.set_observation(agent_observations[-1])
                action = agent.get_action()
            except Exception as e:
                logger.error(f"[智能体工厂] 部署智能体 生成动作失败: {e}")
        else:
            logger.warning(f"[智能体工厂] 部署智能体 不存在")

        return action

    def reset_all(self):
        """重置所有智能体"""
        for agent in self._agents.values():
            agent.reset()
        logger.info("[智能体工厂] 所有智能体已重置")

    def clear(self):
        """清空所有智能体"""
        self._agents.clear()
        self._entity_to_agent.clear()
        self._agent_to_entity.clear()
