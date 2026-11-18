# -*- coding: utf-8 -*-#
"""单实体策略的公共接口，以及由环境驱动的历史与生命周期管理。

AgentManager 为每个受控实体创建一个实例，并逐步调用 get_obs、get_action。
派生类可以自行组织策略工具；基类保存本机数据，提供重置、反馈和关闭扩展点。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, final
from collections import deque

if TYPE_CHECKING:
    # 数组类型只用于注解，导入公共基类时无需加载数值计算库。
    import numpy as np


class BaseAgent(ABC):
    """单实体策略基类，保存观测、历史和本机生命周期状态。

    常规决策在 get_obs/get_action 中完成；自定义状态在 on_reset 中重置，
    学习经验等后处理在 on_feedback 中完成，实例资源在 close 中释放。
    """

    def __init__(self, entity_id: int, init_observation: dict, *, history_length: int = 10):
        """绑定控制对象，保存部署后的初始化资料，并建立有限长度的历史队列。

        :param entity_id: 受控实体编号，用于路由观测、动作和奖励。
        :param init_observation: 本机初态、公开目标目录及运行配置，由管理器组装。
        :param history_length: 各历史队列最多保留的步数，默认 10；超出后丢弃最早项。
        """
        # 身份与初态：初始资料跨步保留，latest_observation 在接收当前观测时更新。
        self.entity_id = entity_id
        self.init_observation = init_observation
        self.latest_observation: dict | None = None

        # 单步反馈历史：四个队列按同一调用顺序追加，用于回看最近的状态、动作与奖励。
        self.history_length = history_length
        self.observation_history = deque(maxlen=history_length)
        self.action_history = deque(maxlen=history_length)
        self.reward_history = deque(maxlen=history_length)
        self.info_history = deque(maxlen=history_length)

    def get_obs(self, observation: dict) -> None:
        """接收并保存本机观测，供本次动作计算使用。

        管理器先为本步各实例分发观测，再逐个请求动作。派生类覆写时可先调用
        super().get_obs(observation)，随后提取特征或向自己实现的协同工具报告数据。
        本机状态使用 pos={x,y,z}（米）、vel={x,y,z}（米/秒），轴顺序为东、北、上。
        红方 self.detectInfo 为实体编号到探测记录字典的映射，记录用 detected_step
        表示产生时的仿真步号；缓存或转发不会改成当前 step，不公开内部毫秒 time。
        """
        self.latest_observation = observation

    @abstractmethod
    def get_action(self) -> np.ndarray:
        """根据已接收观测返回本机动作，由选手实现。

        示例使用 N×4 数值数组，每行由动作类型、自身实体编号和两个参数组成；
        空动作可返回形状为 (0, 4) 的数组。动作的具体解释与执行由管理器和引擎负责。
        """
        pass

    @final
    def record_step(self, observation: dict, action: Any, reward: float, info: dict = None):
        """
        记录步进后的本机反馈，再通知具体策略实现；由环境自动调用。

        :param observation: 本次推进后的本机观测，与决策前 get_obs 的观测时点不同。
        :param action: 环境返回的本机执行动作记录，可能为 None。
        :param reward: 环境提供的逐步奖励，按原值保存。
        :param info: 本步附加资料，如步数与结束标志。
        """
        # 先写公共历史，再调用扩展点，使 on_feedback 能读取本步在内的最近记录。
        self.observation_history.append(observation)
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.info_history.append(info or {})
        self.on_feedback(observation, action, reward, info)

    def get_recent_observations(self, steps: int = None) -> List[dict]:
        """
        获取最近的观测历史
        :param steps: 获取最近多少步，None表示全部
        :return: 观测历史列表
        """
        if steps is None:
            return list(self.observation_history)
        return list(self.observation_history)[-steps:]

    def get_recent_actions(self, steps: int = None) -> List[Any]:
        """获取最近的动作历史"""
        if steps is None:
            return list(self.action_history)
        return list(self.action_history)[-steps:]

    def get_recent_rewards(self, steps: int = None) -> List[float]:
        """获取最近的奖励历史"""
        if steps is None:
            return list(self.reward_history)
        return list(self.reward_history)[-steps:]

    def get_cumulative_reward(self, steps: int = None) -> float:
        """
        累加当前历史队列中的逐步奖励。

        :param steps: 最近多少条记录，None 表示队列中保留的全部记录。
        :return: 选中记录的奖励和；队列长度受 history_length 限制。
        """
        rewards = self.get_recent_rewards(steps)
        return sum(rewards)

    @final
    def reset(self) -> None:
        """清理历史和最新观测，再重置具体实现的本机状态。"""
        # 管理器在开局或重新 reset 环境时调用；派生类通过 on_reset 补充自己的清理。
        self.observation_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
        self.info_history.clear()
        self.latest_observation = None
        self.on_reset()

    def on_reset(self) -> None:
        """重置具体实现持有的单实体回合状态，默认无需额外处理。"""

    def on_feedback(self, observation: dict, action: Any, reward: float, info: dict = None) -> None:
        """处理公共历史已记录的本机反馈，可在此追加学习经验或更新策略统计。"""

    def close(self) -> None:
        """释放具体实现持有的本机资源，由持有策略实例的运行方在结束时调用。"""

    def get_entity_id(self) -> int:
        """获取该智能体控制的实体ID"""
        return self.entity_id
