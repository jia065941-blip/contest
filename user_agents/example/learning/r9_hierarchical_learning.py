"""R9 分层示例：通信群任务层确定目标，学习机动层选择横向动作。

本文件导出独立部署入口 Deployment 和单实体策略入口 RedAttack。
任务规划、模型复用及经验收集由 example 内部组件完成。
"""

from user_agents.base import BaseAgent
from user_agents.example.common.execution import AttackExecution
from user_agents.example.deployment import Deployment
from user_agents.example.common.runtime import _acquire_resources, _release_resources


class RedAttack(BaseAgent):
    """R9 通信群任务与学习机动，每个实例处理本机观测、动作和反馈。"""

    # 任务层采用R9组合逻辑；机动层默认随机掩码，可通过示例配置选择PPO/MAPPO。
    policy_name = "r9_hierarchical_learning"
    motion_policy = "random_masked"

    def __init__(self, entity_id, init_observation):
        """保存本机身份，取得本局示例资源，再建立本机执行状态。"""
        super().__init__(entity_id, init_observation)
        self._episode = 0
        self._closed = False
        # 资源以运行编号和具体实现类分组；本机执行器保存各自的飞行与动作状态。
        self._resource = _acquire_resources(type(self), entity_id, self.init_observation)
        try:
            self._execution = AttackExecution(self, self._resource)
        except BaseException:
            # 执行器初始化失败时归还本机引用，避免留下无人使用的模型和任务组件。
            _release_resources(type(self), self._resource, entity_id, self.init_observation["run_id"])
            raise

    def get_obs(self, observation):
        """保存本机观测，并交给执行器整理目标、通信和机动编码所需资料。"""
        super().get_obs(observation)
        self._execution.on_observation(observation)

    def get_action(self):
        """由本机执行器组合任务层动作与学习机动动作。"""
        return self._execution.decide_action()

    def on_feedback(self, observation, action, reward, info=None):
        """把本机步进反馈交给执行器，按运行模式暂存训练经验。"""
        self._execution.on_feedback(observation, action, reward, info)

    def on_reset(self):
        """按回合序号重置示例公共状态，并清空本机执行状态。"""
        self._episode += 1
        self._resource._reset_for_episode(self._episode)
        self._execution.on_reset()

    def close(self):
        """仅归还一次资源引用；最后一个使用者负责结束公共资源生命周期。"""
        if self._closed:
            return
        self._closed = True
        _release_resources(type(self), self._resource, self.entity_id, self.init_observation["run_id"])
