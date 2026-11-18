"""R5 事件滚动规划的单机入口。

由通信群状态变化事件和冷却条件触发优先级任务更新。
此文件将基础接口接到示例执行与资源组件；算法位于 common.baselines，
规划时机位于 common.commander，动作生成位于 common.execution。
"""

from user_agents.base import BaseAgent
from user_agents.example.common.execution import AttackExecution
# 同文件的 Deployment 导出供检查器取得，部署阶段使用独立对象。
from user_agents.example.deployment import Deployment
from user_agents.example.common.runtime import _acquire_resources, _release_resources


class RedAttack(BaseAgent):
    """R5 事件触发滚动规划，每个实例处理本机观测、动作和反馈。"""

    # 任务层默认策略；具体分配算法由示例内部的通信群规划器调用。
    policy_name = "r5_event_rolling"
    # 机动层在检测到接近本机的威胁时按规则规避，与目标分配分别配置。
    motion_policy = "reactive_evasion"

    def __init__(self, entity_id, init_observation):
        """创建本机执行状态，并取得同一运行、同一具体类的示例私有资源。"""
        super().__init__(entity_id, init_observation)
        # 各实例独立计数；资源通过回合号识别同一轮重复到达的 reset 请求。
        self._episode = 0
        self._closed = False
        # 资源归属在示例内部登记；执行组件负责把本机观测接入其中。
        self._resource = _acquire_resources(type(self), entity_id, self.init_observation)
        try:
            self._execution = AttackExecution(self, self._resource)
        except BaseException:
            # 构造执行组件失败时撤销本机登记，避免遗留模型或通信成员。
            _release_resources(type(self), self._resource, entity_id, self.init_observation["run_id"])
            raise

    def get_obs(self, observation):
        """保存当前本机观测，并交给执行组件报告到示例内部通信端点。"""
        super().get_obs(observation)
        self._execution.on_observation(observation)

    def get_action(self):
        """读取本机任务，组合攻击、搜索、天眼及机动动作，返回 N×4 数组。"""
        return self._execution.decide_action()

    def on_feedback(self, observation, action, reward, info=None):
        """接收环境推进后的本机反馈；学习模式可由执行组件追加经验。"""
        self._execution.on_feedback(observation, action, reward, info)

    def on_reset(self):
        """由基类清理公共历史后调用，重置示例共享回合状态及本机执行状态。"""
        self._episode += 1
        self._resource._reset_for_episode(self._episode)
        self._execution.on_reset()

    def close(self):
        """撤销本机成员资格，最后一个成员退出时关闭示例资源。"""
        # 运行方可能在清理路径重复调用 close，每个实例只释放一次成员资格。
        if self._closed:
            return
        self._closed = True
        _release_resources(type(self), self._resource, self.entity_id, self.init_observation["run_id"])
