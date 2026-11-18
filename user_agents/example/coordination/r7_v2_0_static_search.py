"""R7 2.0：基础协同编组加静态搜索。

沿用 R7 1.0 的公共部署与主攻分配，增加清道夫三机搜索和发现暗巢后的改打。
R7 3.0 使用独立的空间波次部署与作战实现，比较时须同时记录部署差异。
测试版本号用于区分方案，不表示分数排名；内部 policy_name 保持原值。

search.py 根据公开区域和当前提供的平台、哨卫记录产生搜索目的地，执行
组件将目的地转为航点动作；收到暗巢记录后可切换为目标编号攻击。
本文件导出 Deployment 与 RedAttack，并选择静态搜索任务层及规则规避机动层。
"""

from user_agents.base import BaseAgent
from user_agents.example.common.execution import AttackExecution
from user_agents.example.deployment import Deployment
from user_agents.example.common.runtime import _acquire_resources, _release_resources


class RedAttack(BaseAgent):
    """R7 静态三机搜索，每个实例处理本机观测、动作和反馈。"""

    # 搜索几何由任务层内部计算，单机接口仍只接收观测并输出动作数组。
    policy_name = "r7_static_search"
    motion_policy = "reactive_evasion"

    def __init__(self, entity_id, init_observation):
        """建立单机状态，取得搜索策略所用的内部通信与执行资源。"""
        super().__init__(entity_id, init_observation)
        self._episode = 0
        self._closed = False
        # run_id 隔离不同运行，具体入口类隔离同一运行中不同示例的资源。
        self._resource = _acquire_resources(type(self), entity_id, self.init_observation)
        try:
            self._execution = AttackExecution(self, self._resource)
        # 执行组件初始化失败也要归还本机资格，让空资源可以及时关闭。
        except BaseException:
            _release_resources(type(self), self._resource, entity_id, self.init_observation["run_id"])
            raise

    def get_obs(self, observation):
        """保存本机观测并提交报告，供后续任务准备与搜索改打使用。"""
        super().get_obs(observation)
        self._execution.on_observation(observation)

    def get_action(self):
        """把本机搜索或进攻任务与横向机动组合为 N×4 动作数组。"""
        return self._execution.decide_action()

    def on_feedback(self, observation, action, reward, info=None):
        """在公共历史记录之后转交本机反馈。"""
        self._execution.on_feedback(observation, action, reward, info)

    def on_reset(self):
        """进入新回合时重建内部规划状态，再清除本机执行状态。"""
        self._episode += 1
        self._resource._reset_for_episode(self._episode)
        self._execution.on_reset()

    def close(self):
        """退出本机成员资格，最后成员负责清空示例资源。"""
        # 关闭标志让重复清理保持幂等。
        if self._closed:
            return
        self._closed = True
        _release_resources(type(self), self._resource, self.entity_id, self.init_observation["run_id"])
