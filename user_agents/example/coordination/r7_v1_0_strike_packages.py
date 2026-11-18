"""R7 1.0：基础协同编组。

默认使用公共随机部署，为毁灭者与突击者安排同目标进攻，机动层采用规则规避。
R7 2.0 在这套任务分配上增加清道夫三机静态搜索。
测试版本号用于区分方案，不表示分数排名；内部 policy_name 保持原值。

该文件导出部署类 Deployment 和单机类 RedAttack，供文件检查器加载。
具体分配规则位于 common/baselines.py，单机观测与动作衔接由 AttackExecution
完成；本入口负责选择策略默认值，并管理各实体的示例内部资源生命周期。
"""

from user_agents.base import BaseAgent
from user_agents.example.common.execution import AttackExecution
from user_agents.example.deployment import Deployment
from user_agents.example.common.runtime import _acquire_resources, _release_resources


class RedAttack(BaseAgent):
    """R7 协同编组协同，每个实例处理本机观测、动作和反馈。"""

    # 任务策略在推进阶段将毁灭者 H 与突击者 M 配对；机动策略处理单机横向规避。
    policy_name = "r7_strike_packages"
    motion_policy = "reactive_evasion"

    def __init__(self, entity_id, init_observation):
        """先建立单机公共状态，再接入本次运行的示例资源与执行组件。"""
        super().__init__(entity_id, init_observation)
        self._episode = 0
        self._closed = False
        # 资源键由 run_id 和具体入口类组成，本机只登记自己的实体编号。
        self._resource = _acquire_resources(type(self), entity_id, self.init_observation)
        try:
            self._execution = AttackExecution(self, self._resource)
        # 执行组件构造失败时撤销本机成员资格，避免留下无人使用的模型或通信资源。
        except BaseException:
            _release_resources(type(self), self._resource, entity_id, self.init_observation["run_id"])
            raise

    def get_obs(self, observation):
        """保存本机最新观测，再由执行组件提交本帧报告。"""
        super().get_obs(observation)
        self._execution.on_observation(observation)

    def get_action(self):
        """取得群内任务并组合本机机动，返回 N×4 动作数组。"""
        return self._execution.decide_action()

    def on_feedback(self, observation, action, reward, info=None):
        """公共历史记录完成后，向执行组件转交本机反馈。"""
        self._execution.on_feedback(observation, action, reward, info)

    def on_reset(self):
        """报告本机回合计数，资源每局重置一次，再清理本机执行状态。"""
        self._episode += 1
        self._resource._reset_for_episode(self._episode)
        self._execution.on_reset()

    def close(self):
        """释放本机成员资格；最后退出的成员触发示例资源关闭。"""
        # 同一实例可被多次 close，成员资格只释放一次。
        if self._closed:
            return
        self._closed = True
        _release_resources(type(self), self._resource, self.entity_id, self.init_observation["run_id"])
