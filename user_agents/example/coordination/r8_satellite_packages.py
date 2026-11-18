"""R8 天眼支援入口：沿用 R7 协同编组，并为目标选择 H 平台申请天眼。

任务层在本轮已分配的 H 平台中，按 9400/9600 目标去重选择天眼申请者，
执行组件再输出本机天眼动作，实际额度与执行效果由引擎处理。文件同时导出 Deployment
部署入口；生命周期和本机观测、动作接口与其他示例保持一致。
"""

from user_agents.base import BaseAgent
from user_agents.example.common.execution import AttackExecution
from user_agents.example.deployment import Deployment
from user_agents.example.common.runtime import _acquire_resources, _release_resources


class RedAttack(BaseAgent):
    """R8 天眼支援协同，每个实例处理本机观测、动作和反馈。"""

    # R8 只扩展任务层的天眼选择，默认横向机动仍使用规则规避。
    policy_name = "r8_satellite_packages"
    motion_policy = "reactive_evasion"

    def __init__(self, entity_id, init_observation):
        """建立单机状态，并登记本次运行内的示例私有资源。"""
        super().__init__(entity_id, init_observation)
        self._episode = 0
        self._closed = False
        # 同 run_id、同入口类的实体可复用内部工具，每台实体仍各有执行组件。
        self._resource = _acquire_resources(type(self), entity_id, self.init_observation)
        try:
            self._execution = AttackExecution(self, self._resource)
        # 构造中断时撤销当前成员；其余成员持有的资源继续由各自生命周期管理。
        except BaseException:
            _release_resources(type(self), self._resource, entity_id, self.init_observation["run_id"])
            raise

    def get_obs(self, observation):
        """先保存本机观测，再向示例内部邮箱提交该帧数据。"""
        super().get_obs(observation)
        self._execution.on_observation(observation)

    def get_action(self):
        """组合本机任务指令、符合申请条件的天眼指令和横向机动。"""
        return self._execution.decide_action()

    def on_feedback(self, observation, action, reward, info=None):
        """将已记录的本机反馈交回执行组件，补齐其内部状态或经验。"""
        self._execution.on_feedback(observation, action, reward, info)

    def on_reset(self):
        """按回合编号协调一次资源重置，并清理本机天眼与机动状态。"""
        self._episode += 1
        self._resource._reset_for_episode(self._episode)
        self._execution.on_reset()

    def close(self):
        """本机退出示例资源成员表，最后成员负责释放内部资源。"""
        # 关闭标志避免重复退订同一实体。
        if self._closed:
            return
        self._closed = True
        _release_resources(type(self), self._resource, self.entity_id, self.init_observation["run_id"])
