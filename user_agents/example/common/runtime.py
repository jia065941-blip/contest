"""示例按运行标识与具体实现复用通信、Actor 和训练资源。

各 RedAttack 通过私有登记函数取得资源，并仅为自己注册实体编号。
"""

from dataclasses import replace
from weakref import WeakValueDictionary

from .config import LEARNING_MOTION_CHOICES, RedAgentConfig
from .contracts import Position
from .coordination import _CommunicationDirectory
from ..learning.models import build_learning_motion_policy
from .targets import initial_targets_from_observation


class _LocalActor:
    """提供单机动作选择能力，并暴露是否需要收集训练经验。"""

    def __init__(self, policy):
        """保存模型的推理方法引用，执行组件通过此接口选择动作。"""
        self.select_action = policy.select_action
        self.training = bool(getattr(policy, "training", False))


# 弱引用目录仅用于查找同局资源；各 RedAttack 持有强引用并管理实际生命周期。
# run_id 区分不同仿真，具体类区分同一进程中的不同示例实现。
_RESOURCES = WeakValueDictionary()


class _ExampleResources:
    """一局同一示例类共用的内部资源，成员实例负责取得与释放。"""

    def __init__(
        self, init_observation: dict, *, policy_name="r0_random",
        motion_policy="reactive_evasion", model_path=None, training=False,
        config: RedAgentConfig | None = None,
    ):
        """解析示例配置、建立通信目录，并按机动策略创建共享模型。"""
        # 训练入口可显式传完整配置；普通文件入口使用该类声明的默认任务层和机动层。
        options = init_observation.get("options") or {}
        if config is None and options.get("config") is not None:
            config = RedAgentConfig(**options["config"])
        seed = init_observation.get("seed")
        self.config = config or RedAgentConfig(
            policy_name=policy_name, motion_policy=motion_policy,
            seed=1 if seed is None else seed,
            max_steps=init_observation.get("max_steps", 1000),
            model_path=model_path, training=training,
        )
        # 公开搜索边界与公开目标目录是规划初值，后续信息来自各实例实际收到的观测。
        area = init_observation["map_area"]
        polygon = init_observation.get("search_polygon") or (
            (area["xMin"], area["yMin"]), (area["xMax"], area["yMin"]),
            (area["xMax"], area["yMax"]), (area["xMin"], area["yMax"]),
        )
        polygon = tuple(
            point if isinstance(point, Position) else Position(*point)
            for point in polygon
        ) if polygon else None
        self._communications = _CommunicationDirectory(
            (),
            initial_targets_from_observation(init_observation), self.config, polygon,
        )

        # 学习模型和经验队列属于运行实例，执行组件只接收 Actor 与追加能力。
        self._learning_policy = (
            build_learning_motion_policy(
                self.config.motion_policy, seed=self.config.seed, model_path=self.config.model_path,
                max_steps=self.config.max_steps, training=self.config.training,
                observation_dim=90 if self.config.policy_name == "r9_hierarchical_learning" else 85,
            ) if self.config.motion_policy in LEARNING_MOTION_CHOICES else None
        )
        self.actor = _LocalActor(self._learning_policy) if self._learning_policy is not None else None
        # 同一步各实例产生的局部 transition 先入队，训练 runner 在步进后统一消费。
        self._transitions = []
        self.transition_sink = self._transitions.append if self.config.training else None

        # 成员身份由各实例自行登记，资源保存成员编号。
        self._members: set[int] = set()
        # 学习序号只在示例内部使用；成员退出后保留序号，避免其余成员发生重排。
        self._learning_indices: dict[int, int] = {}
        self._episode = 0
        self._closed = False

    def _register_member(self, entity_id: int) -> None:
        """登记当前单机实例的实体编号，并为通信目录增加可用成员。"""
        if self._closed:
            raise RuntimeError("示例资源已经关闭")
        if entity_id in self._members:
            raise ValueError(f"示例实体 {entity_id} 重复登记")
        self._communications.register_member(entity_id)
        self._members.add(entity_id)
        if entity_id not in self._learning_indices:
            self._learning_indices[entity_id] = len(self._learning_indices)

    def learning_identity(self, entity_id: int) -> tuple[int, int]:
        """返回本机登记序号和本运行的登记规模，不暴露其他成员编号。"""
        if entity_id not in self._members:
            raise ValueError(f"示例实体 {entity_id} 尚未登记")
        return self._learning_indices[entity_id], len(self._learning_indices)

    def endpoint(self, entity_id: int):
        """为本机取得邮箱端点，后续报告从该端点进入当前通信群。"""
        return self._communications.endpoint(entity_id)

    def _reset_for_episode(self, episode: int) -> None:
        """以实例上报的回合计数去重，使同一回合的共享资源只重置一次。"""
        if episode <= self._episode:
            return
        self._communications.reset()
        self.reset_training_episode()
        self._episode = episode

    def reset_training_episode(self) -> None:
        """重置学习模型的回合缓存，并清空待提交的单机经验。"""
        if self._learning_policy is not None and self.config.training:
            self._learning_policy.reset_episode()
        self._transitions.clear()

    def begin_training_step(self, observation: dict) -> None:
        """训练调用方在推进前提供全态，供 MAPPO 构建当前集中式 Critic 上下文。"""
        if self._learning_policy is not None and self.config.training:
            self._learning_policy.begin_environment_step(observation)

    def finish_training_step(self, observation: dict) -> None:
        """推进后补齐下一全态，将本步单机经验提交模型并触发更新检查。"""
        policy = self._learning_policy
        if policy is None or not self.config.training:
            return
        # 全态只在显式训练调用中进入模型上下文；单机执行组件提交的样本仍是本机编码。
        policy.end_environment_step(observation)
        global_state, next_global_state = policy.get_global_state_context()
        # 原地清空队列，保留各执行组件持有的 bound append 引用。
        pending = tuple(self._transitions)
        self._transitions.clear()
        for transition in pending:
            policy.observe(replace(
                transition, global_state=global_state, next_global_state=next_global_state,
            ))
        policy.finish_environment_step()

    def save_model(self, path: str) -> None:
        """将已创建的学习模型保存到训练调用方指定的位置。"""
        if self._learning_policy is None:
            raise RuntimeError("该红方示例没有可保存的学习模型")
        self._learning_policy.save(path)

    def _close(self) -> None:
        """最后一个成员退出时关闭通信与模型资源，多次调用只处理一次。"""
        if self._closed:
            return
        self._closed = True
        try:
            self._communications.reset()
        finally:
            self._transitions.clear()
            close = getattr(self._learning_policy, "close", None)
            if close is not None:
                close()


def _acquire_resources(implementation, entity_id: int, init_observation: dict) -> _ExampleResources:
    """以 (run_id, 具体实现类) 为键取得资源，再登记当前实体。"""
    run_id = init_observation.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("示例初始化需要非空字符串 run_id")
    key = (run_id, implementation)
    # 首个成员创建一次模型和目录，后续同局同类成员复用该资源。
    resource = _RESOURCES.get(key)
    if resource is None:
        resource = _ExampleResources(
            init_observation,
            policy_name=implementation.policy_name,
            motion_policy=implementation.motion_policy,
            model_path=getattr(implementation, "model_path", None),
            training=getattr(implementation, "training", False),
        )
        _RESOURCES[key] = resource
    try:
        resource._register_member(entity_id)
    except BaseException:
        # 首次登记失败时收回刚创建的空资源；已有成员的资源仍由这些成员持有。
        if not resource._members:
            try:
                resource._close()
            finally:
                _RESOURCES.pop(key, None)
        raise
    return resource


def _release_resources(implementation, resource: _ExampleResources, entity_id: int, run_id: str) -> None:
    """撤销当前成员；成员集合变空时关闭资源并移除相应缓存键。"""
    resource._members.discard(entity_id)
    try:
        resource._communications.unregister_member(entity_id)
    finally:
        if not resource._members:
            key = (run_id, implementation)
            try:
                resource._close()
            finally:
                if _RESOURCES.get(key) is resource:
                    _RESOURCES.pop(key, None)
