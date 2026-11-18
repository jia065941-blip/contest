# -*-coding:utf-8 -*-
import math

import numpy as np

from envengine.sdk.base_struct.Basic import Vector3d
from user_agents.base_agent import BaseAgent, AgentType
from .config import RedAgentConfig
from .coordination import _CommunicationDirectory
from .priors import initial_targets_from_observation
from .learning import (
    LearningActionAdapter,
    ObservationEncoder,
    PolicyTransition,
)

ACTION_SET_ACC_Z = 0  # 设置Z轴加速度
ACTION_SEARCH_POSITION = 2  # 公开区域中的搜索航点
ACTION_USE_SAT = 3  # 使用天眼
ACTION_ATTACK_TARGET = 4  # 攻击本机合法情报中的目标 ID


class AttackMissileAgent(BaseAgent):
    """
    动作数组格式: np.array([a, b, c, ...], dtype=np.float64)

    横向加速度指令：(加速度值的 1 表示 20G)
    [ACTION_SET_ACC_Z, 实体id, 加速度值, 0]

    使用天眼指令：
    [ACTION_USE_SAT, 实体id, 0, 0]

    Attributes:
        launch_step: 本机首次执行时的帧数，回合开始前为 -1
    """

    def __init__(self, agent_id: int, entity_id: int, init_observation: dict):
        super().__init__(agent_id, entity_id, AgentType.AIRCRAFT, init_observation)
        config = RedAgentConfig()
        communications = _CommunicationDirectory(
            (entity_id,), initial_targets_from_observation(init_observation), config,
        )
        self._configure_example(config, communications, None, None)
        self._commander = None
        self._mailbox = None
        self.set_acc_z_z = 0
        self.acc_start_step = 0
        self.launch_step = -1
        self.sat_used = False
        self.latest_observation: dict | None = None
        self._learning_transition: tuple[np.ndarray, int, np.ndarray] | None = None

    def _configure_example(self, config, communications, actor, transition_sink) -> None:
        """Assemble example components inside user_agents; callers only supply data."""
        self._communications = communications.endpoint(self.entity_id)
        self._motion_policy = config.motion_policy
        self._actor = actor
        self._transition_sink = transition_sink
        self._hierarchical_learning = config.policy_name == "r9_hierarchical_learning"
        self._learning_encoder = (
            ObservationEncoder(
                self.init_observation, max_steps=config.max_steps, agent_id=self.agent_id,
                hierarchical_task_context=self._hierarchical_learning,
            ) if actor is not None else None
        )
        self._learning_adapter = LearningActionAdapter() if actor is not None else None

    def set_observation(self, observation: dict) -> None:
        self.latest_observation = observation
        self._mailbox = self._communications.receive(observation)

    def get_action(self) -> np.array:
        """生成动作"""
        observation = self.latest_observation
        if observation is None or self._mailbox is None:
            raise RuntimeError("The environment must provide an isolated observation")
        self._commander = self._mailbox.prepare(int(observation["step"]))
        actions = []  # 存储多个动作

        step = observation["step"]

        # Deployment starts every red platform before its first decision.
        if self.launch_step < 0:
            self.launch_step = step

        self._commander.report(observation)
        command = self._commander.action_for(self.entity_id, step)
        if command is not None:
            actions.append(command)

        if (
            self.launch_step >= 0
            and not self.sat_used
            and self._commander.should_use_satellite(self.entity_id)
        ):
            actions.append([ACTION_USE_SAT, self.entity_id, 0, 0])
            self.sat_used = True

        if self._actor is not None:
            self._set_acc_z_learning(actions, observation)
        elif self._motion_policy == "reactive_evasion":
            self._set_acc_z_avoid(actions, observation)

        return np.asarray(actions, dtype=np.float64).reshape(-1, 4)

    def _set_acc_z_learning(self, actions: list[list[float]], observation: dict) -> None:
        """Run a shared inference actor on this platform's isolated observation."""

        if self.launch_step < 0:
            return
        assert self._learning_encoder is not None
        assert self._learning_adapter is not None
        assert self._actor is not None
        target_id = self._commander.target_id_for(self.entity_id)
        target_index = next(
            (
                index
                for index, target in enumerate(self._learning_encoder.targets)
                if int(target.get("entity_id", -1)) == target_id
            ),
            None,
        )
        encoded = self._learning_encoder.encode(
            observation,
            launched=True,
            launch_step=self.launch_step,
            satellite_used=self.sat_used,
            maneuver_state=self.set_acc_z_z,
            current_target_index=target_index,
            task_context=(
                self._commander.learning_task_context(self.entity_id)
                if self._hierarchical_learning else None
            ),
        )
        action = self._actor.select_action(
            encoded,
            self._learning_adapter.build_action_mask(),
        )
        if bool(getattr(self._actor, "training", False)):
            self._learning_transition = (encoded, int(action), self._learning_adapter.build_action_mask())
        engine_actions = self._learning_adapter.to_engine_action(int(action), self.entity_id)
        actions.extend(engine_actions.tolist())

    def record_step(self, observation: dict, action, reward: float, info: dict = None):
        """Record local feedback; a separate training caller consumes samples."""

        super().record_step(observation, action, reward, info)
        if self._learning_transition is None or self._actor is None:
            return
        encoded, selected_action, action_mask = self._learning_transition
        self._learning_transition = None
        if not bool(getattr(self._actor, "training", False)):
            return
        assert self._learning_encoder is not None
        next_observation = (
            self._learning_encoder.encode(
                observation,
                launched=True,
                launch_step=self.launch_step,
                satellite_used=self.sat_used,
                maneuver_state=self.set_acc_z_z,
                task_context=(
                    self._commander.learning_task_context(self.entity_id)
                    if self._hierarchical_learning else None
                ),
            )
            if observation.get("self")
            else np.zeros_like(encoded)
        )
        assert self._transition_sink is not None
        self._transition_sink(
            PolicyTransition(
                agent_id=self.agent_id,
                observation=encoded,
                action=selected_action,
                action_mask=action_mask,
                reward=float(reward),
                next_observation=next_observation,
                done=bool((info or {}).get("done", False) or not observation.get("self")),
            )
        )

    def _set_acc_z_avoid(self, actions, observation: dict):
        """
        躲避自爆虫群机动
        :param actions:
        :param observation:
        :return:
        """

        # 当探测到自爆虫群时，进行机动规避
        if self.launch_step < 0:
            return

        interceptors = [
            target
            for target in observation["self"].get("detectInfo", {}).values()
            if (target.get("entity_type") if isinstance(target, dict) else getattr(target, "entity_type", None)) == 24000 and
            self.is_self_interceptor(target, observation["self"])
        ]

        if not interceptors:
            return

        self.acc_start_step += 1

        old_status = self.set_acc_z_z
        if self.set_acc_z_z == 0:
            # 发现来袭自爆虫群后立即横向机动。
            actions.append([ACTION_SET_ACC_Z, self.entity_id, 1, 0])
            self.set_acc_z_z = 1
        elif self.set_acc_z_z == 1 and self.acc_start_step >= 10:
            # 加速一定帧数后，停止加速
            actions.append([ACTION_SET_ACC_Z, self.entity_id, 0, 0])
            self.set_acc_z_z = 2

        if old_status != self.set_acc_z_z:
            self.acc_start_step = 0

        if self.sat_used:
            return
        actions.append([ACTION_USE_SAT, self.entity_id, 0, 0])
        self.sat_used = True

    def is_self_interceptor(self, target, entity: dict) -> bool:
        def field(value, name, default=None):
            return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)

        position = field(target, "pos_ecf")
        velocity = field(target, "vel_ecf")
        if position is None or velocity is None:
            return False
        own = entity["pos_ecf"]
        re_pos = Vector3d(*(own[axis] - float(field(position, axis, 0.0)) for axis in ("x", "y", "z")))
        re_vel = Vector3d(*(float(field(velocity, axis, 0.0)) for axis in ("x", "y", "z")))
        return self.get_vector_angle(re_pos, re_vel) < 20

    def reset(self):
        """重置状态"""
        super().reset()
        self._communications.reset()
        self._commander = None
        self._mailbox = None
        self.set_acc_z_z = 0
        self.acc_start_step = 0
        self.launch_step = -1
        self.sat_used = False
        self._learning_transition = None
        self.latest_observation = None

    @staticmethod
    def get_vector_angle(vector1: Vector3d, vector2: Vector3d) -> float:
        """
        计算两个向量的夹角
        :param vector1: 向量1
        :param vector2: 向量2
        :return: 夹角（度）
        """
        # 点积
        dot = vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z
        # 模长
        norm1 = math.sqrt(vector1.x ** 2 + vector1.y ** 2 + vector1.z ** 2)
        norm2 = math.sqrt(vector2.x ** 2 + vector2.y ** 2 + vector2.z ** 2)
        # 零向量时夹角视为 0 度
        if norm1 == 0 or norm2 == 0:
            return 0.0
        # 余弦值（截断到 [-1, 1] 防止浮点误差）
        cos_theta = max(-1.0, min(1.0, dot / (norm1 * norm2)))
        return math.degrees(math.acos(cos_theta))
