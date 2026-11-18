"""红方示例单机执行：提交观测、取得群内任务、选择机动并整理本机反馈。"""

import math
from functools import partial
from typing import NamedTuple

import numpy as np

from envengine.sdk.base_struct.Basic import DetectInfo
from user_agents.base import BaseAgent
from ..learning.models import (
    LearningActionAdapter,
    ObservationEncoder,
    PolicyTransition,
)
from .runtime import _ExampleResources

ACTION_SET_TARGET = 1  # 设置目标引导坐标
ACTION_USE_SAT = 2  # 使用天眼
ACTION_SET_ACC_Z = 3  # 设置Z轴加速度


class _Vector3(NamedTuple):
    """供规避规则计算夹角的本地三维位置差或速度。"""
    x: float
    y: float
    z: float


class AttackExecution:
    """组合任务指令、天眼与横向机动，生成 N×4 的 float64 动作数组。

    动作每行为 [动作码, 本机实体编号, 参数一, 参数二]。
    机动值为倍率，H/L 按 20G、M 按 15G 换算，0 停止机动。
    launch_step 记录本机首次决策帧；原生启动已在部署方案应用后完成。
    """

    _COS_20_SQ = math.cos(math.radians(20)) ** 2  # ≈ 0.88302222

    def __init__(self, agent: BaseAgent, resource: _ExampleResources):
        # 执行组件只组合本机身份、局部邮箱与 Actor 推理能力。
        """绑定本机、邮箱与 Actor，并初始化当前回合的机动和天眼状态。"""
        self._agent = agent
        self.entity_id = agent.entity_id
        self.init_observation = {
            "entities": agent.init_observation.get("entities", {}),
            "map_area": agent.init_observation["map_area"],
        }
        self._communications = resource.endpoint(self.entity_id)
        self._motion_policy = resource.config.motion_policy
        self._actor = resource.actor
        self._transition_sink = resource.transition_sink
        self._hierarchical_learning = resource.config.policy_name == "r9_hierarchical_learning"
        # 所有成员注册完成后再读取本机学习序号，实体ID本身不参与特征运算。
        self._learning_identity = partial(resource.learning_identity, self.entity_id)
        # 普通学习机动使用 85 维编码；R9 额外拼入五维群内任务上下文。
        self._learning_encoder = (
            ObservationEncoder(
                self.init_observation, max_steps=resource.config.max_steps,
                hierarchical_task_context=self._hierarchical_learning,
            ) if self._actor is not None else None
        )
        self._learning_adapter = LearningActionAdapter() if self._actor is not None else None
        # 邮箱在 get_obs 时取得，规划器在 get_action 的本帧准备阶段取得。
        self._commander = None
        self._mailbox = None
        # 规避状态 0/1/2 分别表示尚未机动、正在横向加速、已停止加速。
        self.set_acc_z_z = 0
        self.acc_start_step = 0
        self.launch_step = -1
        self.sat_used = False
        # 保存本次推理的编码、离散动作和掩码，等待 record_step 后补齐下一状态。
        self._learning_transition: tuple[np.ndarray, int, np.ndarray] | None = None

    @property
    def latest_observation(self) -> dict | None:
        """通过策略实例读取其最新一次 get_obs 保存的本机观测。"""
        return self._agent.latest_observation

    def on_observation(self, observation: dict) -> None:
        """在观测阶段提交本机报告，保留当前帧所属的群邮箱。"""
        self._mailbox = self._communications.receive(observation)

    def decide_action(self) -> np.array:
        """准备群内规划，依次合成任务、天眼和机动动作，并统一输出数组形状。"""
        observation = self.latest_observation
        if observation is None or self._mailbox is None:
            raise RuntimeError("The environment must provide an isolated observation")
        # 当前群首个成员请求时汇总同帧报告；其余成员共享已准备好的本帧上下文。
        self._commander = self._mailbox.prepare(int(observation["step"]))
        actions = []  # 同一帧允许组合目标指令、天眼和机动。

        step = observation["step"]

        # 部署阶段已统一启动原生模型，此处仅记录本机第一次参与决策的帧。
        if self.launch_step < 0:
            self.launch_step = step

        # 任务层先输出目标引导坐标，机动层随后追加横向控制。
        self._commander.report(observation)
        command = self._commander.waypoint_action_for(self.entity_id, step, self.init_observation['map_area'])
        if command is not None:
            actions.append(command)

        # 任务层选中的平台只申请一次天眼；实际动作仍交由引擎检查额度和适用条件。
        if (
            self.launch_step >= 0
            and not self.sat_used
            and self._commander.should_use_satellite(self.entity_id)
        ):
            actions.append([ACTION_USE_SAT, self.entity_id, 0, 0])
            self.sat_used = True

        # 有 Actor 时走学习机动；规则规避仅在 reactive_evasion 模式启用。
        if self._actor is not None:
            self._set_acc_z_learning(actions, observation)
        elif self._motion_policy == "reactive_evasion":
            self._set_acc_z_avoid(actions, observation)

        return np.asarray(actions, dtype=np.float64).reshape(-1, 4)

    def _set_acc_z_learning(self, actions: list[list[float]], observation: dict) -> None:
        """编码本机观测和任务上下文，选择离散机动动作并转成引擎动作。"""

        if self.launch_step < 0:
            return
        assert self._learning_encoder is not None
        assert self._learning_adapter is not None
        assert self._actor is not None
        self._learning_encoder.set_agent_index(*self._learning_identity())
        target_id = self._commander.target_id_for(self.entity_id)
        # 编码器的公开目标表使用索引，先将规划器的目标编号转换到该表位置。
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
        # Actor 只消费这架平台的编码；掩码描述该离散机动空间当前允许的动作。
        action = self._actor.select_action(
            encoded,
            self._learning_adapter.build_action_mask(),
        )
        # 训练模式才暂存推理输入；推理模式只把选定机动转换成引擎动作。
        if bool(getattr(self._actor, "training", False)):
            self._learning_transition = (encoded, int(action), self._learning_adapter.build_action_mask())
        engine_actions = self._learning_adapter.to_engine_action(int(action), self.entity_id)
        actions.extend(engine_actions.tolist())

    def on_feedback(self, observation: dict, action, reward: float, info: dict = None):
        """将已选择的学习动作与推进后的本机反馈组成一条经验，提交到示例队列。"""

        if self._learning_transition is None or self._actor is None:
            return
        encoded, selected_action, action_mask = self._learning_transition
        self._learning_transition = None
        if not bool(getattr(self._actor, "training", False)):
            return
        assert self._learning_encoder is not None
        # 仍有本机状态时编码下一观测；本机退出后使用同维零向量结束该经验。
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
        # 这里追加本机经验，Critic 所需全态由训练 runner 在步进后另行补齐。
        self._transition_sink(
            PolicyTransition(
                entity_id=self.entity_id,
                observation=encoded,
                action=selected_action,
                action_mask=action_mask,
                reward=float(reward),
                next_observation=next_observation,
                done=bool((info or {}).get("done", False) or not observation.get("self")),
            )
        )

    def _set_acc_z_avoid(self, actions, observation: dict):
        """根据接近本机的自爆虫观测执行一次横向加速、停止加速和天眼规则。"""

        # 当探测到自爆虫群时，进行机动规避
        if self.launch_step < 0 or (self.set_acc_z_z == 2 and self.sat_used):
            return

        self_info = observation["self"]
        if self_info is None or "detectInfo" not in self_info:
            return

        incoming = False
        for track in self_info.get("detectInfo", {}).values():
            if track.entity_type == 24000 and self.is_self_interceptor(track, self_info):
                incoming = True
                break

        if not incoming:
            return

        # 计数只在当前帧仍探测到朝本机飞来的自爆虫时推进。
        self.acc_start_step += 1

        old_status = self.set_acc_z_z
        if self.set_acc_z_z == 0:
            # 发现接近本机的自爆虫群后立即横向机动。
            actions.append([ACTION_SET_ACC_Z, self.entity_id, 1, 0])
            self.set_acc_z_z = 1
        elif self.set_acc_z_z == 1 and self.acc_start_step >= 10:
            # 加速一定帧数后，停止加速
            actions.append([ACTION_SET_ACC_Z, self.entity_id, 0, 0])
            self.set_acc_z_z = 2

        if old_status != self.set_acc_z_z:
            self.acc_start_step = 0

        # 规避规则可申请一次天眼，与任务层共用同一个 sat_used 标记。
        if self.sat_used:
            return
        actions.append([ACTION_USE_SAT, self.entity_id, 0, 0])
        self.sat_used = True

    def is_self_interceptor(self, target: DetectInfo, entity: dict) -> bool:
        """用自爆虫速度与指向本机的位置差夹角判断其是否朝本机飞来。"""
        position = target.pos
        velocity = target.vel
        if position is None or velocity is None:
            return False

        en_pos = entity["pos"]
        ox, oy, oz = en_pos["x"], en_pos["y"], en_pos["z"]
        px, py, pz = position.x, position.y, position.z
        vx, vy, vz = velocity.x, velocity.y, velocity.z
        dx, dy, dz = ox - px, oy - py, oz - pz
        dot = dx * vx + dy * vy + dz * vz
        norm1_sq = dx * dx + dy * dy + dz * dz
        norm2_sq = vx * vx + vy * vy + vz * vz
        if norm1_sq == 0.0 or norm2_sq == 0.0:
            return True  # 沿用 get_vector_angle 的零向量 0 度约定
        return dot > 0.0 and dot * dot > self._COS_20_SQ * norm1_sq * norm2_sq

    def on_reset(self):
        """清空当前回合的规划引用、机动状态、天眼标记与待提交经验。"""
        self._commander = None
        self._mailbox = None
        self.set_acc_z_z = 0
        self.acc_start_step = 0
        self.launch_step = -1
        self.sat_used = False
        self._learning_transition = None
