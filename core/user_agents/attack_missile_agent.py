# -*-coding:utf-8 -*-
import math

import numpy as np

from envengine.sdk.base_struct.Basic import Vector3d
from user_agents.base_agent import BaseAgent, AgentType

ACTION_SET_ACC_Z = 0  # 设置Z轴加速度
ACTION_USE_SAT = 3  # 使用卫星


class AttackMissileAgent(BaseAgent):
    """
    动作数组格式: np.array([a, b, c, ...], dtype=np.float64)

    横向加速度指令：(加速度值的 1 表示 20G)
    [ACTION_SET_ACC_Z, 实体id, 加速度值, 0]

    使用卫星指令：
    [ACTION_USE_SAT, 实体id, 0, 0]

    Attributes:
        launch_step: 发射时的帧数，为发射为 -1
    """

    def __init__(self, agent_id: int, entity_id: int, init_observation:dict, commander,
                 motion_policy: str = "reactive_evasion"):
        super().__init__(agent_id, entity_id, AgentType.AIRCRAFT, init_observation)
        self.commander = commander
        if motion_policy not in {"straight", "reactive_evasion"}:
            raise ValueError(f"Unsupported red motion policy: {motion_policy}")
        self.motion_policy = motion_policy
        self.set_acc_z_z = 0
        self.acc_start_step = 0
        self.launch_step = -1
        self.sat_used = False
        self.latest_observation: dict | None = None

    def set_observation(self, observation: dict) -> None:
        self.latest_observation = observation

    def get_action(self) -> np.array:
        """生成动作"""
        observation = self.latest_observation
        actions = []  # 存储多个动作

        step = observation["step"]

        self.commander.report(observation)
        launch = self.commander.action_for(self.entity_id, step)
        if launch is not None:
            actions.append(launch)
            self.launch_step = step

        if self.motion_policy == "reactive_evasion":
            self._set_acc_z_avoid(actions, observation)

        return np.array(actions, dtype=np.float64)

    def _set_acc_z_avoid(self, actions, observation: dict):
        """
        躲避拦截弹机动
        :param actions:
        :param observation:
        :return:
        """

        # 当探测到拦截弹时，进行机动规避
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
            # 发现来袭拦截弹后立即横向机动。
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

    def is_self_interceptor(self, target:dict, entity:dict)->bool:
        re_pos = Vector3d(entity["pos_ecf"]["x"] - target.pos_ecf.x, entity["pos_ecf"]["y"] - target.pos_ecf.y, entity["pos_ecf"]["z"] - target.pos_ecf.z)
        re_vel = target.vel_ecf
        return self.get_vector_angle(re_pos, re_vel) < 20

    def reset(self):
        """重置状态"""
        super().reset()
        self.set_acc_z_z = 0
        self.acc_start_step = 0
        self.launch_step = -1
        self.sat_used = False

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
