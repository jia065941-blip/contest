# -*-coding:utf-8 -*-
import logging
import random

import numpy as np

from envengine.sdk.base_struct.Basic import DetectInfo
from user_agents.base_agent import BaseAgent, AgentType

ACTION_SET_ACC_Z = 0  # 设置Z轴加速度
ACTION_LAUNCH = 1  # 发射
ACTION_CHANGE_TARGET = 2  # 修改目标点
ACTION_USE_SAT = 3 # 使用卫星


class AttackMissileAgent(BaseAgent):
    """
    动作数组格式: np.array([a, b, c, ...], dtype=np.float64)

    横向加速度指令：(加速度值的 1 表示 20G)
    [ACTION_SET_ACC_Z, 实体id, 加速度值, 0]

    发射指令：
    [ACTION_LAUNCH, 实体id, 经度, 纬度]

    更改目标指令：
    [ACTION_CHANGE_TARGET, 实体id, 经度, 纬度]

    使用卫星指令：
    [ACTION_USE_SAT, 实体id, 0, 0]

    Attributes:
        launch_step: 发射时的帧数，为发射为 -1
    """

    def __init__(self, agent_id: int, entity_id: int, init_observation:dict):
        super().__init__(agent_id, entity_id, AgentType.AIRCRAFT, init_observation)
        self.set_acc_z_z = 0
        self.acc_start_step = 0
        self.launch_step = -1
        self.sat_used = False

    def get_action(self, observation: dict) -> np.array:
        # print("observation:", observation)
        """生成动作"""
        # 这里可以根据observation做出更智能的决策
        num = random.randint(1, 200)
        actions = []  # 存储多个动作

        entity_type = observation["self"]["type"]
        step = observation["step"]

        # 发射
        self._launch(actions, entity_type, step)

        # 横向加速度
        self._set_acc_z_avoid(actions, observation)

        # 使用卫星
        self._use_satellite(actions, step)

        return np.array(actions, dtype=np.float64)

    def _launch(self, actions, entity_type:int, step):
        """
        高中性能弹 200帧后发射，低性能弹立即发射
        :param actions:
        :param entity_type:
        :param step:
        :return:
        """

        if self.launch_step >= 0:
            return

        if entity_type == 21000:
            # 高性能弹，在一定时间后发射
            if step > 50:
                target:dict = self.get_random_target()
                if target:
                    actions.append([ACTION_LAUNCH, self.entity_id, target["position"]["lon"], target["position"]["lat"]])
                    self.launch_step = step
        else:
                target:dict = self.get_random_target()
                if target:
                    actions.append([ACTION_LAUNCH, self.entity_id, target["position"]["lon"], target["position"]["lat"]])
                    self.launch_step = step

    def get_random_target(self)->dict:
        targets = []

        # 获取所有目标
        for entity_id, entity_info in self.init_observation["entities"].items():
            if entity_info["type"] == 9400:
                targets.append(entity_info)

        if targets:
            return random.choice(targets)
        else:
            return None

    def _set_acc_z(self, actions, entity_type, step):
        """
        横向加速度
        1、发射
        2、100 帧后，施加一个加速度
        3、105 停止加速，进行一定时间直线运动
        3、200 反向加速
        4、205 停止加速，并重设目标点

        :param actions:
        :param step:
        :return:
        """

        if entity_type != 21001:
            return

        if self.launch_step < 0:
            return

        # 横向加速状态机
        # set_acc_z_z 记录当前横向加速度状态
        # acc_start_step 记录当前横向加速度状态的开始时间
        self.acc_start_step += 1

        old_status = self.set_acc_z_z
        if self.set_acc_z_z == 0 and self.acc_start_step >= 100:
            # 发射 100 帧后，施加一个加速度
            actions.append([ACTION_SET_ACC_Z, self.entity_id, 1, 0])
            self.set_acc_z_z = 1
        elif self.set_acc_z_z == 1 and self.acc_start_step >= 5:
            # 加速 10 帧后，停止加速
            actions.append([ACTION_SET_ACC_Z, self.entity_id, 0, 0])
            self.set_acc_z_z = 2
        # elif self.set_acc_z_z == 2 and self.acc_start_step >= 110:
        #     # 停止加速 200 帧后，反向加速
        #     actions.append([ACTION_SET_ACC_Z, self.entity_id, -1, 0])
        #     self.set_acc_z_z = 3
        # elif self.set_acc_z_z == 3 and self.acc_start_step >= 2:
        #     # 反向加速 10 帧后，停止加速
        #     actions.append([ACTION_SET_ACC_Z, self.entity_id, 0, 0])
        #     self.set_acc_z_z = 4
        #     target = self.get_random_target()
        #     if target:
        #         actions.append([ACTION_CHANGE_TARGET, self.entity_id, target["position"]["lon"], target["position"]["lat"]])

        if old_status != self.set_acc_z_z:
            self.acc_start_step = 0
            logging.info(f"[进攻弹智能体] 对应实体{self.entity_id}, 横向加速度状态由{old_status}变为{self.set_acc_z_z}")

    def _set_acc_z_avoid(self, actions, observation: dict):
        """
        躲避拦截弹机动
        :param actions:
        :param observation:
        :return:
        """

        # 当探测到拦截弹时，进行机动规避
        interceptors = [ target for target in observation["self"]["detectInfo"].values() if target.entity_type==24000]

        if not interceptors:
            return

        self.acc_start_step += 1

        old_status = self.set_acc_z_z
        if self.set_acc_z_z == 0:
            # 发射 100 帧后，施加一个加速度
            actions.append([ACTION_SET_ACC_Z, self.entity_id, 1, 0])
            self.set_acc_z_z = 1
        elif self.set_acc_z_z == 1 and self.acc_start_step >= 10:
            # 加速一定帧数后，停止加速
            actions.append([ACTION_SET_ACC_Z, self.entity_id, 0, 0])
            self.set_acc_z_z = 2

        if old_status != self.set_acc_z_z:
            self.acc_start_step = 0

    def _use_satellite(self, actions, step):
        """
        使用卫星
        :param actions:
        :return:
        """
        if self.sat_used or self.launch_step < 0 or step - self.launch_step < 100:
            return
        actions.append([ACTION_USE_SAT, self.entity_id, 0, 0])
        self.sat_used = True

    def reset(self):
        """重置状态"""
        super().reset()
        self.set_acc_z_z = 0
        self.launch_step = -1
