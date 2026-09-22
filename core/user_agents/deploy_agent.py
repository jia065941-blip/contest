# -*-coding:utf-8 -*-
import logging
import random
import time
import numpy as np

from user_agents.base_agent import BaseAgent, AgentType

ACTION_DEPLOY = 0
ACTION_COMPLETE = 1
ACTION_DEACTIVATE = 2


class DeployAgent(BaseAgent):
    """部署智能体（示例）"""

    def __init__(self, agent_id: int, entity_id: int, init_observation: dict,
                 deploy_coordinates: list[list[list[float]]], deploy_coordinatesHM: list[list[list[float]]],
                 unified_policy=None):
        super().__init__(agent_id, entity_id, AgentType.DEPLOY, init_observation)
        # 红方可部署区域
        self.deploy_coordinates: list[list[list[float]]] = deploy_coordinates
        self.deploy_coordinatesHM: list[list[list[float]]] = deploy_coordinatesHM
        self.index = 0
        self.latest_observation = None
        self.unified_policy = unified_policy

    def set_observation(self, observation: dict) -> None:
        self.latest_observation = observation

    def get_action(self) -> np.ndarray:
        """
        生成动作

        Returns:
            np.ndarray: 二维数组，每行格式为 [action_type, entity_id, lon, lat, alt]
            行数表示同时执行的动作数量
            当 action_type=1 时，只有一行，且 entity_id=-1
        """
        # random.seed(42)

        observation = self.latest_observation["entities"]
        keys = list(observation.keys())

        # 可以配置每次部署的实体数量
        batch_size = 10

        remaining = max(0, len(keys) - self.index)
        processed_count = min(batch_size, remaining)
        actions = []
        for i in range(processed_count):
            entity_id = keys[self.index + i]

            entity_type = observation[entity_id]["type"]
            coordinates = (
                self.deploy_coordinatesHM
                if entity_type in (21000, 21001) and self.deploy_coordinatesHM
                else self.deploy_coordinates
            )
            decision = None
            if self.unified_policy is not None and entity_type in (21000, 21001, 21002):
                polygon = coordinates[0]
                lons = [point[0] for point in polygon]
                lats = [point[1] for point in polygon]
                decision = self.unified_policy.select_deployment(
                    entity_id,
                    self.latest_observation,
                    (min(lons), max(lons), min(lats), max(lats)),
                )
            if decision is not None and not decision["presence"]:
                actions.append([ACTION_DEACTIVATE, entity_id, 0.0, 0.0, 0.0])
                continue
            # Trajectory MAPPO keeps not-yet-launched missiles out of the
            # physical scene. Their sampled initial coordinate is applied in
            # the same simulation step as the first LAUNCH command.
            if decision is not None and decision.get("defer_position", False):
                continue
            if decision is None:
                lon, lat = self.random_point_in_rect(coordinates)
            else:
                lon, lat = float(decision["lon"]), float(decision["lat"])

            if entity_type == 21002:
                actions.append([
                    ACTION_DEPLOY,
                    entity_id,
                    lon,
                    lat,
                    10000  # observation[entity_id]["position"]["alt"]
                ])
            else:
                actions.append([
                    ACTION_DEPLOY,
                    entity_id,
                    lon,
                    lat,
                    0
                ])

        self.index += processed_count

        # 没有更多实体，返回完成标志
        if not actions and self.index >= len(keys):
            return np.array([[ACTION_COMPLETE, -1, 0, 0, 0]], dtype=np.float64)
        time.sleep(0.1)
        if not actions:
            return np.empty((0, 5), dtype=np.float64)
        return np.array(actions, dtype=np.float64)

    def random_point_in_rect(self, coordinates):
        """
        在矩形区域内生成随机点

        Args:
            coordinates: [[[lon1, lat1], [lon2, lat2], [lon3, lat3], [lon4, lat4]]]

        Returns:
            (lon, lat): 随机经纬度
        """
        # 提取矩形的四个角
        polygon = coordinates[0]

        # 获取经纬度范围
        lons = [point[0] for point in polygon]
        lats = [point[1] for point in polygon]

        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        # 在范围内随机生成
        lon = random.uniform(min_lon, max_lon)
        lat = random.uniform(min_lat, max_lat)

        return lon, lat

    def reset(self):
        """重置状态"""
        super().reset()
        self.index = 0
