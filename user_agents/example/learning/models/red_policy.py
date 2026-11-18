"""红方共享机动策略的观测编码、经验记录和动作适配。

Actor 选择向左、向右或停止横向机动。目标分配由通信群内的任务层完成；
MAPPO 的全局状态只供显式训练时的 Critic 使用。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping, Sequence

import numpy as np


# 固定目标槽按公开目标ID排序，不足时补零，保持不同实体的编码布局一致。
DEFAULT_TARGET_SLOTS = 5


class HighLevelAction(IntEnum):
    """模型输出的离散机动编号，由适配器映射为横向加速度指令。"""
    MANEUVER_LEFT = 0
    STOP_MANEUVER = 1
    MANEUVER_RIGHT = 2


ACTION_DIM = len(HighLevelAction)


@dataclass(frozen=True)
class _MapScale:
    """地图水平边界，用于位置、相对位移和距离的归一化。"""

    x_min: float
    y_min: float
    width: float
    height: float

    @classmethod
    def from_area(cls, area: Mapping[str, Any]) -> "_MapScale":
        values = [float(area[name]) for name in ("xMin", "xMax", "yMin", "yMax")]
        x_min, x_max, y_min, y_max = values
        if not all(np.isfinite(values)) or x_max <= x_min or y_max <= y_min:
            raise ValueError("map_area 必须包含有限、递增的 x/y 边界")
        return cls(x_min, y_min, x_max - x_min, y_max - y_min)

    @property
    def diagonal(self) -> float:
        return float(np.hypot(self.width, self.height))

    def normalized_x(self, x: float) -> float:
        return float(np.clip((x - self.x_min) / self.width, 0.0, 1.0))

    def normalized_y(self, y: float) -> float:
        return float(np.clip((y - self.y_min) / self.height, 0.0, 1.0))


def _motion(velocity: Mapping[str, Any]) -> tuple[float, float, float]:
    """从米每秒的三维速度求速率、竖直速度和从北向起算的水平航向。"""
    vx, vy, vz = (float(velocity.get(axis, 0.0)) for axis in ("x", "y", "z"))
    return float(np.sqrt(vx * vx + vy * vy + vz * vz)), vz, float(np.arctan2(vx, vy))


@dataclass(frozen=True)
class PolicyTransition:
    """一条本机经验，训练工具将各实体经验交给共享策略。

    observation/next_observation是固定长度本机编码，action为离散动作编号，
    action_mask记录采样时可用动作。MAPPO训练另外填写全局状态编码。
    reward来自本机反馈链路，其含义由调用环境提供。
    """

    entity_id: int
    observation: np.ndarray
    action: int
    action_mask: np.ndarray
    reward: float
    next_observation: np.ndarray
    done: bool
    global_state: np.ndarray | None = None
    next_global_state: np.ndarray | None = None


class SharedPolicy(ABC):
    """参数共享策略接口。

    select_action用于本机决策，其余钩子由示例经验装配层和训练runner调用。
    模型对象的共享、创建和释放由example的私有资源层管理。
    """

    @abstractmethod
    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        """在合法动作中选择一个高层动作。"""

    def observe(self, transition: PolicyTransition) -> None:
        """接收环境 transition；无在线学习需求的策略可以忽略。"""

    def reset_episode(self) -> None:
        """用户侧训练入口在回合开始调用一次，保留跨回合学习权重。"""

    def begin_environment_step(self, full_observation: Mapping[str, Any]) -> None:
        """训练入口在联合动作生成前保存全局状态。"""

    def end_environment_step(self, full_observation: Mapping[str, Any]) -> None:
        """训练入口在环境步进后保存下一全局状态。"""

    def finish_environment_step(self) -> None:
        """训练入口在全部本地 transition 提交后完成本步更新。"""

    def get_global_state_context(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """返回当前/下一全局编码；普通策略默认没有集中训练上下文。"""
        return None, None

    def save(self, path: str) -> None:
        """由支持持久化的算法实现模型保存。"""
        raise NotImplementedError("该策略没有实现模型保存")

    def load(self, path: str) -> None:
        """由支持持久化的算法实现checkpoint恢复。"""
        raise NotImplementedError("该策略没有实现模型加载")


class RandomMaskedPolicy(SharedPolicy):
    """仅使用NumPy的随机合法动作示例。

    用于验证观测、掩码、动作转换和经验回传链路。
    """

    def __init__(self, seed: int | None = None):
        """使用独立随机源，记录收到的经验条数。"""
        self._rng = np.random.default_rng(seed)
        self.transition_count = 0

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        """在掩码允许的动作中均匀采样；空集合时返回停止横向机动。"""
        del observation
        valid_actions = np.flatnonzero(action_mask)
        if valid_actions.size == 0:
            return int(HighLevelAction.STOP_MANEUVER)
        return int(self._rng.choice(valid_actions))

    def observe(self, transition: PolicyTransition) -> None:
        """只累计链路已提交的经验数量，不执行参数训练。"""
        del transition
        self.transition_count += 1


class ObservationEncoder:
    """将本机观测与本机执行状态编码为固定长度float32向量。

    默认布局为21维本机特征 + 5×12维公开目标槽 + 4维探测统计 = 85维。
    R9在末尾附加5维通信群任务上下文，形成90维输入。
    """

    # 这些常量同时确定向量布局和神经网络输入尺寸。
    SELF_FEATURES = 21
    TARGET_FEATURES = 12
    DETECTION_FEATURES = 4
    TASK_FEATURES = 5

    def __init__(
        self,
        init_observation: Mapping[str, Any],
        target_slots: int = DEFAULT_TARGET_SLOTS,
        max_steps: int = 1000,
        agent_index: int = 0,
        team_size: int = 1,
        hierarchical_task_context: bool = False,
    ):
        """固定目标目录及归一化参数，用内部登记序号生成个体特征。"""
        self.target_slots = target_slots
        self.max_steps = max(1, max_steps)
        self.map_scale = _MapScale.from_area(init_observation["map_area"])
        self.set_agent_index(agent_index, team_size)
        self.hierarchical_task_context = bool(hierarchical_task_context)
        # 只从初始化提供的母巢/哨卫目录创建目标槽；探测更新在每次encode中读取。
        # 槽顺序固定，因此current_target_index可以直接表示本机选中的目录项。
        entities = init_observation.get("entities", {})
        self.targets = sorted(
            (dict(value, entity_id=int(key)) for key, value in entities.items()
             if int(value.get("type", -1)) in (9400, 9600)),
            key=lambda item: item["entity_id"],
        )[:target_slots]

    @staticmethod
    def _field(value, name, default):
        """统一读取字典和SDK属性对象中的探测字段。"""
        return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)

    def set_agent_index(self, agent_index: int, team_size: int) -> None:
        """设置算法内部的零起始序号和登记规模，与仿真实体ID分离。"""
        agent_index, team_size = int(agent_index), int(team_size)
        if team_size < 1 or not 0 <= agent_index < team_size:
            raise ValueError("学习序号必须位于登记规模范围内")
        self.agent_index = agent_index
        self.team_size = team_size

    @property
    def observation_dim(self) -> int:
        """根据目标槽数和是否附带任务上下文计算实际输出长度。"""
        base_dim = (
            self.SELF_FEATURES
            + self.target_slots * self.TARGET_FEATURES
            + self.DETECTION_FEATURES
        )
        return base_dim + (self.TASK_FEATURES if self.hierarchical_task_context else 0)

    def encode(
        self,
        observation: Mapping[str, Any],
        *,
        launched: bool,
        launch_step: int,
        satellite_used: bool,
        maneuver_state: int,
        current_target_index: int | None = None,
        target_switch_elapsed: int = 0,
        task_context: Sequence[float] | None = None,
    ) -> np.ndarray:
        """编码本机状态、各目标相对关系、探测统计及可选任务上下文。

        launched、launch_step、satellite_used等参数由本机执行器提供；
        observation提供本机物理状态与已收到的探测/通信信息。
        """
        # 先读取物理状态与执行状态；时间按max_steps缩放，位置按地图水平范围归一化。
        self_info = observation["self"]
        position = self_info["pos"]
        x = float(position["x"])
        y = float(position["y"])
        step = int(observation.get("step", 0))
        entity_type = int(self_info.get("type", 0))
        elapsed = 0 if not launched else max(0, step - launch_step)
        velocity = self_info.get("vel") or {}
        speed, vertical_speed, heading = _motion(velocity)
        # 内部登记序号提供个体区分，与实体ID的数值和当前通信群成员数无关。
        agent_slot = np.clip(
            self.agent_index / max(self.team_size - 1, 1), 0.0, 1.0
        )
        role_index = self.agent_index % 3

        # 本机21维：时间/位置/血量、类型独热、出发与天眼状态、机动/切换状态、
        # 速度/航向及编号角色。位置单位为米，航向以sin/cos避免角度接缝。
        features = [
            np.clip(step / self.max_steps, 0.0, 1.0),
            self.map_scale.normalized_x(x),
            self.map_scale.normalized_y(y),
            np.clip(float(position.get("z", 0.0)) / 20000.0, -1.0, 1.0),
            np.clip(float(self_info.get("health", 0.0)) / 100.0, 0.0, 1.0),
            float(entity_type == 21000),
            float(entity_type == 21001),
            float(entity_type == 21002),
            float(launched),
            np.clip(elapsed / self.max_steps, 0.0, 1.0),
            float(satellite_used),
            np.clip(float(maneuver_state) / 2.0, -1.0, 1.0),
            np.clip(float(target_switch_elapsed) / 50.0, 0.0, 1.0),
            np.clip(speed / 1500.0, 0.0, 2.0),
            np.clip(vertical_speed / 500.0, -1.0, 1.0),
            np.sin(heading),
            np.cos(heading),
            agent_slot,
            float(role_index == 0),
            float(role_index == 1),
            float(role_index == 2),
        ]

        # 将已收到的探测记录按目标ID索引，供固定目标槽查询最新可用位置。
        detect_info = self_info.get("detectInfo") or {}
        detections = {
            int(self._field(info, "entity_id", key)): info
            for key, info in detect_info.items()
        }

        # 每个目标占12维；不存在的目录槽补零，有探测时采用探测坐标和记录年龄。
        for index in range(self.target_slots):
            if index >= len(self.targets):
                features.extend([0.0] * self.TARGET_FEATURES)
                continue
            target = self.targets[index]
            target_id = int(target["entity_id"])
            detection = detections.get(target_id)
            if detection is not None:
                pos = self._field(detection, "pos", None)
                target_position = {
                    "x": self._field(pos, "x", target["pos"]["x"]),
                    "y": self._field(pos, "y", target["pos"]["y"]),
                }
                detection_age = max(0, step - int(self._field(detection, "detected_step", step)))
            else:
                # 尚无本机探测时使用初始化公开位置，并将信息年龄编码为最大值。
                target_position = target["pos"]
                detection_age = self.max_steps
            entity_type = int(target.get("type", -1))
            delta_x = float(target_position["x"]) - x
            delta_y = float(target_position["y"]) - y
            # 平面位移直接以米计算；方位从北向起算，再减去本机航向。
            distance_m = float(np.hypot(delta_x, delta_y))
            bearing = float(np.arctan2(delta_x, delta_y))
            relative_bearing = bearing - heading
            # 目标12维：相对位置/距离、方位、探测标志/年龄、目标类型、
            # 当前目标标志，以及相对本机航向的sin/cos。
            features.extend([
                np.clip(delta_x / self.map_scale.width, -1.0, 1.0),
                np.clip(delta_y / self.map_scale.height, -1.0, 1.0),
                np.clip(distance_m / self.map_scale.diagonal, 0.0, 1.0),
                np.sin(bearing),
                np.cos(bearing),
                float(detection is not None),
                np.clip(detection_age / self.max_steps, 0.0, 1.0),
                float(entity_type == 9400),
                float(entity_type == 9600),
                float(current_target_index == index),
                np.sin(relative_bearing),
                np.cos(relative_bearing),
            ])

        # 末尾4维概括探测总量、自爆虫/暗巢数量和本机通信列表长度。
        detected_entity_types = [int(self._field(info, "entity_type", -1)) for info in detect_info.values()]
        comm_count = len(self_info.get("commRangeInfo") or [])
        features.extend([
            np.clip(len(detect_info) / 200.0, 0.0, 1.0),
            np.clip(sum(et == 24000 for et in detected_entity_types) / 200.0, 0.0, 1.0),
            np.clip(sum(et == 9500 for et in detected_entity_types) / 10.0, 0.0, 1.0),
            np.clip(comm_count / 64.0, 0.0, 1.0),
        ])
        # R9任务层传入最多5维上下文；不足补零，保证模型输入尺寸不随任务变化。
        if self.hierarchical_task_context:
            context = tuple(task_context or ())[:self.TASK_FEATURES]
            features.extend(context)
            features.extend([0.0] * (self.TASK_FEATURES - len(context)))
        return np.asarray(features, dtype=np.float32)


class GlobalStateEncoder:
    """将训练runner显式传入的完整态势编码为Critic状态。

    默认70维：1维进度、3×5维红方统计、5×3维蓝方统计、9维红方聚集状态、
    5×6维关键目标槽。训练算法将该向量与对应本机编码拼接后估值。
    """

    # 统计按实体类型组织，实体总数变化时仍保持相同维度。
    RED_TYPES = (21000, 21001, 21002)
    BLUE_TYPES = (9400, 9600, 44000, 24000, 9500)
    EXTRA_RED_FEATURES = 9

    def __init__(self, max_steps: int = 1000, target_slots: int = DEFAULT_TARGET_SLOTS):
        """保存时间归一化尺度和固定关键目标槽数。"""
        self.max_steps = max(1, int(max_steps))
        self.target_slots = int(target_slots)

    @property
    def state_dim(self) -> int:
        """根据固定类型分组、目标槽数和额外统计计算Critic编码尺寸。"""
        return (
            1
            + len(self.RED_TYPES) * 5
            + len(self.BLUE_TYPES) * 3
            + self.target_slots * 6
            + self.EXTRA_RED_FEATURES
        )

    @staticmethod
    def _mean(items, getter):
        """计算指定标量字段的均值，空分组返回零。"""
        values = [float(getter(item)) for item in items]
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _distance_km(first_position, second_position):
        """计算两个米制位置的水平距离，返回千米。"""
        dx = float(first_position["x"]) - float(second_position["x"])
        dy = float(first_position["y"]) - float(second_position["y"])
        return float(np.hypot(dx, dy) / 1000.0)

    def encode(self, observation: Mapping[str, Any]) -> np.ndarray:
        """从完整实体快照统计数量、存活、血量、位置与目标接近程度。"""
        scale = _MapScale.from_area(observation["map_area"])
        entities_map = observation.get("entities") or {}
        entities = [dict(item, entity_id=int(entity_id)) for entity_id, item in entities_map.items()]
        features = [np.clip(float(observation.get("step", 0)) / self.max_steps, 0.0, 1.0)]

        # 每种红方类型5维：存活比例、全组均血量，以及存活实体的平均三维位置。
        for entity_type in self.RED_TYPES:
            group = [item for item in entities if int(item.get("type", 0)) == entity_type]
            alive = [item for item in group if float(item.get("health", 0.0)) > 0]
            features.extend([
                len(alive) / max(len(group), 1),
                np.clip(self._mean(group, lambda x: x.get("health", 0.0)) / 100.0, 0.0, 1.0),
                scale.normalized_x(self._mean(alive, lambda x: x["pos"]["x"])) if alive else 0.0,
                scale.normalized_y(self._mean(alive, lambda x: x["pos"]["y"])) if alive else 0.0,
                np.clip(self._mean(alive, lambda x: x.get("pos", {}).get("z", 0.0)) / 20000.0, -1.0, 1.0),
            ])

        # 每种蓝方类型3维：数量缩放、存活比例和全组平均血量。
        for entity_type in self.BLUE_TYPES:
            group = [item for item in entities if int(item.get("type", 0)) == entity_type]
            alive_count = sum(float(item.get("health", 0.0)) > 0 for item in group)
            features.extend([
                np.clip(len(group) / 200.0, 0.0, 1.0),
                alive_count / max(len(group), 1),
                np.clip(self._mean(group, lambda x: x.get("health", 0.0)) / 100.0, 0.0, 1.0),
            ])

        # 汇总存活红方的速度、空间离散程度及其到存活母巢的最近距离。
        alive_red = [
            item for item in entities
            if int(item.get("type", 0)) in self.RED_TYPES and float(item.get("health", 0.0)) > 0
        ]
        key_targets = [
            item for item in entities
            if int(item.get("type", 0)) == 9400
            and float(item.get("health", 0.0)) > 0
        ]
        nearest_distances = [
            min(
                self._distance_km(item.get("pos", {}), target.get("pos", {}))
                for target in key_targets
            )
            for item in alive_red
        ] if key_targets else []
        speeds = [_motion(item.get("vel") or {})[0] for item in alive_red]
        x_values = [float(item.get("pos", {}).get("x", 0.0)) for item in alive_red]
        y_values = [float(item.get("pos", {}).get("y", 0.0)) for item in alive_red]
        altitudes = [float(item.get("pos", {}).get("z", 0.0)) for item in alive_red]
        # 9维额外统计依次为速度均值/标准差、三维位置标准差、
        # 最近目标距离的最小值/均值/标准差，以及速度大于1的实体比例。
        features.extend([
            np.clip(float(np.mean(speeds)) / 1500.0, 0.0, 2.0) if speeds else 0.0,
            np.clip(float(np.std(speeds)) / 1500.0, 0.0, 1.0) if speeds else 0.0,
            np.clip(float(np.std(x_values)) / scale.width, 0.0, 1.0) if x_values else 0.0,
            np.clip(float(np.std(y_values)) / scale.height, 0.0, 1.0) if y_values else 0.0,
            np.clip(float(np.std(altitudes)) / 20000.0, 0.0, 1.0) if altitudes else 0.0,
            np.clip(min(nearest_distances) * 1000.0 / scale.diagonal, 0.0, 2.0) if nearest_distances else 0.0,
            np.clip(float(np.mean(nearest_distances)) * 1000.0 / scale.diagonal, 0.0, 2.0) if nearest_distances else 0.0,
            np.clip(float(np.std(nearest_distances)) * 1000.0 / scale.diagonal, 0.0, 1.0) if nearest_distances else 0.0,
            sum(speed > 1.0 for speed in speeds) / max(len(alive_red), 1),
        ])

        # 固定目标槽按ID排序，每槽6维保存位置、血量、存活与母巢类型标志。
        targets = sorted(
            [item for item in entities if int(item.get("type", 0)) in [9400, 9600]],
            key=lambda item: item["entity_id"],
        )[:self.target_slots]
        for index in range(self.target_slots):
            if index >= len(targets):
                features.extend([0.0] * 6)
                continue
            target = targets[index]
            position = target.get("pos", {})
            entity_type = int(target.get("type", 0))
            features.extend([
                scale.normalized_x(float(position["x"])),
                scale.normalized_y(float(position["y"])),
                np.clip(float(position.get("z", 0.0)) / 20000.0, -1.0, 1.0),
                np.clip(float(target.get("health", 0.0)) / 100.0, 0.0, 1.0),
                float(float(target.get("health", 0.0)) > 0),
                float(entity_type == 9400),
            ])
        return np.asarray(features, dtype=np.float32)


class LearningActionAdapter:
    """构造动作掩码，并将高层动作翻译成引擎四元动作。"""

    # 对外机动动作码为 3；模型的离散分类编号仍使用 HighLevelAction 的 0/1/2。
    ACTION_SET_ACC_Z = 3

    def build_action_mask(self) -> np.ndarray:
        """当前机动层同时开放左转、停止和右转，返回长度3的布尔掩码。"""
        return np.ones(ACTION_DIM, dtype=np.bool_)

    def to_engine_action(self, action: int, entity_id: int) -> np.ndarray:
        """将离散动作映射为[动作类型, 本机ID, 横向值, 0]的1×4数组。"""
        # 模型编号0/1/2依次对应横向值-1/0/+1，交给引擎解释具体机动。
        action = int(action)
        if action == HighLevelAction.MANEUVER_LEFT:
            return np.asarray([[self.ACTION_SET_ACC_Z, entity_id, -1.0, 0.0]], dtype=np.float64)
        if action == HighLevelAction.MANEUVER_RIGHT:
            return np.asarray([[self.ACTION_SET_ACC_Z, entity_id, 1.0, 0.0]], dtype=np.float64)
        if action == HighLevelAction.STOP_MANEUVER:
            return np.asarray([[self.ACTION_SET_ACC_Z, entity_id, 0.0, 0.0]], dtype=np.float64)
        raise ValueError(f"未知的高层动作: {action}")
