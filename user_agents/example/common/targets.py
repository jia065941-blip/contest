"""示例任务价值、公开目标初值与本机接收轨迹的融合。"""

from __future__ import annotations

from .contracts import Position
from .baselines import TargetPrior


# 目标价值使用比赛类型权重，供任务分配与接收情报共用。
OBJECTIVE_VALUE_BY_TYPE = {
    9400: 5.0,
    9600: 2.0,
    9500: 1.0,
}


def objective_value(entity_type: int) -> float:
    """将目标类型映射成任务分配价值，未列出的类型采用单位价值。"""
    return OBJECTIVE_VALUE_BY_TYPE.get(int(entity_type), 1.0)


# 初始化只读取公开的母巢和哨卫巢目录。
def initial_targets_from_observation(initial_observation: dict) -> tuple[TargetPrior, ...]:
    """从初始化公开目录提取母巢与哨卫巢，按实体编号排序。"""

    targets: list[TargetPrior] = []
    for raw_id, entity in initial_observation.get("entities", {}).items():
        location = entity.get("pos", {})
        entity_type = int(entity["type"])
        if entity_type not in (9400, 9600):
            continue
        targets.append(
            TargetPrior(
                entity_id=int(raw_id),
                entity_type=entity_type,
                position=Position(
                    x=float(location.get("x", 0.0)),
                    y=float(location.get("y", 0.0)),
                    z=float(location.get("z", 0.0)),
                ),
                value=objective_value(entity_type),
            )
        )
    return tuple(sorted(targets, key=lambda item: item.entity_id))


# 后续目标位置由各实例收到的探测快照更新，再由合法通信群内汇总。
class ReceivedTrackFusion:
    """用探测快照更新目标位置，并登记已收到的暗巢轨迹。

    公开探测记录提供编号、类型、运动状态和探测步号；目标存活标记沿用已有记录。
    """

    def __init__(self, targets: tuple[TargetPrior, ...]) -> None:
        """保存公开初值，并单独记录各目标已采纳的探测步号。"""
        self._targets = {item.entity_id: item for item in targets}
        self._detected_steps: dict[int, int] = {}

    @property
    def targets(self) -> tuple[TargetPrior, ...]:
        """以稳定编号顺序提供当前目标快照，供规划与可复现排序使用。"""
        return tuple(sorted(self._targets.values(), key=lambda item: item.entity_id))

    def ingest(self, observation: dict) -> bool:
        """读取本机 detectInfo，以探测步号更新目标位置，返回目录是否发生变化。"""

        changed = False
        for raw_id, track in (observation.get("self", {}).get("detectInfo") or {}).items():
            entity_id = int(self._field(track, "entity_id", raw_id))
            detected_step = int(self._field(track, "detected_step", observation.get("step", 0)))
            # 引擎已按毫秒时间融合；同一步的后续权威快照仍可更新位置。
            previous_step = self._detected_steps.get(entity_id)
            if previous_step is not None and detected_step < previous_step:
                continue
            current = self._targets.get(entity_id)
            # 公开目录之外，仅在实际探测记录包含 9500 类型和位置时加入新暗巢。
            if current is None:
                entity_type = int(self._field(track, "entity_type", -1))
                pos = self._field(track, "pos", None)
                if entity_type != 9500 or pos is None:
                    continue
                self._targets[entity_id] = TargetPrior(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    position=Position(
                        x=float(self._field(pos, "x", 0.0)),
                        y=float(self._field(pos, "y", 0.0)),
                        z=float(self._field(pos, "z", 0.0)),
                    ),
                    value=objective_value(entity_type),
                )
                self._detected_steps[entity_id] = detected_step
                changed = True
                continue
            # 已知目标沿用价值和存活标记，只接受同类型记录对其位置作更新。
            entity_type = int(self._field(track, "entity_type", current.entity_type))
            if entity_type != current.entity_type:
                continue
            pos = self._field(track, "pos", None)
            if pos is None:
                continue
            position = Position(
                x=float(self._field(pos, "x", current.position.x)),
                y=float(self._field(pos, "y", current.position.y)),
                z=float(self._field(pos, "z", current.position.z)),
            )
            self._detected_steps[entity_id] = detected_step
            # 即使位置未移动也推进探测步号；只有坐标变化才触发重规划事件。
            if position == current.position:
                continue
            self._targets[entity_id] = TargetPrior(
                entity_id=current.entity_id,
                entity_type=current.entity_type,
                position=position,
                value=current.value,
                alive=current.alive,
            )
            changed = True
        return changed

    @staticmethod
    def _field(value, name: str, default):
        """兼容字典与 SDK 对象两种探测记录表示。"""
        return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)
