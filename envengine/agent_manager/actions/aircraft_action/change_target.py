# -*- coding: utf-8 -*-#
from typing import Optional
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from envengine.common import SimmerCommandType
from envengine.common.BaseStruct import Vector3d


@dataclass_json
@dataclass
class ChangeTargetAction(object):
    """
    移动与目标引导：按已知目标编号进攻，或设置地图范围内的引导航点。

    Attributes:
        executor_id : 执行者id
        commandType_id : 指令类型
        intent : track 或 waypoint
        target_id : track 指令的目标编号，由引擎解析合法目标位置
        target : waypoint 指令的航点（沿用仿真坐标格式）
    """

    executor_id: Optional[int] = -1
    commandType_id: int = SimmerCommandType.CHANGE_MISSILE_TARGET
    intent: str = "waypoint"
    target_id: Optional[int] = None
    target: Optional[Vector3d] = None
