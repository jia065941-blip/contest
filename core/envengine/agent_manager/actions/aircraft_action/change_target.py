# -*- coding: utf-8 -*-#
import json
from typing import Optional
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from envengine.common import SimmerCommandType
from envengine.sdk.base_struct.Basic import Vector3d


@dataclass_json
@dataclass
class ChangeTargetAction(object):
    """
    导弹发射

    Attributes:
        executor_id : 执行者id
        commandType_id : 指令类型
        target : 目标点经纬高
    """

    executor_id: Optional[int] = -1
    commandType_id: int = SimmerCommandType.CHANGE_MISSILE_TARGET
    target: Vector3d = field(default_factory=lambda: Vector3d(0, 0, 0))
