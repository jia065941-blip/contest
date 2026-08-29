# -*- coding: utf-8 -*-#
import json
from typing import Optional
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from envengine.common import SimmerCommandType
from envengine.sdk.base_struct.Basic import Vector3d


@dataclass_json
@dataclass
class UseSatelliteAction(object):
    """
    导弹发射

    Attributes:
        executor_id : 执行者id
        commandType_id : 指令类型
    """

    executor_id: Optional[int] = -1
    commandType_id: int = SimmerCommandType.EXECUTE_SATELLITE_DETECTION
    requested: bool = True
