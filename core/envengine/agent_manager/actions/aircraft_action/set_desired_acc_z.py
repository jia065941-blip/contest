# -*- coding: utf-8 -*-#
import json
from typing import Optional
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from envengine.common import SimmerCommandType
from envengine.sdk.base_struct.Basic import Vector3d


@dataclass_json
@dataclass
class SetDesiredAccZ(object):
    """
    设置侧向加速度
    """
    # 执行者Id
    executor_id: Optional[int] = -1

    # 指令类型Id
    commandType_id: int = SimmerCommandType.SET_DESIRED_ACC_Z

    # 侧向加速度
    acc_z: float = 0.0
