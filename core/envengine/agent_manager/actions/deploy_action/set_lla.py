# -*- coding: utf-8 -*-#
from typing import Optional
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from envengine.common import SimmerCommandType
from envengine.sdk.base_struct.Basic import Vector3d


@dataclass_json
@dataclass
class SetLLA:
    """
    设置位置
    """
    # 执行者Id
    executor_id: Optional[int] = -1

    # 指令类型Id
    commandType_id: int = SimmerCommandType.DEPLOY

    # 位置
    lla: Vector3d = field(default_factory=lambda: Vector3d(0, 0, 0))
