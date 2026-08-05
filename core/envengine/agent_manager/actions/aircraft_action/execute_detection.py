# -*- coding: utf-8 -*-#
from typing import Optional
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from envengine.common import SimmerCommandType
from envengine.sdk.base_struct.Basic import Vector3d


@dataclass_json
@dataclass
class ExecuteDetection:
    """
    执行探测
    """
    # 执行者Id
    executor_id: Optional[int] = -1

    # 指令类型Id
    commandType_id: int = SimmerCommandType.EXECUTE_DETECTION

    # 目标点
    detect_point: Vector3d = field(default_factory=lambda: Vector3d(0, 0, 0))
