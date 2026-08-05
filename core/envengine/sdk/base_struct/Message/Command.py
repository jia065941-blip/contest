# -*-coding:utf-8 -*-
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Optional, Any


@dataclass_json
@dataclass
class Command(object):
    """
    Attributes:
        prevTriggerId : 触发器Id, 用户主动触发时为空
        prevTriggerTypeId : 前序触发类型
        executorId : 调用器Id
        commandTypeId : 指令类型Id
        commandAttributes : 指令传参
    """
    # *触发器Id, 用户主动触发时为空
    prevTriggerId: Optional[int] = -1

    # *前序触发类型
    prevTriggerTypeId: Optional[int] = -1

    # *调用器Id
    executorId: Optional[int] = -1

    # *指令类型Id
    commandTypeId: int = 0

    # *指令传参
    commandAttributes: Any = None
