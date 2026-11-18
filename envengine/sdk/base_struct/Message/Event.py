# -*-coding:GBK -*-
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import Optional
from .Command import Command


@dataclass_json
@dataclass
class Event(object):
    # *前序触发器Id, 用户主动触发时为空
    prevTriggerId: Optional[int] = -1

    # *前序触发类型
    prevTriggerTypeId: Optional[int] = -1

    # *触发id
    triggerId: int = 0

    # *触发类型
    triggerTypeId: int = 0

    # *触发器调用指令
    commandList: list[Command] = field(default_factory=list)
