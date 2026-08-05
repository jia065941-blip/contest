# -*-coding:GBK -*-
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .Command import Command


@dataclass_json
@dataclass
class Task(object):
    # *循环步长
    cycleStep: int = 0

    # *循环次数
    cycleTimes: int = 0

    # *任务类型
    taskType: int = 0

    # *初次运行时间
    executorTime: int = 0

    # *携带的指令
    commands: list[Command] = field(default_factory=list)

