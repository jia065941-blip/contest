# -*-coding:utf-8 -*-
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from envengine.sdk.base_struct.Entity.EntityExt import EntityExt
from envengine.sdk.base_struct.Message import Task, Event, MapArea
from envengine.sdk.base_struct.Message.Area import Area


@dataclass_json
@dataclass
class Imagine(object):
    # *模拟步长
    simStep: int = 0

    # *DateTime
    simTime: float = 0

    # *模拟运行的总时长
    simEndLogicTime: float = 0

    # *模型列表
    entityList: list[EntityExt] = field(default_factory=list)

    # *事件列表
    eventList: list[Event] = field(default_factory=list)

    # *任务
    taskList: list[Task] = field(default_factory=list)

    # 红方区域
    redArea: Area = field(default=None)

    # 地图区域
    mapArea: MapArea = field(default=None)
