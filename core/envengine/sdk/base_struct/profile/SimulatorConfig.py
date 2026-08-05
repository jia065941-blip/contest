# -*-coding:utf-8 -*-
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class SimulatorConfig(object):
    # 模型类型
    entityType: int = 0

    # 模型类型子id
    entityTypeId: list[int] = field(default_factory=list)

    # 动态库名称
    dynamicLibraryPath: str = ""
