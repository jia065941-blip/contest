# -*-coding:utf-8 -*-
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from typing import Optional
from envengine.sdk.base_struct.profile.SimulatorConfig import SimulatorConfig


@dataclass_json
@dataclass
class Environment(object):
    # *MQTTip
    mqttIp: str = ""

    # *域id
    domain: int = 0

    # *分域
    subDomain: str = ""

    # *当前场景工程名称
    projectName: str = ""

    # *引擎名称
    engineName: str = ""

    # *引擎id
    engineId: int = 0

    # *引擎类型
    role: Optional[int] = 0

    # *父节点id
    parentId: Optional[int] = -1

    # *子节点id
    childrenId: list[int] = field(default_factory=list)

    # *动态库属性
    dynamicLibraryConfigs: list[SimulatorConfig] = field(default_factory=list)
