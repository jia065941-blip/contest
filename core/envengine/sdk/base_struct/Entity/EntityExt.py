# -*-coding:utf-8 -*-
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from envengine.sdk.base_struct.Entity.Entity import Entity


@dataclass_json
@dataclass
class EntityExt(object):
    # 基本属性
    entity: Entity = field(default_factory=Entity)
    # 额外属性
    external: str = ""
