# -*- coding: utf-8 -*-#
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from envengine.common.BaseStruct import Vector3d


@dataclass_json
@dataclass(frozen=True)
class DetectInfo:
    """
    探测信息
    """
    # 探测信息来源
    detect_from: int = 0
    # 探测时间步
    step: int = 0
    # 实体id
    entity_id: int = 0
    # 实体类型
    entity_type: int = -1
    # 实体名称
    nameChn: str = ""
    # ecf位置
    pos: Vector3d = field(default_factory=Vector3d)
    # ecf 速度
    vel: Vector3d = field(default_factory=Vector3d)
