# -*- coding: utf-8 -*-#
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from envengine.sdk.base_struct.Basic import Vector3d


@dataclass_json
@dataclass
class DetectInfo:
    """
    探测信息
    """
    # 探测信息来源
    detect_from: int = 0
    # 探测时间
    time: int = 0
    # 实体id
    entity_id: int = 0
    # 实体名称
    nameChn: str = ""
    # 实体位置
    lla: Vector3d = field(default_factory=Vector3d)
    # ecf位置
    pos_ecf: Vector3d = field(default_factory=Vector3d)
    # ecf 速度
    vel_ecf: Vector3d = field(default_factory=Vector3d)

