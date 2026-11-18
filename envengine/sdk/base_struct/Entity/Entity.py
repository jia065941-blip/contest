# -*-coding:utf-8 -*-
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import Optional, Dict, Any

from envengine.sdk.base_struct.Basic import DetectInfo
from envengine.common.BaseStruct import Vector3d


@dataclass_json
@dataclass
class Entity(object):
    """
    Attributes:
        id : 实体id
        entityType : 实体类型（21000-毁灭者，21001-突击者，21002-清道夫，9202-天眼，9400-母巢，24000-自爆虫群）
        posEcf : 位置
        att : 姿态

        stage: 运动状态: 默认 - 1, Launch = 1, Rise = 2, Cruise = 3, Terminal = 4, Explode = 5
    """

    # id
    id: int = 0
    # 父节点id
    parentId: Optional[int] = 0
    # 子节点id
    childrenId: list[int] = field(default_factory=list)
    # 类型
    entityType: int = 0
    # 型号id
    typeId: int = 0
    # 阵营id
    sideId: int = 0
    # 威胁等级
    threatLevel: int = 0
    # 当前时刻
    time: float = 0.0
    # 当前步长(毫秒)
    step: int = 0
    # 生命值
    survivePoints: float = 0.0
    # 最大生命值
    maxSurvivePoints: float = 0.0
    # 健康状态
    healthState: bool = True
    # 能量状态
    powerState: bool = True

    #运动状态: 默认 - 1, 1: 启动阶段，2：机动阶段，3：进攻阶段
    stage: float = -1
    # 位置
    posEcf: Vector3d = field(default_factory=Vector3d)
    # 速度
    velEcf: Vector3d = field(default_factory=Vector3d)
    # 姿态
    att: Vector3d = field(default_factory=Vector3d)
    # 是否可见
    isVisible: bool = False
    # 中文名
    nameChn: str = ""
    # 英文名
    nameEn: str = ""
    # 探测信息
    detectInfo: dict[int, DetectInfo] = field(default_factory=dict)
    # 通信范围内实体id
    commRangeInfo: Dict[int, Any] = field(default_factory=dict)
