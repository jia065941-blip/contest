# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2023/10/18 17:28:07.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .EntityInfo import EntityInfo
from .Vector3d import Vector3d
from .RoutePoint import RoutePoint


@dataclass_json
@dataclass
class ArriveTargetPonitTrigger:
	# 模型
	entity: EntityInfo = field(default_factory=EntityInfo)
	# 更新时间
	time: int = 0
	# 当前位置
	currentPos: Vector3d = field(default_factory=Vector3d)
	# 当前速度
	currentVel: Vector3d = field(default_factory=Vector3d)
	# 剩余路径点
	routePoint: list[RoutePoint] = field(default_factory=list[RoutePoint])
	# 目标点
	targetPoint: Vector3d = field(default_factory=Vector3d)
