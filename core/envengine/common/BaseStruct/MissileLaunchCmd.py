# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2023/10/18 15:27:23.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .Vector3d import Vector3d


@dataclass_json
@dataclass
class MissileLaunchCmd:
	# 载体
	carrierId: int = 0
	# 目标
	targetId: int = 0
	# 武器类型
	missileType: int = 0
	# 武器发射时间
	time: float = 0.0
	# 目标位置
	targetPos: Vector3d = field(default_factory=Vector3d)
	# 目标速度
	targetVel: Vector3d = field(default_factory=Vector3d)
	# RoutePoint
	routePoint: list[Vector3d] = field(default_factory=list[Vector3d])
	# 发射数量
	launchCount: int = 0
	# 是否编组
	isFomate: bool = False
