# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2023/10/18 11:41:04.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .Vector3d import Vector3d


@dataclass_json
@dataclass
class RoutePoint:
	# 是否到达
	isArrival: int = 0
	# 到达速度大小
	vel: float = 0.0
	# 位置(经纬高)
	pos: Vector3d = field(default_factory=Vector3d)
