# -*- coding: utf-8 -*-#
# Version:3, Last Modified time: 2023/10/18 14:00:01.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .Vector3d import Vector3d
from .MissileDeployInfo import MissileDeployInfo


@dataclass_json
@dataclass
class VehicleDeployInfo:
	# id
	id: int = 0
	# 位置（经纬高）
	pos: Vector3d = field(default_factory=Vector3d)
	# 类型
	type: int = 0
	# 血量
	hp: float = 0.0
	# 弹信息
	missiles: list[MissileDeployInfo] = field(default_factory=list[MissileDeployInfo])
