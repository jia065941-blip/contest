# -*- coding: utf-8 -*-#
# Version:3, Last Modified time: 2023/10/18 14:00:23.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .Vector3d import Vector3d


@dataclass_json
@dataclass
class MissileTargetInfo:
	# id
	id: int = 0
	# 位置经纬高
	posLLA: Vector3d = field(default_factory=Vector3d)
	# ecf位置
	posEcf: Vector3d = field(default_factory=Vector3d)
	# ecf速度
	velEcf: Vector3d = field(default_factory=Vector3d)
	# 血量
	hp: float = 0.0
