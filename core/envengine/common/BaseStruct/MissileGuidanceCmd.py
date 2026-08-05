# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2023/10/18 15:31:39.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .Vector3d import Vector3d


@dataclass_json
@dataclass
class MissileGuidanceCmd:
	# 目标ID
	targetId: int = 0
	# 目标位置（经纬高）
	targetPos: Vector3d = field(default_factory=Vector3d)
	# 目标预测速度（NUE）
	targetVel: Vector3d = field(default_factory=Vector3d)
