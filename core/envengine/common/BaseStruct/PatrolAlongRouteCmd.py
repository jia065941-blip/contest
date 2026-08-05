# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2024/05/30 14:35:43.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .Vector3d import Vector3d


@dataclass_json
@dataclass
class PatrolAlongRouteCmd:
	# 路径点集合
	routes: list[Vector3d] = field(default_factory=list[Vector3d])
	# 任务时长
	taskTime: float = 0.0
