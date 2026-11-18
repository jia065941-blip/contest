# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/06/07 17:36:41.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .Vector3d import Vector3d


@dataclass_json
@dataclass
class MissileChangeRouteCmd:
	# 新航迹
	route: list[Vector3d] = field(default_factory=list[Vector3d])
