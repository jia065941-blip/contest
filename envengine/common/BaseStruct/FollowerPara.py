# -*- coding: utf-8 -*-#
# Version:3, Last Modified time: 2023/10/18 11:42:37.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .Vector3d import Vector3d


@dataclass_json
@dataclass
class FollowerPara:
	# 跟随着id
	followerId: int = 0
	# 相对位置（前上右）
	relativePos: Vector3d = field(default_factory=Vector3d)
