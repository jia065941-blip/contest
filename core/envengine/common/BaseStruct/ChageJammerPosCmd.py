# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2024/10/13 19:56:45.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .Vector3d import Vector3d


@dataclass_json
@dataclass
class ChageJammerPosCmd:
	# 经纬高
	lla: Vector3d = field(default_factory=Vector3d)
