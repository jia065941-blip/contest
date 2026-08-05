# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/10/18 00:13:21.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .Vector3d import Vector3d


@dataclass_json
@dataclass
class ChagePoseCmd:
	# 经纬高
	lla: Vector3d = field(default_factory=Vector3d)
	# 姿态
	att: Vector3d = field(default_factory=Vector3d)
