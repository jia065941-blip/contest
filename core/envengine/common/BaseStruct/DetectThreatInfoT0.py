# -*- coding: utf-8 -*-#
# Version:5, Last Modified time: 2023/10/18 14:00:52.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .Vector3d import Vector3d
from .DetectTargetT0 import DetectTargetT0


@dataclass_json
@dataclass
class DetectThreatInfoT0:
	# id
	id: int = 0
	# 位置
	pos: Vector3d = field(default_factory=Vector3d)
	# 类型
	type: int = 0
	# 携带的探测器信息
	radar: list[DetectTargetT0] = field(default_factory=list[DetectTargetT0])
