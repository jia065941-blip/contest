# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/06/13 23:08:20.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class JammerMap:
	# 初始距离（米）
	beginDistance: float = 0.0
	# 初始角度（°）
	beginAngle: float = 0.0
	# 环内数量
	loopPoints: int = 0
	# 距离增量（米）
	distanceIncrement: float = 0.0
	# 角度增量(°)
	angleIncrement: float = 0.0
