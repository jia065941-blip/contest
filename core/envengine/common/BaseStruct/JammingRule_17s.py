# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2026/05/09 10:05:32.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .Parameters_jam import Parameters_jam


@dataclass_json
@dataclass
class JammingRule_17s:
	# 距离
	Distance: float = 0.0
	# 距离阈值
	DistanceThreshold: float = 0.0
	# 干扰类型
	jammingType: int = 0
	# 次数
	Count: int = 0
	# 离船距离
	DistanceFromShip: float = 0.0
	# 位置
	Position: str = ""
	# 参数
	Parameters: list[Parameters_jam] = field(default_factory=list[Parameters_jam])
