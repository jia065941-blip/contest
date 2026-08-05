# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2026/05/07 10:02:15.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .Parameters import Parameters


@dataclass_json
@dataclass
class JammingRule:
	# 距离
	Distance: float = 0.0
	# 距离阈值
	DistanceThreshold: float = 0.0
	# 干扰类型
	jammingType: int = 0
	# 个数
	Count: int = 0
	# 与船的距离
	DistanceFromShip: float = 0.0
	# 干扰参数
	Parameters: 'list[Parameters]' = field(default_factory=list[Parameters])
