# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2023/11/07 15:26:34.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .JammerMap import JammerMap


@dataclass_json
@dataclass
class JamMode:
	# 干扰类型
	jammerType: int = 0
	# 干扰距离
	jamDistance: float = 0.0
	# 释放数量
	releaseCount: int = 0
	# 阵型
	jammerMap: JammerMap = field(default_factory=JammerMap)
	# 干扰样式
	jammerStyle: int = 0
