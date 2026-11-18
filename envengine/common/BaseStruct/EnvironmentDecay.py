# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2024/10/15 01:18:27.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class EnvironmentDecay:
	# 自由空间衰减
	spaceDecay: float = 0.0
	# 大气衰减
	airDecay: float = 0.0
	# 雨衰减
	rainDecay: float = 0.0
	# 雪衰减
	snowDecay: float = 0.0
	# 云衰减
	cloudDecay: float = 0.0
	# 雾衰减
	fogDecay: float = 0.0
	# 霾衰减
	hazeDecay: float = 0.0
	# 闪烁衰落
	flickerDecay: float = 0.0
	# 总衰减
	totalDecay: float = 0.0
