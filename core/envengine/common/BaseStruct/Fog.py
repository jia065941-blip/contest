# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2024/10/14 21:42:55.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Fog:
	# 降雾条件,0-轻,1-浓
	fogCondition: int = 0
	# 雾起始距离
	fogStartDis: float = 0.0
	# 雾终止距离
	fogStopDis: float = 0.0
	# 雾液态水含量
	fogWaterContent: float = 0.0
	# 雾液态水温度
	cloudMistWaterContent: float = 0.0
