# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2024/07/03 23:16:25.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class SatelliteDb:
	# 历元时刻（秒）
	epochTime: float = 0.0
	# 半长轴（米）
	semimajorAxis: float = 0.0
	# 偏心率：（0~1）
	e: float = 0.0
	# 轨道倾角（度）（0~180）
	i: float = 0.0
	# 近地点辐角（度）（0~360）
	argOfPerigee: float = 0.0
	# 升交点赤经（度）（0~360）
	rann: float = 0.0
	# 平近点角（度）（0~360）
	meanAnom: float = 0.0
