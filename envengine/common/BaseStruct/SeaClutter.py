# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/10/15 01:25:42.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class SeaClutter:
	# 距离单元散射幅度
	landUnitScatterRadiance: float = 0.0
	# 平均散射系数
	avgScatterCoefficient: float = 0.0
	# 距离单元相位
	landUnitScatterPhase: float = 0.0
	# 海杂波在慢时间对应时刻
	timeInSeaClutterSlow: float = 0.0
	# 海杂波在快时间对应距离
	timeInSeaClutterQuick: float = 0.0
	# 频率
	frequency: float = 0.0
