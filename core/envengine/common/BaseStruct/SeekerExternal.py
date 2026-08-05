# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/06/04 15:08:43.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class SeekerExternal:
	# ID
	id: int = 0
	# 英文名称
	nameEn: str = ""
	# 中文名称
	nameChn: str = ""
	# 最大生命值
	maxHitPoints: float = 0.0
	# 轴角
	axialAngle: float = 0.0
	# 径向角
	radialAngle: float = 0.0
	# 扫描周期
	scanPeriod: float = 0.0
	# 分辨率
	resolution: float = 0.0
	# 型号对应关系
	typeIdIndex: int = 0
	# 最大探测距离
	maxDistance: float = 0.0
