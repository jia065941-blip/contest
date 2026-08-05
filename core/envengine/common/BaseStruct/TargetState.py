# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2026/05/07 15:04:08.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class TargetState:
	# 类型
	Type: str = ""
	# 目标类型
	TargetType: str = ""
	# 经度
	Longitude: float = 0.0
	# 纬度
	Latitude: float = 0.0
	# 高度
	Altitude: float = 0.0
	# 距离
	Distance: float = 0.0
	# 方位角
	Azimuth: float = 0.0
	# 海拔
	Elevation: float = 0.0
	# 识别结果
	RecognitionResult: str = ""
