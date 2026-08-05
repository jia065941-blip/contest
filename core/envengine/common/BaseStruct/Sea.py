# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/10/14 21:51:09.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Sea:
	# 海况等级，0-5级
	seaStateLevel: int = 0
	# 浪向角
	waveDirectionAngle: float = 0.0
	# 浪高
	waveHeight: float = 0.0
