# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/10/14 18:37:39.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Atmosphere:
	# 仰角
	elevation: float = 0.0
	# 温度
	temperature: float = 0.0
	# 湿度
	wet: float = 0.0
	# 大气压强
	airPressure: float = 0.0
