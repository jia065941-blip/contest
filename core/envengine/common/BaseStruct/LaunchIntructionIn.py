# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/06/17 22:41:04.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class LaunchIntructionIn:
	# 待补充
	lcTheta: float = 0.0
	# 待补充
	lcGamma: float = 0.0
	# 待补充
	flagLaunchOrNot: float = 0.0
