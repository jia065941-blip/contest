# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/06/19 19:12:19.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class LaunchWindow:
	# 最早发射时间
	minLaunchTime: float = 0.0
	# 最晚发射时间
	maxLaunchTime: float = 0.0
