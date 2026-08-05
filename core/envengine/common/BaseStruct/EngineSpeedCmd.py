# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/06/12 14:04:58.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class EngineSpeedCmd:
	# 引擎倍速
	speed: int = 0
