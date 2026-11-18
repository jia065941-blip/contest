# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2023/10/18 15:41:40.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class StartDetectCmd:
	# 方位指向
	sensorAzAngle: float = 0.0
	# 俯仰指向
	sensorElAngle: float = 0.0
	# 待探测目标类型集合
	targetTypes: list[int] = field(default_factory=list[int])
	# 待探测目标id集合
	targetIds: list[int] = field(default_factory=list[int])
