# -*- coding: utf-8 -*-#
# Version:3, Last Modified time: 2023/10/18 14:00:43.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class DetectTargetT0:
	# id
	id: int = 0
	# 类型
	type: int = 0
	# 探测器最远探测距离
	detectDistance: int = 0
	# 血量
	hp: float = 0.0
