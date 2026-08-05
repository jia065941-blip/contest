# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2023/11/07 15:29:04.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .JamMode import JamMode


@dataclass_json
@dataclass
class JammerCenterRule:
	# 干扰方式
	jamMode: list[JamMode] = field(default_factory=list[JamMode])
	# 干扰目标类型
	type: int = 0
