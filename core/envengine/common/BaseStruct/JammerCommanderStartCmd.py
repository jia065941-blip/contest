# -*- coding: utf-8 -*-#
# Version:3, Last Modified time: 2024/06/12 11:30:35.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .JammerCenterRule import JammerCenterRule


@dataclass_json
@dataclass
class JammerCommanderStartCmd:
	# 干扰规则列表
	jammerRulerList: list[JammerCenterRule] = field(default_factory=list[JammerCenterRule])
	# 可用干扰挂架列表
	availableCarriers: list[int] = field(default_factory=list[int])
