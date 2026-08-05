# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/07/05 18:04:38.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .JammerCenterRuleNew import JammerCenterRuleNew


@dataclass_json
@dataclass
class JamCommandRuleStartCmd:
	# 干扰规则
	cmdRuleJam: list[JammerCenterRuleNew] = field(default_factory=list[JammerCenterRuleNew])
