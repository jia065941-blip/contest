# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2024/07/05 19:07:06.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class GrCommandStartC:
	# 干扰规则
	cmd: JamCommandRuleStartCmd = field(default_factory=JamCommandRuleStartCmd)
