# -*- coding: utf-8 -*-#
# Version:4, Last Modified time: 2024/06/25 19:09:23.
# Created by Administrator on 2026/05/18 14:50:47.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class EngineSpeedChangeT:
	# 引擎倍速控制
	event: EngineSpeedTrigger = field(default_factory=EngineSpeedTrigger)
