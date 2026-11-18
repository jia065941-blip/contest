# -*- coding: utf-8 -*-#
# Version:9, Last Modified time: 2023/10/18 18:17:06.
# Created by Administrator on 2026/05/18 14:50:47.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class ArriveTracePointT:
	# 到达途径点事件
	event: ArriveTracePointTrigger = field(default_factory=ArriveTracePointTrigger)
