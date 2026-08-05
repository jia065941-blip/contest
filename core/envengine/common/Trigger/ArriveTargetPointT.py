# -*- coding: utf-8 -*-#
# Version:8, Last Modified time: 2023/10/18 18:17:12.
# Created by Administrator on 2026/05/18 14:50:47.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class ArriveTargetPointT:
	# 到达目标点事件
	event: ArriveTargetPonitTrigger = field(default_factory=ArriveTargetPonitTrigger)
