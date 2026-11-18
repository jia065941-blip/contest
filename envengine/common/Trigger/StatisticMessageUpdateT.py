# -*- coding: utf-8 -*-#
# Version:9, Last Modified time: 2024/05/22 17:11:22.
# Created by Administrator on 2026/05/18 14:50:47.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class StatisticMessageUpdateT:
	# 统计信息变动事件
	event: StatisticMessageUpdateTrigger = field(default_factory=StatisticMessageUpdateTrigger)
