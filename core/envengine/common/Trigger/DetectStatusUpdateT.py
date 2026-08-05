# -*- coding: utf-8 -*-#
# Version:13, Last Modified time: 2024/07/05 22:17:34.
# Created by Administrator on 2026/05/18 14:50:47.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class DetectStatusUpdateT:
	# 探测器探测状态更新传输事件
	event: DetectStatusUpdateTrigger = field(default_factory=DetectStatusUpdateTrigger)
