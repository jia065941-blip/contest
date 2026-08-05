# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2024/06/12 14:45:06.
# Created by Administrator on 2026/05/18 14:50:47.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class SensorScanUpdateT:
	# 导引头扫描更新
	event: SensorDetectUpdateTrigger = field(default_factory=SensorDetectUpdateTrigger)
