# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2023/10/18 17:27:39.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .EntityInfo import EntityInfo
from .DetectedEntity import DetectedEntity


@dataclass_json
@dataclass
class SensorDetectUpdateTrigger:
	# 探测器
	sensor: EntityInfo = field(default_factory=EntityInfo)
	# 时间
	time: int = 0
	# 探测到的对象
	detectTarget: list[DetectedEntity] = field(default_factory=list[DetectedEntity])
