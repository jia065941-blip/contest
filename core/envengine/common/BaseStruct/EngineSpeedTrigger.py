# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/06/12 14:05:32.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .EntityInfo import EntityInfo


@dataclass_json
@dataclass
class EngineSpeedTrigger:
	# 引擎速度
	speed: str = ""
	# 时间
	time: int = 0
	# 实体
	entityInfo: EntityInfo = field(default_factory=EntityInfo)
