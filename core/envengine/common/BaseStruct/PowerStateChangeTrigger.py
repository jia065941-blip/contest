# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/06/04 12:24:41.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .EntityInfo import EntityInfo


@dataclass_json
@dataclass
class PowerStateChangeTrigger:
	# 时间
	time: int = 0
	# 开关机状态
	powerState: int = 0
	# 实体
	entityInfo: EntityInfo = field(default_factory=EntityInfo)
