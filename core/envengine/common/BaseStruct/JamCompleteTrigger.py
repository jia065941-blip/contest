# -*- coding: utf-8 -*-#
# Version:3, Last Modified time: 2023/10/18 17:28:38.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .EntityInfo import EntityInfo


@dataclass_json
@dataclass
class JamCompleteTrigger:
	# 干扰源
	source: EntityInfo = field(default_factory=EntityInfo)
	# 目标
	target: EntityInfo = field(default_factory=EntityInfo)
	# 更新时间
	time: int = 0
	# 干扰源状态
	sourceStatus: int = 0
	# 干扰是否成功
	success: bool = False
