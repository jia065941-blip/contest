# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/07/09 23:39:24.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .EntityInfo import EntityInfo


@dataclass_json
@dataclass
class ActiveJammerPowerOnTrigger:
	# 有源干扰
	jammer: EntityInfo = field(default_factory=EntityInfo)
	# 目标id
	targetId: int = 0
