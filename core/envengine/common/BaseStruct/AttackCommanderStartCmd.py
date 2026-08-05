# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/06/19 17:18:35.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class AttackCommanderStartCmd:
	# 可用进攻实体列表
	avaliableEntityList: list[int] = field(default_factory=list[int])
	# 选用武器类型
	weaponType: int = 0
