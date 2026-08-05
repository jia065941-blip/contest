# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2023/10/18 17:27:52.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .EntityInfo import EntityInfo


@dataclass_json
@dataclass
class HitCompleteTrigger:
	# 武器
	weapon: EntityInfo = field(default_factory=EntityInfo)
	# 目标
	target: EntityInfo = field(default_factory=EntityInfo)
	# 更新时间
	time: float = 0.0
	# 毁伤点数
	damagePoint: int = 0
	# 毁伤是否成功
	success: bool = False
	# 脱靶量
	cep: float = 0.0
