# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2023/10/18 17:27:54.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .EntityInfo import EntityInfo


@dataclass_json
@dataclass
class DestroyTrigger:
	# 被摧毁模型
	entity: EntityInfo = field(default_factory=EntityInfo)
	# 伤害来源
	destroySrc: EntityInfo = field(default_factory=EntityInfo)
	# 更新时间
	time: float = 0.0
	# 最后一击伤害
	damagePoint: int = 0
