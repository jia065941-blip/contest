# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2023/10/18 17:27:27.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .EntityInfo import EntityInfo
from .Vector3d import Vector3d


@dataclass_json
@dataclass
class AttackTrigger:
	# 武器
	weapon: EntityInfo = field(default_factory=EntityInfo)
	# 更新时间
	time: int = 0
	# 目标ID
	targetId: int = 0
	# 目标位置
	targetPos: Vector3d = field(default_factory=Vector3d)
