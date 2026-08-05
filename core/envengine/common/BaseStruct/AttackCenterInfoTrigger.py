# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2023/10/18 17:28:47.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .BeforeShootInfo import BeforeShootInfo
from .FlyingInfo import FlyingInfo


@dataclass_json
@dataclass
class AttackCenterInfoTrigger:
	# 当前仿真时间戳
	timeStamp: int = 0
	# 射前数据
	beforeShootInfo: BeforeShootInfo = field(default_factory=BeforeShootInfo)
	# 飞行中数据
	flyingInfo: FlyingInfo = field(default_factory=FlyingInfo)
