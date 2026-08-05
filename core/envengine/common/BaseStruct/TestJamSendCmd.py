# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/06/03 17:37:27.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .Vector3d import Vector3d


@dataclass_json
@dataclass
class TestJamSendCmd:
	# 干扰发射位置
	targetBodyEndPos: Vector3d = field(default_factory=Vector3d)
	# 需要释放的干扰类型
	jamtype: int = 0
	# 发射时间
	jamLunchTime: float = 0.0
	# 载体id
	bodyId: int = 0
