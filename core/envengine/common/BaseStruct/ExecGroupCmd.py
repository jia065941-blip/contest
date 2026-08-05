# -*- coding: utf-8 -*-#
# Version:3, Last Modified time: 2024/05/30 14:11:26.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .Vector3d import Vector3d


@dataclass_json
@dataclass
class ExecGroupCmd:
	# 主机ID
	leaderId: int = 0
	# 是否编组
	isFormat: bool = False
	# 相对位置
	relativePos: Vector3d = field(default_factory=Vector3d)
