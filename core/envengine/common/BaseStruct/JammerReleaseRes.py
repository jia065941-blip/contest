# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/07/05 19:48:14.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .Vector3d import Vector3d


@dataclass_json
@dataclass
class JammerReleaseRes:
	# 唯一标识
	iD: int = 0
	# 唯一标识
	parentID: int = 0
	# 状态转枚举
	tag: int = 0
	# 模型类型转枚举
	entityType: int = 0
	# 当前时刻
	time: float = 0.0
	# 位置
	lla: Vector3d = field(default_factory=Vector3d)
	# 当前状态（实体不同，含义不同）
	status: int = 0
	# 地固系下位置
	posEcf: Vector3d = field(default_factory=Vector3d)
	# 地固系下速度
	velEcf: Vector3d = field(default_factory=Vector3d)
	# 姿态
	att: Vector3d = field(default_factory=Vector3d)
