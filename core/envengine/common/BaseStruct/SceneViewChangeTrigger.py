# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/06/18 23:14:05.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .EntityInfo import EntityInfo
from .Vector3d import Vector3d


@dataclass_json
@dataclass
class SceneViewChangeTrigger:
	# 时间
	time: int = 0
	# 实体信息
	entity: EntityInfo = field(default_factory=EntityInfo)
	# 相机位置
	viewPointPosLLA: Vector3d = field(default_factory=Vector3d)
	# 相机偏角
	heading: float = 0.0
	# 相机倾角
	pitch: float = 0.0
