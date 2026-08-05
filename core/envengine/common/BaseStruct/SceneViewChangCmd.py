# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2024/06/18 23:27:24.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .Vector3d import Vector3d


@dataclass_json
@dataclass
class SceneViewChangCmd:
	# 相机位置
	cameraPosLLA: Vector3d = field(default_factory=Vector3d)
	# 相机偏角
	heading: float = 0.0
	# 相机倾角
	pitch: float = 0.0
