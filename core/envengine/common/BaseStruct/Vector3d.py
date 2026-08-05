# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/07/03 18:23:40.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Vector3d:
	# x
	x: float = 0.0
	# y
	y: float = 0.0
	# z
	z: float = 0.0
