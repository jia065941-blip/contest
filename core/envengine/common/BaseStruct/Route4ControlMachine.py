# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2023/10/18 11:44:15.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Route4ControlMachine:
	# 路径点id
	way_index: int = 0
	# 位置
	pos_info: list[float] = field(default_factory=list[float])
	# 速度
	speed: float = 0.0
	# 飞行模式
	fly_mode: int = 0
	# 任务模式
	task_mode: int = 0
	# 路径属性
	way_attr: int = 0
