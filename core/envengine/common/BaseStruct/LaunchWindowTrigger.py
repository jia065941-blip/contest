# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/07/02 21:28:19.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .LaunchWindow import LaunchWindow


@dataclass_json
@dataclass
class LaunchWindowTrigger:
	# 载体id
	carrierId: int = 0
	# 发射窗口信息，key为目标id
	launchWindowMap: dict[int, LaunchWindow] = field(default_factory=dict[int, LaunchWindow])
