# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/10/15 01:46:22.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .EnvironmentDecay import EnvironmentDecay
from .SeaClutter import SeaClutter
from .GroundClutter import GroundClutter


@dataclass_json
@dataclass
class TransmissionInfoSetCmd:
	# 环境衰减
	environmentDecay: EnvironmentDecay = field(default_factory=EnvironmentDecay)
	# 海杂波
	seaClutter: SeaClutter = field(default_factory=SeaClutter)
	# 地杂波
	groundClutter: GroundClutter = field(default_factory=GroundClutter)
