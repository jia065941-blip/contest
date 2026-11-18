# -*- coding: utf-8 -*-#
# Version:3, Last Modified time: 2024/06/04 12:32:45.
# Created by Administrator on 2026/05/18 14:50:47.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class PowerStateChangeT:
	# 开关机状态修改
	event: PowerStateChangeTrigger = field(default_factory=PowerStateChangeTrigger)
