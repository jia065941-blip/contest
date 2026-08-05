# -*- coding: utf-8 -*-#
# Version:9, Last Modified time: 2024/06/25 19:02:17.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class PowerStateChangeC:
	# 电源状态切换指令
	cmd: PowerStateChangeCmd = field(default_factory=PowerStateChangeCmd)
