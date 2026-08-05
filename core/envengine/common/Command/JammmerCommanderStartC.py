# -*- coding: utf-8 -*-#
# Version:7, Last Modified time: 2024/07/05 17:38:45.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class JammmerCommanderStartC:
	# 规则信息及可用载体列表
	cmd: JammerCommanderStartCmd = field(default_factory=JammerCommanderStartCmd)
