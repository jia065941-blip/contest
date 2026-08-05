# -*- coding: utf-8 -*-#
# Version:5, Last Modified time: 2023/10/18 18:19:00.
# Created by Administrator on 2026/05/18 14:50:47.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class JamStartT:
	# 启动干扰事件
	event: JamStartTrigger = field(default_factory=JamStartTrigger)
