# -*- coding: utf-8 -*-#
# Version:7, Last Modified time: 2023/10/18 18:19:13.
# Created by Administrator on 2026/05/18 14:50:47.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class DefendCenterInfoT:
	# 拦截指控触发信息事件
	event: DefendCenterInfoTrigger = field(default_factory=DefendCenterInfoTrigger)
