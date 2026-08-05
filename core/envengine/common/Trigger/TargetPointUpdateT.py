# -*- coding: utf-8 -*-#
# Version:6, Last Modified time: 2023/10/18 18:18:18.
# Created by Administrator on 2026/05/18 14:50:47.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class TargetPointUpdateT:
	# 修改目标点事件
	event: TargetPointUpdateTrigger = field(default_factory=TargetPointUpdateTrigger)
