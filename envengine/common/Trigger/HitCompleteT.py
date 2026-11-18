# -*- coding: utf-8 -*-#
# Version:7, Last Modified time: 2023/10/18 18:14:41.
# Created by Administrator on 2026/05/18 14:50:47.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class HitCompleteT:
	# 打击结束事件
	event: HitCompleteTrigger = field(default_factory=HitCompleteTrigger)
