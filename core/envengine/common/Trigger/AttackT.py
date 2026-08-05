# -*- coding: utf-8 -*-#
# Version:15, Last Modified time: 2023/11/11 14:18:37.
# Created by Administrator on 2026/05/18 14:50:47.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class AttackT:
	# 导弹发射事件
	event: AttackTrigger = field(default_factory=AttackTrigger)
