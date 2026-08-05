# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2024/06/19 19:43:10.
# Created by Administrator on 2026/05/18 14:50:47.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class NetEmitT:
	# 发射网捕
	event: AttackTrigger = field(default_factory=AttackTrigger)
