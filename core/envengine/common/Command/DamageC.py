# -*- coding: utf-8 -*-#
# Version:17, Last Modified time: 2026/01/20 16:48:08.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class DamageC:
	# 伤害指令
	cmd: DamageCmd = field(default_factory=DamageCmd)
