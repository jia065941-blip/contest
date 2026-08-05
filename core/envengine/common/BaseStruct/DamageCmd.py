# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2023/10/18 14:13:05.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class DamageCmd:
	# 伤害血量
	damagePoint: int = 0
