# -*- coding: utf-8 -*-#
# Version:3, Last Modified time: 2023/10/18 14:01:44.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class DamageInfo_Infrared:
	# id
	id: int = 0
	# 类型
	type: int = 0
	# 剩余血量
	hp: float = 0.0
