# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/10/14 21:20:04.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Wind:
	# 风力等级，0-17级
	windScale: int = 0
	# 风速
	windSpeed: float = 0.0
	# 风向:0北风,1东北风,2东风，3东南风，4南风，5西南风，6西风，7西北风，8无风向
	windDirect: int = 0
