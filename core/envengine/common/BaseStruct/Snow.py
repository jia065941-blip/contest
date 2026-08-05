# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2024/10/14 21:43:06.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Snow:
	# 降雪条件,0-小,1-中,2-大
	snowCondition: int = 0
	# 降雪量
	snowFall: float = 0.0
	# 降雪起始距离
	snowStartDis: float = 0.0
	# 降雪终止距离
	snowStopDis: float = 0.0
