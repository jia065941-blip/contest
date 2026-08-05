# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/10/14 18:37:57.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Rain:
	# 降雨条件0小，1中，2大
	rainConditon: int = 0
	# 降雨量
	rainFall: float = 0.0
	# 雨顶高度
	rainHeight: float = 0.0
	# 降雨起始距离
	rainStartDis: float = 0.0
	# 降雨终止距离
	rainStopDis: float = 0.0
