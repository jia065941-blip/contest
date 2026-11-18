# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2023/10/18 17:28:50.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class StatisticMessageUpdateTrigger:
	# 人族单位出发数量
	redMissileLaunchNum: int = 0
	# 人族单位突破成功数量
	redMissileBreakoutNum: int = 0
	# 人族单位命中数量
	redMissileHitNum: int = 0
	# 自爆虫释放数量
	blueInterceptorLaunchNum: int = 0
	# 自爆虫成功命中数量
	blueInterceptorSuccessNum: int = 0
