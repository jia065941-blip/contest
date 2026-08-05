# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2023/10/18 14:52:06.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class InterceptorInfoCmd:
	# 拦截波次
	waveCount: int = 0
	# 波次间隔
	waveGap: float = 0.0
	# 拦截策略
	interceptStrategy: int = 0
