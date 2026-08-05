# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/10/14 21:26:23.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Cloud:
	# 云条件,0-积云,1-层云,2-层积云,3-高层云,4-雨积云,5-卷云
	cloudCondition: int = 0
	# 云液态水含量
	cloudWaterContent: float = 0.0
	# 云液态水温度
	cloudWaterTemperature: float = 0.0
