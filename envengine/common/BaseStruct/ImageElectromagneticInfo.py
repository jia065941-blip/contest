# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2024/10/15 02:29:38.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ImageElectromagneticInfo:
	# 目标编号
	targetCode: float = 0.0
	# 仿真类型
	simType: float = 0.0
	# 功率谱类型
	powerSpectrum: float = 0.0
	# 幅度谱类型
	amplitudeSpectrum: float = 0.0
	# 分布参数
	distrbution: list[float] = field(default_factory=list[float])
	# 区域类型标志
	areaType: float = 0.0
	# 区域风速
	areaWindSpeed: float = 0.0
