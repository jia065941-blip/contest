# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/10/15 01:25:22.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class MissileDefenseElectromagneticInfo:
	# 目标编号
	targetCode: str = ""
	# 仿真类型，0：散射系数分布模型；1：功率谱；2：幅度谱
	simType: int = 0
	# 功率谱类型，0：高斯谱；1：柯西谱；2：立方谱
	powerSpectrum: int = 0
	# 幅度谱类型，0：瑞利分布；1：对数正太分布；2：威布尔分布；3：K分布
	amplitudeSpectrum: int = 0
	# 分布参数
	distrbutionParam: list[float] = field(default_factory=list[float])
	# 区域类型标志，0:农田；1:丘陵；2:高山；3:泥地；4:雪地；5:水泥地；6:草地等
	areaType: int = 0
	# 区域风速
	areaWindSpeed: float = 0.0
