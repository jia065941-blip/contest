# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/10/15 01:28:26.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class GroundClutter:
	# 仿真类型
	simType: float = 0.0
	# 地面散射系数分布点
	landScatterCoefficientDistributionPoint: float = 0.0
	# 地面单元散射幅度，地面散射总强度
	landUnitScatterRadiance: float = 0.0
	# 平均散射系数
	avgScatterCoefficient: float = 0.0
	# 地面单元散射相位
	landUnitScatterPhase: float = 0.0
	# 地面单元中心位置经度
	landUnitPosLon: float = 0.0
	# 地面单元中心位置纬度
	landUnitPosLat: float = 0.0
	# 地面单元中心位置高度
	landUnitPosHeight: float = 0.0
	# 频率
	frequency: float = 0.0
