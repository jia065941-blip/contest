# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/10/15 01:44:16.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class AircraftElectromagneticInfo:
	# 目标编号
	targetCode: float = 0.0
	# 数据类型
	dataType: float = 0.0
	# RCS起伏统计模型标志
	rCSFluctModel: float = 0.0
	# 目标中心点位置经度
	targetCenterLon: float = 0.0
	# 目标中心点位置纬度
	targetCenterLat: float = 0.0
	# 目标中心点位置高度
	targetCenterHeight: float = 0.0
	# 目标初始姿态角X
	targetOriginAngleX: float = 0.0
	# 目标初始姿态角Y
	targetOriginAngleY: float = 0.0
	# 目标初始姿态角Z
	targetOriginAngleZ: float = 0.0
	# 目标运动模型所需输入，如舰船目标需要海况、浪向角、航速、航向等
	conditon: list[str] = field(default_factory=list[str])
