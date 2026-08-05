# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/10/15 02:06:41.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class RadarElectromagneticInfo:
	# 目标编号
	targetCode: str = ""
	# 数据类型，0：窄带RCS数据；1：散射中心；2：同时计算窄带RCS数据和散射中心；3：宽带RCS数据；4：RCS起伏统计模型
	dataType: int = 0
	# RCS起伏统计模型标志，0:无起伏；1：斯威林起伏模型I；2:斯威林起伏模型II；3:斯威林起伏模型III；4:斯威林起伏模型IV
	rCSFluctModel: int = 0
	# 目标中心点位置经度
	targetCenterLon: float = 0.0
	# 目标中心点位置纬度
	targetCenterLat: float = 0.0
	# 目标中心点位置高度
	targetCenterHeight: float = 0.0
	# 目标初始姿态角X
	targetOriginAngleX: float = 0.0
	# 目标初始姿态角Y，相对目标本体坐标系
	targetOriginAngleY: float = 0.0
	# 目标初始姿态角Z，相对目标本体坐标系
	targetOriginAngleZ: float = 0.0
	# 目标运动模型所需输入，如舰船目标需要海况、浪向角、航速、航向等
	conditon: list[str] = field(default_factory=list[str])
