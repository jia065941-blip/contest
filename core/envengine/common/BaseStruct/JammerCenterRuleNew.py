# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/07/05 19:35:44.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .TriggerDistance import TriggerDistance
from .Vector3d import Vector3d


@dataclass_json
@dataclass
class JammerCenterRuleNew:
	# 是否执行
	isExecute: int = 0
	# 触发器距离
	mTriggerDistance: TriggerDistance = field(default_factory=TriggerDistance)
	# 载体ID
	bodyID: int = 0
	# GR目标ID
	jamTargetID: int = 0
	# GR实体类型
	jammerType: int = 0
	# GR实体数量
	jammerNum: int = 0
	# 载体节点相应GR类型的子GR实体ID集合
	jammerIdList: list[int] = field(default_factory=list[int])
	# GR模式枚举(0:无GR 1:压制GR 2:欺骗GR 3:假目标GR 4:角反GR 5:箔条GR 6:箔条冲淡GR 7:箔条质心GR 8:舰船轴对称“冲 4”GR 9:"DD轴对称“冲 4”GR" 10:红外诱饵GR 11:烟幕GR)
	jamMode: int = 0
	# GR概率（0-1）
	jamProbability: float = 0.0
	# 开始干扰时刻
	startJamTime: float = 0.0
	# 干扰时长
	jamTimeLength: float = 0.0
	# GR实体相对载体距离
	jammerRelDis: float = 0.0
	# GR实体相对载体位置（北天东）
	jammerRelPosNue: Vector3d = field(default_factory=Vector3d)
	# GR实体相对载体位置偏差（北天东）
	jammerRelPosNueDeviation: Vector3d = field(default_factory=Vector3d)
	# GR实体相对载体基础位置（北天东）
	jammerRelPosNueBasic: Vector3d = field(default_factory=Vector3d)
	# GR实体相对载体速度（北天东）
	jammerRelVelNue: Vector3d = field(default_factory=Vector3d)
