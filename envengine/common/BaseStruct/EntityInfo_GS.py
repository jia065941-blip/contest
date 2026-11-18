# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2024/10/15 00:22:01.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class EntityInfo_GS:
	# 实体编号
	id: int = 0
	# 实体类型
	entityType: int = 0
	# 实体中文名称
	entityName: str = ""
	# 实体唯一编号
	entityCode: str = ""
	# 父节点（实体）编号
	parentId: int = 0
	# 子节点ID数组
	childrenId: list[int] = field(default_factory=list[int])
	# 经度
	longitude: float = 0.0
	# 纬度
	latitude: float = 0.0
	# 高度
	height: float = 0.0
	# 地固系位置X
	positionX: float = 0.0
	# 地固系位置Y
	positionY: float = 0.0
	# 地固系位置Z
	positionZ: float = 0.0
	# 地固系速度X
	speedX: float = 0.0
	# 地固系速度Y
	speedY: float = 0.0
	# 地固系速度Z
	speedZ: float = 0.0
	# 偏航角
	yawAngle: float = 0.0
	# 俯仰角
	pitchAngle: float = 0.0
	# 滚转角
	rollAngle: float = 0.0
	# 安装位置X
	installPositionX: float = 0.0
	# 安装位置Y
	installPositionY: float = 0.0
	# 安装位置Z
	installPositionZ: float = 0.0
	# 工作频点
	carrierFreq: list[float] = field(default_factory=list[float])
	# 工作模式
	workMode: list[float] = field(default_factory=list[float])
	# 开机/释放时间
	bootTime: float = 0.0
