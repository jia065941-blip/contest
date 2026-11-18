# -*- coding: utf-8 -*-#
# Version:3, Last Modified time: 2023/10/18 14:01:02.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .VehicleDeployInfo import VehicleDeployInfo
from .SpaceDetectTargetInfo import SpaceDetectTargetInfo
from .DetectThreatInfoT0 import DetectThreatInfoT0
from .IcptThreatInfoT0 import IcptThreatInfoT0


@dataclass_json
@dataclass
class BeforeShootInfo:
	# 红方部署数据
	redDeploy: list[VehicleDeployInfo] = field(default_factory=list[VehicleDeployInfo])
	# 空间探测目标列表
	spaceDetectTargets: list[SpaceDetectTargetInfo] = field(default_factory=list[SpaceDetectTargetInfo])
	# 探测威胁列表
	detectThreats: list[DetectThreatInfoT0] = field(default_factory=list[DetectThreatInfoT0])
	# 拦截威胁列表
	icptThreats: list[IcptThreatInfoT0] = field(default_factory=list[IcptThreatInfoT0])
