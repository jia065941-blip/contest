# -*- coding: utf-8 -*-#
# Version:3, Last Modified time: 2023/10/24 10:05:57.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .EntityInfo import EntityInfo


@dataclass_json
@dataclass
class DetectStatusUpdateTrigger:
	# 探测器
	sensor: EntityInfo = field(default_factory=EntityInfo)
	# 更新时间
	time: int = 0
	# 要探测的类型(可选)
	detectType: list[int] = field(default_factory=list[int])
	# 要探测的Id(可选)
	detectId: list[int] = field(default_factory=list[int])
	# 状态信息
	statusInfo: str = ""
