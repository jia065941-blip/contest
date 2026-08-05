# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2023/11/18 16:57:29.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .Target import Target


@dataclass_json
@dataclass
class TargetList:
	# 目标数组
	targetList: list[Target] = field(default_factory=list[Target])
