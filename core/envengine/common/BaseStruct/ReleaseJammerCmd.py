# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/07/05 18:35:09.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .JammerReleaseRes import JammerReleaseRes


@dataclass_json
@dataclass
class ReleaseJammerCmd:
	# 释放状态结果
	associatedRes: JammerReleaseRes = field(default_factory=JammerReleaseRes)
