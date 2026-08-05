# -*- coding: utf-8 -*-#
# Version:3, Last Modified time: 2024/07/05 19:05:01.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class GrReleaseC:
	# 干扰释放
	cmd: ReleaseJammerCmd = field(default_factory=ReleaseJammerCmd)
