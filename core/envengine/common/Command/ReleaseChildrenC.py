# -*- coding: utf-8 -*-#
# Version:4, Last Modified time: 2024/06/26 00:50:42.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class ReleaseChildrenC:
	# 释放子物体
	cmd: ReleaseChildrenCmd = field(default_factory=ReleaseChildrenCmd)
