# -*- coding: utf-8 -*-#
# Version:4, Last Modified time: 2024/06/26 21:07:26.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class ChildSyncC:
	# 子物体位置同步
	cmd: PosVelAcc = field(default_factory=PosVelAcc)
