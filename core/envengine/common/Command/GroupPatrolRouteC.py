# -*- coding: utf-8 -*-#
# Version:3, Last Modified time: 2024/06/25 18:59:02.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class GroupPatrolRouteC:
	# 组号
	groupId: int = 0
	# 编组按路径巡逻指令
	cmd: PatrolAlongRouteCmd = field(default_factory=PatrolAlongRouteCmd)
