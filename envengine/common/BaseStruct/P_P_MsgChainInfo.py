# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/06/05 09:18:27.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class P_P_MsgChainInfo:
	# 开始id
	startId: int = 0
	# 终止点id
	endId: int = 0
	# 子网id
	groupId: int = 0
	# 是否激活
	isActive: bool = False
	# 起点父id
	startPid: int = 0
	# 终点父id
	endPid: int = 0
