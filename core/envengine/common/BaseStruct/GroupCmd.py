# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2023/10/18 14:15:47.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .FollowerPara import FollowerPara


@dataclass_json
@dataclass
class GroupCmd:
	# 组ID
	groupId: int = 0
	# 主机ID
	leaderId: int = 0
	# 从机信息
	followers: list[FollowerPara] = field(default_factory=list[FollowerPara])
