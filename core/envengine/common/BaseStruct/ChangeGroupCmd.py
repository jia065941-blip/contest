# -*- coding: utf-8 -*-#
# Version:3, Last Modified time: 2024/06/03 16:55:58.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .FollowerPara import FollowerPara


@dataclass_json
@dataclass
class ChangeGroupCmd:
	# 组Id
	groupId: int = 0
	# 移除跟随者列表
	removeFollowList: list[int] = field(default_factory=list[int])
	# 添加/更新follow信息
	updateFollowList: list[FollowerPara] = field(default_factory=list[FollowerPara])
