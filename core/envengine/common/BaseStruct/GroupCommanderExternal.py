# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/06/03 18:13:21.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class GroupCommanderExternal:
	# 可用实体列表
	availableEntityList: list[int] = field(default_factory=list[int])
