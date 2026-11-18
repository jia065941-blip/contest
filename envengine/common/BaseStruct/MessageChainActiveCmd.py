# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/06/04 13:33:01.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class MessageChainActiveCmd:
	# 起始点id
	startPointIdList: list[int] = field(default_factory=list[int])
	# 终止点id
	endPointIdList: list[int] = field(default_factory=list[int])
