# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2023/10/18 15:47:31.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class MessageChainCreateCmd:
	# 可控中继器集合
	msgRepeaterIdList: list[int] = field(default_factory=list[int])
	# 通信网id
	id: int = 0
