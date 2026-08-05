# -*- coding: utf-8 -*-#
# Version:4, Last Modified time: 2024/06/03 16:58:33.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class RetrieveChildrenCmd:
	# 子物体ID列表
	childrenId: list[int] = field(default_factory=list[int])
