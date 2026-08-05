# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2023/11/18 16:57:38.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Target:
	# 目标
	target: list[int] = field(default_factory=list[int])
	# 型号id
	typeId: int = 0
	# 测试类型
	test_enttyType: int = 0
