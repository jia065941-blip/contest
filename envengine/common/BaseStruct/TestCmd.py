# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2024/07/03 23:25:58.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class TestCmd:
	# 测试任务
	taskTest: str = ""
