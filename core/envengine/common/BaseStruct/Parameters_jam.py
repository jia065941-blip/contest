# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2026/05/09 10:12:57.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Parameters_jam:
	# 参数名字
	type: str = ""
	# 参数数值
	value: float = 0.0
