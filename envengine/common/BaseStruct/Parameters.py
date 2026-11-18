# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2026/05/07 10:06:43.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Parameters:
	# 参数名称
	param_name: str = ""
	# 参数值
	param_value: float = 0.0
