# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/10/14 21:48:16.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Haze:
	# 霾条件，0-轻微,1-轻度,2-中度,3-重度
	hazeCondition: int = 0
