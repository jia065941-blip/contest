# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/06/04 10:49:37.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class MsgNetExternal:
	# 可用中继器id
	id: list[int] = field(default_factory=list[int])
