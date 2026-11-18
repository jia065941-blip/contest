# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2023/10/18 14:14:43.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class EasyEventSendCmd:
	# 触发器类型
	triggerType: int = 0
	# 内容
	event: str = ""
