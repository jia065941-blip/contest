# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/07/05 18:07:08.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class TriggerDistance:
	# 起始实体ID
	startEntityId: int = 0
	# 终止实体ID
	endEntityId: int = 0
	# 触发距离
	triggerDis: float = 0.0
	# 偏差距离
	deviationDis: float = 0.0
	# 基础距离
	basicDis: float = 0.0
	# 触发器类型(0:时间触发器 1:距离触发器 2:条件触发器)
	triggerType: int = 0
