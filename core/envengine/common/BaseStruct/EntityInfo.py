# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2023/10/18 11:42:43.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class EntityInfo:
	# id
	id: int = 0
	# 类型
	type: int = 0
	# 传感器型号
	typeId: int = 0
	# 阵营
	side: int = 0
