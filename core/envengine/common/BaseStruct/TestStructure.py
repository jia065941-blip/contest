# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2024/10/15 17:44:12.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class TestStructure:
	# 实体ID
	entityId: int = 0
	# 实体类型
	entityType: int = 0
	# 实体中文名称
	entityNameChn: str = ""
	# 实体英文名称
	entityNameEn: str = ""
	# 实体型号
	entityTypeId: int = 0
