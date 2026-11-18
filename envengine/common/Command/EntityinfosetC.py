# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2024/10/15 00:47:08.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class EntityinfosetC:
	# 实体信息
	entityInfo: EntityInfoSetCmd = field(default_factory=EntityInfoSetCmd)
