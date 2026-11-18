# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/10/15 00:36:06.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .EntityInfo_GS import EntityInfo_GS


@dataclass_json
@dataclass
class EntityInfoSetCmd:
	# 红方实体信息
	redEntityInfo: list[EntityInfo_GS] = field(default_factory=list[EntityInfo_GS])
	# 蓝方实体信息
	buleTargetEntityInfo: list[EntityInfo_GS] = field(default_factory=list[EntityInfo_GS])
