# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/06/19 17:58:50.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .EntityInfo import EntityInfo


@dataclass_json
@dataclass
class SetRelatedEntityCmd:
	# 相关实体信息列表
	relatedInfoList: list[EntityInfo] = field(default_factory=list[EntityInfo])
	# 是否相关
	isRelated: bool = False
