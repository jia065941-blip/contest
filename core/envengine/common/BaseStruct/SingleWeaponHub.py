# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/06/11 10:56:09.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class SingleWeaponHub:
	# 该类型武器集合
	weaponIdList: list[int] = field(default_factory=list[int])
