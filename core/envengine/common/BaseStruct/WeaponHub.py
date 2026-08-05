# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/06/13 20:43:08.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .SingleWeaponHub import SingleWeaponHub


@dataclass_json
@dataclass
class WeaponHub:
	# 不同类型的武器仓库
	weaponHubList: dict[int, SingleWeaponHub] = field(default_factory=dict[int, SingleWeaponHub])
