# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/06/18 21:32:12.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .WeaponLaunchStrategy import WeaponLaunchStrategy


@dataclass_json
@dataclass
class DefendCommanderStartCmd:
	# 拦截策略
	weaponLaunchStrategy: list[WeaponLaunchStrategy] = field(default_factory=list[WeaponLaunchStrategy])
	# 可用拦截挂架
	availableBody: list[int] = field(default_factory=list[int])
