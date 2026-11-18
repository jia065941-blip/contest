# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/06/18 21:33:46.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class WeaponLaunchStrategy:
	# 执行轮次
	waveCount: int = 0
	# 单轮释放单位数量
	weaponCount: int = 0
	# 轮次间释放间隔（秒）
	waveInterval: float = 0.0
	# 目标类型
	targetTypes: list[int] = field(default_factory=list[int])
