# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2024/10/15 02:22:39.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class TargetcharacterinfoC:
	# 目标特性指令装订
	targetCharacterInfo: TargetCharacterInfoSetCmd = field(default_factory=TargetCharacterInfoSetCmd)
