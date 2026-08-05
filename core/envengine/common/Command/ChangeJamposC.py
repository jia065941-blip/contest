# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2024/10/13 20:09:23.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class ChangeJamposC:
	# 干扰位置改变
	cmd: ChageJammerPosCmd = field(default_factory=ChageJammerPosCmd)
