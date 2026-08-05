# -*- coding: utf-8 -*-#
# Version:12, Last Modified time: 2024/06/25 18:58:36.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class MoveC:
	# 开始移动指令
	cmd: MoveCmd = field(default_factory=MoveCmd)
