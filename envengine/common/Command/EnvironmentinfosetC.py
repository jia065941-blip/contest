# -*- coding: utf-8 -*-#
# Version:4, Last Modified time: 2024/10/15 00:43:26.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class EnvironmentinfosetC:
	# 环境设置
	cmd: EnvironmentSetCmd = field(default_factory=EnvironmentSetCmd)
