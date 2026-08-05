# -*- coding: utf-8 -*-#
# Version:3, Last Modified time: 2024/06/25 19:03:06.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class TransLaunchWindowC:
	# 拦截窗口
	cmd: TransLaunchWindowCmd = field(default_factory=TransLaunchWindowCmd)
