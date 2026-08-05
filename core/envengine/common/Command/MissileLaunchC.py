# -*- coding: utf-8 -*-#
# Version:14, Last Modified time: 2026/01/20 12:40:50.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class MissileLaunchC:
	# 进攻弹发射指令
	cmd: MissileLaunchCmd = field(default_factory=MissileLaunchCmd)
