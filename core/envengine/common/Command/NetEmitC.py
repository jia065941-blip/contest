# -*- coding: utf-8 -*-#
# Version:3, Last Modified time: 2024/06/25 18:58:31.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class NetEmitC:
	# 发射网捕
	cmd: MissileLaunchCmd = field(default_factory=MissileLaunchCmd)
