# -*- coding: utf-8 -*-#
# Version:11, Last Modified time: 2024/06/26 00:51:06.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class DestroyC:
	# 摧毁指令
	cmd: DestoryCmd = field(default_factory=DestoryCmd)
