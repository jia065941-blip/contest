# -*- coding: utf-8 -*-#
# Version:4, Last Modified time: 2024/06/26 00:50:47.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class RetrieveChildrenC:
	# 回收子物体
	cmd: RetrieveChildrenCmd = field(default_factory=RetrieveChildrenCmd)
