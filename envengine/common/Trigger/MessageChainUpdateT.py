# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2024/06/04 16:34:25.
# Created by Administrator on 2026/05/18 14:50:47.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class MessageChainUpdateT:
	# 数据链更新
	event: MsgChainTrigger = field(default_factory=MsgChainTrigger)
