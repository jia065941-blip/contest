# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2024/06/05 09:17:22.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .P_P_MsgChainInfo import P_P_MsgChainInfo


@dataclass_json
@dataclass
class MsgChainTrigger:
	# 数据链信息列表
	msgChainInfoList: list[P_P_MsgChainInfo] = field(default_factory=list[P_P_MsgChainInfo])
