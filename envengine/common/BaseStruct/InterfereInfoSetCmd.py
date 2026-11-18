# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2024/10/15 01:58:26.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .InterfereInfo import InterfereInfo


@dataclass_json
@dataclass
class InterfereInfoSetCmd:
	# 干扰信息
	interfereInfo: list[InterfereInfo] = field(default_factory=list[InterfereInfo])
