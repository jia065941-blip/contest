# -*- coding: utf-8 -*-#
# Version:1, Last Modified time: 2026/05/09 11:17:02.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .JammingRule_17s import JammingRule_17s


@dataclass_json
@dataclass
class JammingRule_17s_v:
	# 干扰数组
	raderjam: list[JammingRule_17s] = field(default_factory=list[JammingRule_17s])
