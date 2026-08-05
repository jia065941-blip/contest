# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2024/10/15 01:53:36.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class TransmissioninfosetC:
	# 传输指令装订
	transmissionInfo: TransmissionInfoSetCmd = field(default_factory=TransmissionInfoSetCmd)
