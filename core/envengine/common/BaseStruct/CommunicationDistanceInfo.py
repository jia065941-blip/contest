# -*- coding: utf-8 -*-#
# Version:3, Last Modified time: 2023/10/18 14:00:56.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class CommunicationDistanceInfo:
	# 有线连接距离, 为-1则为无限
	wiredDistance: int = 0
	# 无线连接距离, 为-1则为无限
	wirelessDistance: int = 0
	#  允许无线连接
	enableWireless: int = 0
