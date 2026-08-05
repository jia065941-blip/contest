# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2024/10/14 22:01:49.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .Atmosphere import Atmosphere
from .Wind import Wind
from .Cloud import Cloud
from .Rain import Rain
from .Fog import Fog
from .Snow import Snow
from .Haze import Haze
from .Sea import Sea


@dataclass_json
@dataclass
class EnvironmentInfo:
	# 大气条件信息
	atmosphere: Atmosphere = field(default_factory=Atmosphere)
	# 风条件
	wind: Wind = field(default_factory=Wind)
	# 云条件
	cloud: Cloud = field(default_factory=Cloud)
	# 雨条件
	rain: Rain = field(default_factory=Rain)
	# 雾条件
	fog: Fog = field(default_factory=Fog)
	# 雪条件
	snow: Snow = field(default_factory=Snow)
	# 霾条件
	haze: Haze = field(default_factory=Haze)
	# 海条件
	sea: Sea = field(default_factory=Sea)
