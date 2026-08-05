# -*- coding: utf-8 -*-#
# Version:2, Last Modified time: 2026/01/20 16:32:05.
# Created by Administrator on 2026/05/18 14:50:46.
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
class EnvironmentSetCmd:
	# 大气条件信息
	atmosphere: Atmosphere = field(default_factory=Atmosphere)
	# 风条件信息
	wind: Wind = field(default_factory=Wind)
	# 云条件信息
	cloud: Cloud = field(default_factory=Cloud)
	# 雨条件信息
	rain: Rain = field(default_factory=Rain)
	# 雾条件信息
	fog: Fog = field(default_factory=Fog)
	# 雪条件信息
	snow: Snow = field(default_factory=Snow)
	# 霾条件信息
	haze: Haze = field(default_factory=Haze)
	# 海况条件信息
	sea: Sea = field(default_factory=Sea)
