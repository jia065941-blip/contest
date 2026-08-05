# -*- coding: utf-8 -*-#
# Version:4, Last Modified time: 2024/06/18 23:47:01.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class SceneViewChangeC:
	# 视景视角切换
	cmd: SceneViewChangCmd = field(default_factory=SceneViewChangCmd)
