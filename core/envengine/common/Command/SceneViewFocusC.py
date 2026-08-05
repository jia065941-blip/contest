# -*- coding: utf-8 -*-#
# Version:3, Last Modified time: 2024/06/18 23:46:49.
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from ..BaseStruct import *


@dataclass_json
@dataclass
class SceneViewFocusC:
	# 视景聚焦
	cmd: SceneViewFocusCmd = field(default_factory=SceneViewFocusCmd)
