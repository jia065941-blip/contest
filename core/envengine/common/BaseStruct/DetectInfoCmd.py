# -*- coding: utf-8 -*-#
# Version:3, Last Modified time: 2023/10/18 14:01:05.
# Created by Administrator on 2026/05/18 14:50:45.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from .Vector3d import Vector3d
from .EntityInfo import EntityInfo


@dataclass_json
@dataclass
class DetectInfoCmd:
    pass

