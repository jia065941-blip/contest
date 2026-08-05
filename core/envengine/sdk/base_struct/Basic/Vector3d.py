# -*- coding: utf-8 -*-#
from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Vector3d:
    # x
    x: float = 0.0
    # y
    y: float = 0.0
    # z
    z: float = 0.0
