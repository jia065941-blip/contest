# -*-coding:utf-8 -*-
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from envengine.sdk.base_struct.profile.Environment import Environment
from envengine.sdk.base_struct.profile.Imagine import Imagine


@dataclass_json
@dataclass
class Profile(object):
    # 环境设定
    environmentProfile: Environment = field(default_factory=Environment)
    # 想定设定
    imagineProfile: Imagine = field(default_factory=Imagine)
