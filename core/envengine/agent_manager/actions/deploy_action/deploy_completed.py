# -*- coding: utf-8 -*-#
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from envengine.common import SimmerCommandType


@dataclass_json
@dataclass
class DeployCompleted:
    """
    部署完成
    """
    # 指令类型Id
    commandType_id: int = SimmerCommandType.DEPLOY_COMPLETED
