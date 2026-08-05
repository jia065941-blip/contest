# -*-coding:utf-8 -*-
from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Negotiation:
    """协商数据"""
    type: str
    """类型"""
    version: str
    """版本"""
    @staticmethod
    def current() -> "Negotiation":
        """获取当前值"""
        negotiation = Negotiation(type="PYTHON", version="v1.0")
        return negotiation