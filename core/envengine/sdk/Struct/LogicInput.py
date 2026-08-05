# -*-coding:utf-8 -*-
from typing import NamedTuple


class LogicInput(NamedTuple):
    """逻辑回调输入"""
    round: int
    """轮次数"""
    state: str
    """状态信息"""
    any_result_enable: bool
    """是否允许返回任意字符串"""
