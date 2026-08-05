# -*-coding:utf-8 -*-

from .Package import Package

class ReceiveSaver:
    """
    接收
    """

    def __init__(self, state: bool = False, package: Package | None = None):
        self.state: bool = state
        """状态"""
        self.package: Package | None = package
        """内容"""

    def save(self, _content: Package):
        """保存"""
        self.state = True
        self.package = _content

    def load(self) -> Package | None:
        """获取"""
        self.state = False
        package = self.package
        self.package = None
        return package