# -*-coding:utf-8 -*-

class Package:
    """
    传输包
    """

    def __init__(self, seq: int, content: str):
        self.seq: int = seq
        """序列号"""
        self.content: str = content
        """内容"""

    def serialize(self) -> str:
        """序列化"""
        return str(self.seq) + "|" + self.content

    @staticmethod
    def deserialize(data: str) -> "Package":
        """反序列化"""
        index = data.find("|")
        seq = int(data[:index])
        content = data[index + 1:]
        return Package(seq, content)
