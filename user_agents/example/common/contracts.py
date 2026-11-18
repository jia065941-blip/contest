"""任务分配和搜索几何共用的不可变位置数据。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Position:
    """本地坐标位置，x 向东、y 向北、z 向上，三个分量的单位均为米。

    距离函数和搜索几何在需要时将米转换为千米。
    """

    x: float
    y: float
    z: float = 0.0
