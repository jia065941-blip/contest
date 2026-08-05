from typing import Tuple

# 假设你已经在 UtilsPy.pyi 中定义了这些
from UtilsPy import State_py, Vector3D, Quaternion


# =========================
# JTC_OrbitModel
# =========================
class JTC_OrbitModel:
    """
    轨道模型（JTC_Basic_OrbitModel）

    构造方式：
        JTC_OrbitModel(year, month, day, hour, minute, second,
                       semi_majorAxis, eccentricity, inclination, argOfPerigee, rann, meananom)

    使用方式：
        model.update(year, month, day, hour, minute, second, step)
        state = model.getState()
    """

    def __init__(
        self,
        year: int,
        month: int,
        day: int,
        hour: int,
        minute: int,
        second: float,
        semi_majorAxis: float,
        eccentricity: float,
        inclination: float,
        argOfPerigee: float,
        rann: float,
        meananom: float,
    ) -> None: ...

    def update(
        self,
        year: int,
        month: int,
        day: int,
        hour: int,
        minute: int,
        second: float,
        step: float,
    ) -> None:
        """
        计算给定时刻的位置速度
        """

    def getState(self) -> State_py:
        """
        返回当前时刻的状态（地惯系 / 地固系 / 经纬高）
        """