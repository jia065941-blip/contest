"""场景平面坐标辅助函数；x向东、y向北、z为高度，单位为米。"""

import math


def board_to_position(x, y, width, height, area, altitude=0):
    """把左上角为原点的画板位置映射到场景范围。"""
    if width <= 0 or height <= 0:
        raise ValueError("画板宽高必须为正数。")
    return {"x": area["xMin"] + x / width * (area["xMax"] - area["xMin"]),
            "y": area["yMax"] - y / height * (area["yMax"] - area["yMin"]),
            "z": altitude}


def set_position(entity, pos):
    """写入位置并清零速度，不改变实体身份或拓扑。"""
    entity["pos"] = dict(pos)
    entity["vel"] = {"x": 0.0, "y": 0.0, "z": 0.0}


def blue_detects_position(sensor, target, stage=3, visible=True, health=1):
    """检查存活、可见目标是否位于20公里三维探测范围内。"""
    distance = math.dist(tuple(sensor[axis] for axis in "xyz"),
                         tuple(target[axis] for axis in "xyz"))
    return visible and health > 0 and stage >= 3 and distance <= 20_000
