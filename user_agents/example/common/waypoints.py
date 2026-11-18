from math import isfinite


def bounded_waypoint(x: float, y: float, map_area: dict) -> tuple[float, float]:
    # 合法探测记录可能位于地图外；只约束提交航点，不修改目标记录。
    if not isfinite(x) or not isfinite(y):
        raise ValueError('航点坐标必须是有限数值')
    return (max(map_area['xMin'], min(map_area['xMax'], x)),
            max(map_area['yMin'], min(map_area['yMax'], y)))
