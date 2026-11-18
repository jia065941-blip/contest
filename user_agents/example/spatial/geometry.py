import json
from math import isfinite
from pathlib import Path


_MAPS = json.loads(Path(__file__).with_name('public_geometry.json').read_text(encoding='utf-8-sig'))


def public_geometry(coordinates):
    vertices = tuple(sorted(tuple(point) for point in coordinates))
    if len(vertices) != 4 or any(len(point) != 2 or not all(
            type(value) in (int, float) and isfinite(value) for value in point) for point in vertices):
        raise ValueError('空间波次部署需要四个有限二维顶点')
    matches = [row for row in _MAPS
               if tuple(sorted(tuple(point) for point in row['coordinates'])) == vertices]
    if not matches:
        raise ValueError('部署区域不在公开地图几何表中，请补充公开边界和步长')
    first = matches[0]
    if any(row['map_area'] != first['map_area'] or row['step_duration_ms'] != first['step_duration_ms']
           for row in matches):
        raise ValueError('公开部署区域对应的地图边界或步长不唯一')
    return dict(first['map_area']), first['step_duration_ms']
