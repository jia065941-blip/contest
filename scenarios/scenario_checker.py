"""读取场景 JSON，检查物理结构、实体关系和地图说明。"""

from __future__ import annotations

import json
import math
from pathlib import Path


class SceneValidationError(ValueError):
    """记录场景中具体 JSON 字段路径的检查错误。"""

    def __init__(self, path: str, message: str):
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


# 保留对象原始键序列与特殊数值标记，用于识别重复键和非有限数值。
class _Pairs(list):
    pass


class _NonFinite:
    def __init__(self, token):
        self.token = token


def _fail(path, message):
    raise SceneValidationError(path, message)


def _strict_json(text: str, path: str = "$"):
    """解析 JSON，同时检查重复键、特殊数值和数值溢出。"""
    try:
        parsed = json.loads(text, object_pairs_hook=_Pairs, parse_constant=_NonFinite)
    except json.JSONDecodeError as error:
        _fail(path, f"invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}")

    def unpack(value, location):
        if isinstance(value, _Pairs):
            result = {}
            for key, item in value:
                child = f"{location}.{key}"
                if key in result:
                    _fail(child, "duplicate JSON key")
                result[key] = unpack(item, child)
            return result
        if isinstance(value, list):
            return [unpack(item, f"{location}[{index}]") for index, item in enumerate(value)]
        if isinstance(value, _NonFinite):
            _fail(location, f"non-finite JSON number {value.token}")
        if isinstance(value, float) and not math.isfinite(value):
            _fail(location, "number must be finite")
        return value

    return unpack(parsed, path)


# 对直接传入的字典也检查 JSON 值类型，保持文件与内存检查入口一致。
def _json_values(value, path):
    if type(value) is float and not math.isfinite(value):
        _fail(path, "number must be finite")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(path, "object keys must be strings")
            _json_values(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _json_values(item, f"{path}[{index}]")
    elif type(value) not in (str, int, float, bool, type(None)):
        _fail(path, "value is not JSON data")


# 基础字段检查将错误定位到对象、键或具体向量分量。
def _object(value, path):
    if not isinstance(value, dict):
        _fail(path, "must be an object")
    return value


def _required(value, key, path):
    if key not in value:
        _fail(f"{path}.{key}", "missing required field")
    return value[key]


def _number(value, path):
    if type(value) not in (int, float):
        _fail(path, "must be a finite number, not a boolean")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        _fail(path, "number must be finite")
    return value


def _integer(value, path):
    if type(value) is not int:
        _fail(path, "must be an integer, not a boolean")
    return value


def _vector(value, path):
    value = _object(value, path)
    return tuple(_number(_required(value, axis, path), f"{path}.{axis}") for axis in ("x", "y", "z"))

# 部署多边形通过面积、顶点与边相交检查确定几何有效性。
def _orientation(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a, b, c, d):
    ab_c, ab_d = _orientation(a, b, c), _orientation(a, b, d)
    cd_a, cd_b = _orientation(c, d, a), _orientation(c, d, b)
    if ab_c * ab_d < 0 and cd_a * cd_b < 0:
        return True
    for value, point, start, end in ((ab_c, c, a, b), (ab_d, d, a, b),
                                      (cd_a, a, c, d), (cd_b, b, c, d)):
        if value == 0 and min(start[0], end[0]) <= point[0] <= max(start[0], end[0]) and min(start[1], end[1]) <= point[1] <= max(start[1], end[1]):
            return True
    return False


def _polygons(value, path):
    if not isinstance(value, list) or not value:
        _fail(path, "must be a nonempty list of polygons")
    for index, ring in enumerate(value):
        location = f"{path}[{index}]"
        if not isinstance(ring, list):
            _fail(location, "polygon must be a list of x/y pairs in metres")
        points = []
        for point_index, point in enumerate(ring):
            point_path = f"{location}[{point_index}]"
            if not isinstance(point, list) or len(point) != 2:
                _fail(point_path, "must contain x and y")
            x, y = (_number(v, f"{point_path}[{axis}]") for axis, v in enumerate(point))
            points.append((x, y))
        if len(points) > 1 and points[0] == points[-1]:
            points.pop()
        if len(points) < 3 or len(set(points)) != len(points):
            _fail(location, "polygon needs at least three distinct vertices without repeated edges")
        area = sum(_orientation(points[0], points[i], points[i + 1]) for i in range(1, len(points) - 1))
        if area == 0:
            _fail(location, "polygon must have nonzero area")
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if j == i + 1 or (i == 0 and j == len(points) - 1):
                    continue
                if _segments_intersect(points[i], points[(i + 1) % len(points)], points[j], points[(j + 1) % len(points)]):
                    _fail(location, "polygon edges must not intersect")


def _model_types(environment):
    """检查实体类型与模型路径的一一映射，返回已声明类型。"""
    path = "$.environmentProfile.dynamicLibraryConfigs"
    rows = _required(environment, "dynamicLibraryConfigs", "$.environmentProfile")
    if not isinstance(rows, list) or not rows:
        _fail(path, "must be a nonempty list of model mappings")
    result = set()
    for index, row in enumerate(rows):
        location = f"{path}[{index}]"
        row = _object(row, location)
        kind = _integer(_required(row, "entityType", location), f"{location}.entityType")
        if kind in result:
            _fail(f"{location}.entityType", f"duplicate model mapping for type {kind}")
        result.add(kind)
        model = _required(row, "dynamicLibraryPath", location)
        if not isinstance(model, str) or not model.strip():
            _fail(f"{location}.dynamicLibraryPath", "must be a nonempty model path")
    return result


def _entity_graph(entities):
    """检查父子引用、双向对应与实体树中的环。"""
    parents, children = {}, {}
    for entity_id, (entity, path) in entities.items():
        parent = _integer(entity.get("parentId", -1), f"{path}.parentId")
        if parent != -1 and parent not in entities:
            _fail(f"{path}.parentId", f"unknown parent entity {parent}")
        parents[entity_id] = parent
        listed = entity.get("childrenId", [])
        if not isinstance(listed, list):
            _fail(f"{path}.childrenId", "must be a list of entity IDs")
        seen = set()
        for index, child in enumerate(listed):
            location = f"{path}.childrenId[{index}]"
            child = _integer(child, location)
            if child not in entities:
                _fail(location, f"unknown child entity {child}")
            if child in seen:
                _fail(location, f"duplicate child entity {child}")
            seen.add(child)
        children[entity_id] = seen
    finished = set()
    for entity_id in entities:
        trail = set()
        current = entity_id
        while current != -1 and current not in finished:
            if current in trail:
                _fail(f"{entities[current][1]}.parentId", "entity parent graph contains a cycle")
            trail.add(current)
            current = parents[current]
        finished.update(trail)
    for entity_id, parent in parents.items():
        if parent != -1 and entity_id not in children[parent]:
            _fail(f"{entities[entity_id][1]}.parentId", "parent does not list this entity in childrenId")
        for child in children[entity_id]:
            if parents[child] != entity_id:
                _fail(f"{entities[entity_id][1]}.childrenId", f"child {child} has a different parentId")


def validate_scene(data: dict) -> None:
    """检查场景字典的结构和物理一致性，保持输入数据原样。"""
    _json_values(data, "$")
    data = _object(data, "$")
    if "metadata" in data:
        metadata = _object(data["metadata"], "$.metadata")
        for field in ("id", "title", "difficulty", "description"):
            if field in metadata and (not isinstance(metadata[field], str) or not metadata[field].strip()):
                _fail(f"$.metadata.{field}", "must be a nonempty string")
    environment = _object(_required(data, "environmentProfile", "$"), "$.environmentProfile")
    model_types = _model_types(environment)
    profile = _object(_required(data, "imagineProfile", "$"), "$.imagineProfile")
    path = "$.imagineProfile"
    # 时间字段以毫秒计；事件和任务沿用当前引擎支持的空列表形式。
    step = _integer(_required(profile, "simStep", path), f"{path}.simStep")
    start = _number(_required(profile, "simTime", path), f"{path}.simTime")
    end = _number(_required(profile, "simEndLogicTime", path), f"{path}.simEndLogicTime")
    logic = _number(profile.get("logicTime", 0), f"{path}.logicTime")
    if step <= 0:
        _fail(f"{path}.simStep", "must be positive milliseconds")
    if start < 0:
        _fail(f"{path}.simTime", "must be a nonnegative absolute timestamp")
    if end < step:
        _fail(f"{path}.simEndLogicTime", "must allow at least one simulation step")
    if not 0 <= logic < end:
        _fail(f"{path}.logicTime", "must be nonnegative and less than simEndLogicTime")
    for field in ("eventList", "taskList"):
        if field in profile:
            if not isinstance(profile[field], list):
                _fail(f"{path}.{field}", "must be a list")
            if profile[field]:
                _fail(f"{path}.{field}", "nonempty event/task lists are not supported")
    # 地图边界与红方部署区共同定义场景空间范围。
    area_path = f"{path}.mapArea"
    area = _object(_required(profile, "mapArea", path), area_path)
    bounds = {key: _number(_required(area, key, area_path), f"{area_path}.{key}") for key in ("xMin", "xMax", "yMin", "yMax")}

    if bounds["xMin"] >= bounds["xMax"] or bounds["yMin"] >= bounds["yMax"]:
        _fail(area_path, "minimum x/y must be less than maximum")
    red_path = f"{path}.redArea"
    red = _object(_required(profile, "redArea", path), red_path)
    _polygons(_required(red, "coordinates", red_path), f"{red_path}.coordinates")
    rows = _required(profile, "entityList", path)
    if not isinstance(rows, list) or not rows:
        _fail(f"{path}.entityList", "must be a nonempty list")
    # 逐实体检查身份、物理初态和模型声明，收齐后统一检查父子图。
    entities = {}
    for index, row in enumerate(rows):
        row_path = f"{path}.entityList[{index}]"
        row = _object(row, row_path)
        entity_path = f"{row_path}.entity"
        entity = _object(_required(row, "entity", row_path), entity_path)
        entity_id = _integer(_required(entity, "id", entity_path), f"{entity_path}.id")
        if entity_id == -1:
            _fail(f"{entity_path}.id", "-1 is reserved for no parent")
        if entity_id in entities:
            _fail(f"{entity_path}.id", f"duplicate entity ID {entity_id}")
        entities[entity_id] = (entity, entity_path)
        kind = _integer(_required(entity, "entityType", entity_path), f"{entity_path}.entityType")
        _integer(_required(entity, "sideId", entity_path), f"{entity_path}.sideId")
        if kind not in model_types:
            _fail(f"{entity_path}.entityType", f"type {kind} has no model mapping")
        health = _number(_required(entity, "survivePoints", entity_path), f"{entity_path}.survivePoints")
        if health < 0:
            _fail(f"{entity_path}.survivePoints", "health must be nonnegative")
        if "isVisible" in entity and type(entity["isVisible"]) is not bool:
            _fail(f"{entity_path}.isVisible", "must be a boolean")
        for field in ("nameChn", "nameEn"):
            if field in entity and not isinstance(entity[field], str):
                _fail(f"{entity_path}.{field}", "must be a string")
        for field in ("vel",):
            if field in entity:
                _vector(entity[field], f"{entity_path}.{field}")
        for field in ("stage", "time"):
            if field in entity:
                value = _number(entity[field], f"{entity_path}.{field}")
                if field == "time" and value < 0:
                    _fail(f"{entity_path}.{field}", "must be a nonnegative absolute timestamp")
        if "step" in entity:
            interval = _integer(entity["step"], f"{entity_path}.step")
            if interval < 0:
                _fail(f"{entity_path}.step", "must be nonnegative milliseconds")
        _vector(_required(entity, "pos", entity_path), f"{entity_path}.pos")
        external = row.get("external", "")
        if not isinstance(external, str):
            _fail(f"{row_path}.external", "must be an empty string or an encoded JSON object")
        if external:
            _object(_strict_json(external, f"{row_path}.external"), f"{row_path}.external")
    _entity_graph(entities)


def check_scenario(path: str | Path) -> dict:
    """单次读取并检查场景，返回可直接生成 Profile 的数据字典。"""
    path = Path(path).expanduser()
    data = _strict_json(path.read_bytes().decode("utf-8"))
    validate_scene(data)

    # 场景以 -1 表示无父节点，显式补齐后交给 Profile 转换。
    for row in data["imagineProfile"]["entityList"]:
        row["entity"].setdefault("parentId", -1)
    return data
