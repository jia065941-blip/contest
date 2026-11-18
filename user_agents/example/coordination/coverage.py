"""为公开凸搜索区域生成往复扫描路径，并分配各平台的连续搜索段。

输入 Position 使用米制 x、y、z；算法将水平坐标平移并换算为千米，
输出时恢复为米制 Position。高度随参考值保留，
覆盖和分配计算只使用水平距离。

处理顺序为：检查凸多边形 → 比较扫描方向 → 生成往复扫描线 → 加密路径 →
按平台接近距离与搜索距离分段。下方还提供静态三机搜索使用的扇形边界采样。
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from math import atan2, ceil, cos, degrees, hypot, isfinite, pi, radians, sin
from typing import Sequence

from ..common.contracts import Position


_EPSILON = 1e-9  # 几何边界、平行线和浮点目标比较使用的容差。
_MAX_ALLOCATION_EDGES = 4096  # 参与切段优化的路径边数控制尺度。
_XY = tuple[float, float]  # (向东距离, 向北距离)，单位 km。


@dataclass(frozen=True)
class SearchPlatformStart:
    """待参与搜索的平台编号及其 Position 起点。"""

    platform_id: int
    position: Position


@dataclass(frozen=True)
class CoverageRoute:
    """分给一台平台的连续航点段及距离统计，距离单位均为 km。"""

    platform_id: int
    waypoints: tuple[Position, ...]
    preparation_distance_km: float  # 从平台起点到该路线较近端点的距离。
    search_distance_km: float  # 沿分配到的扫描折线行进的长度。

    @property
    def total_distance_km(self) -> float:
        """平台接近路线并完成搜索所需的水平总距离。"""
        return self.preparation_distance_km + self.search_distance_km


@dataclass(frozen=True)
class CoveragePlan:
    """完整覆盖计划：分机路线、公共扫描折线及扫描方向、间距和长度。"""

    routes: tuple[CoverageRoute, ...]
    coverage_waypoints: tuple[Position, ...]  # 加密和分段之前的公共往复扫描折线。
    search_radius_km: float
    lane_spacing_km: float
    lane_count: int
    sweep_bearing_degrees: float
    coverage_distance_km: float
    idle_platform_ids: tuple[int, ...] = ()  # 平台多于可分路径段时的剩余编号。

    @property
    def max_total_distance_km(self) -> float:
        """所有平台中最长的接近加搜索距离，空计划返回 0。"""
        return max((route.total_distance_km for route in self.routes), default=0.0)


@dataclass(frozen=True)
class _LocalFrame:
    """将米制位置平移至参考点，并转换为二维千米坐标。"""

    reference_x: float
    reference_y: float
    altitude: float

    def to_xy(self, position: Position) -> _XY:
        """将位置转换为相对参考点的二维 km 坐标。"""
        return (
            (position.x - self.reference_x) / 1000.0,
            (position.y - self.reference_y) / 1000.0,
        )

    def to_position(self, point: _XY) -> Position:
        """将二维 km 点恢复为米制位置，并保留参考高度。"""
        return Position(
            x=self.reference_x + point[0] * 1000.0,
            y=self.reference_y + point[1] * 1000.0,
            z=self.altitude,
        )


@dataclass(frozen=True)
class _SweepPath:
    """一个候选扫描方向对应的局部二维折线及其几何统计。"""

    points: tuple[_XY, ...]
    lane_spacing_km: float
    lane_count: int
    bearing_degrees: float
    length_km: float


@dataclass(frozen=True)
class _Allocation:
    """路径切点及每段的平台索引、接近距离和总距离。"""

    cuts: tuple[int, ...]
    platform_by_segment: tuple[int, ...]
    preparation_by_segment: tuple[float, ...]
    total_by_segment: tuple[float, ...]

    @property
    def objective(self) -> tuple[float, float]:
        """先降低最长平台路程，再以总路程平方和消除相同最大值的并列。"""
        return max(self.total_by_segment), sum(value * value for value in self.total_by_segment)


def plan_static_coverage(
    polygon: Sequence[Position],
    search_radius_km: float,
    platforms: Sequence[SearchPlatformStart],
) -> CoveragePlan:
    """按搜索半径生成凸区域扫描路径，并给各平台分配连续航点段。

    polygon 是按边界顺序排列的 Position 顶点；search_radius_km 为圆形探测
    半径，扫描线中心间距至多为其两倍。routes 使用连续且衔接的公共路径段，
    每段可反向，使平台从较近一端进入。输入检查要求凸区域、有限坐标、
    正半径和互不重复的平台编号。
    """

    # 先统一闭合点表示并检查输入，再以区域均值作为局部投影参考点。
    vertices = _normalize_polygon(polygon)
    platform_starts = tuple(platforms)
    _validate_platforms(platform_starts)
    if not isfinite(search_radius_km) or search_radius_km <= 0:
        raise ValueError("search radius must be a positive finite distance")
    if not platform_starts:
        raise ValueError("at least one search platform is required")

    projection = _projection_for(vertices)
    polygon_xy = tuple(projection.to_xy(vertex) for vertex in vertices)
    _validate_convex_polygon(polygon_xy)

    # 对每种边方向生成候选扫描线；随后加密折线，提供更细的可分段位置。
    sweep = _choose_sweep_path(polygon_xy, search_radius_km)
    platform_xy = tuple((item.platform_id, projection.to_xy(item.position)) for item in platform_starts)
    dense_path = _densify_path(sweep.points, search_radius_km, len(platform_xy))
    routes_xy, idle_platform_ids = _allocate_path(dense_path, platform_xy)

    # 局部二维规划结束后恢复 Position；编号排序使外部读取顺序稳定。
    routes = tuple(
        CoverageRoute(
            platform_id=platform_id,
            waypoints=tuple(projection.to_position(point) for point in route_points),
            preparation_distance_km=preparation_distance,
            search_distance_km=search_distance,
        )
        for platform_id, route_points, preparation_distance, search_distance in routes_xy
    )
    return CoveragePlan(
        routes=tuple(sorted(routes, key=lambda route: route.platform_id)),
        coverage_waypoints=tuple(projection.to_position(point) for point in sweep.points),
        search_radius_km=search_radius_km,
        lane_spacing_km=sweep.lane_spacing_km,
        lane_count=sweep.lane_count,
        sweep_bearing_degrees=sweep.bearing_degrees,
        coverage_distance_km=sweep.length_km,
        idle_platform_ids=tuple(sorted(idle_platform_ids)),
    )


def radial_fan_endpoints(
    origin: Position,
    bounds: Sequence[Position],
    count: int,
    max_range_km: float,
) -> tuple[Position, ...]:
    """从起点向公开凸边界均匀生成 count 条射线，返回 Position 端点。

    起点位于区域内时覆盖整圆；位于区域外时覆盖包含全部顶点的最小角域。
    每条射线取远侧边界交点，并以 max_range_km 截断。端点沿用起点高度，
    后续搜索策略可据此生成各方向的候选编队目的地。
    """

    # 同时参考区域与起点构造投影，以起点高度作为返回航点高度。
    vertices = _normalize_polygon(bounds)
    if count <= 0:
        raise ValueError("radial endpoint count must be positive")
    if not isfinite(max_range_km) or max_range_km <= 0:
        raise ValueError("maximum radial range must be a positive finite distance")
    if not all(isfinite(value) for value in (origin.x, origin.y, origin.z)):
        raise ValueError("origin coordinates must be finite")

    base_projection = _projection_for(vertices + (origin,))
    projection = _LocalFrame(
        base_projection.reference_x,
        base_projection.reference_y,
        origin.z,
    )
    polygon_xy = tuple(projection.to_xy(vertex) for vertex in vertices)
    _validate_convex_polygon(polygon_xy)
    origin_xy = projection.to_xy(origin)

    # 区域内部不限制方向；外部只采样朝向区域的可见角域。
    if _point_in_convex_polygon(origin_xy, polygon_xy):
        bearings = tuple(2.0 * pi * index / count for index in range(count))
    else:
        bearings = _visible_bearings(origin_xy, polygon_xy, count)

    # 射线与凸多边形可能有进入、离开两个交点，这里使用离开边界的距离。
    endpoints: list[Position] = []
    for bearing in bearings:
        direction = cos(bearing), sin(bearing)
        boundary_distance = _ray_polygon_exit_distance(origin_xy, direction, polygon_xy)
        distance = min(max_range_km, boundary_distance)
        endpoints.append(projection.to_position(_add(origin_xy, _scale(direction, distance))))
    return tuple(endpoints)


def _normalize_polygon(polygon: Sequence[Position]) -> tuple[Position, ...]:
    """移除重复的闭合顶点，并确认至少有三个有限坐标顶点。"""
    vertices = tuple(polygon)
    if len(vertices) >= 2 and vertices[0] == vertices[-1]:
        vertices = vertices[:-1]
    if len(vertices) < 3:
        raise ValueError("a search polygon requires at least three vertices")
    for vertex in vertices:
        if not all(isfinite(value) for value in (vertex.x, vertex.y, vertex.z)):
            raise ValueError("polygon coordinates must be finite")
    return vertices


def _validate_platforms(platforms: tuple[SearchPlatformStart, ...]) -> None:
    """检查平台编号唯一性及起点三个坐标分量的有限性。"""
    identifiers = [item.platform_id for item in platforms]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("search platform identifiers must be unique")
    for item in platforms:
        if not all(isfinite(value) for value in (item.position.x, item.position.y, item.position.z)):
            raise ValueError("platform coordinates must be finite")


def _projection_for(vertices: tuple[Position, ...]) -> _LocalFrame:
    """取顶点均值构造本地米制参考坐标。"""
    count = len(vertices)
    projection = _LocalFrame(
        reference_x=sum(item.x for item in vertices) / count,
        reference_y=sum(item.y for item in vertices) / count,
        altitude=sum(item.z for item in vertices) / count,
    )
    return projection


def _validate_convex_polygon(vertices: tuple[_XY, ...]) -> None:
    """在二维 km 平面检查面积和连续边转向，要求有序凸边界。"""
    # 鞋带公式检查非退化面积，顺时针与逆时针均可使用。
    signed_area_twice = sum(
        _cross(vertices[index], vertices[(index + 1) % len(vertices)])
        for index in range(len(vertices))
    )
    if abs(signed_area_twice) <= _EPSILON:
        raise ValueError("search polygon must have positive area")

    # 非共线的相邻边必须始终朝同一方向转弯。
    turn_sign = 0
    for index in range(len(vertices)):
        first = vertices[index]
        second = vertices[(index + 1) % len(vertices)]
        third = vertices[(index + 2) % len(vertices)]
        turn = _cross(_subtract(second, first), _subtract(third, second))
        if abs(turn) <= _EPSILON:
            continue
        current_sign = 1 if turn > 0 else -1
        if turn_sign and current_sign != turn_sign:
            raise ValueError("search polygon must be convex with ordered boundary vertices")
        turn_sign = current_sign
    if turn_sign == 0:
        raise ValueError("search polygon must have positive area")


def _choose_sweep_path(polygon: tuple[_XY, ...], radius_km: float) -> _SweepPath:
    """枚举互异边轴方向，以总折线长度、线数、方位角选择扫描方案。"""
    candidates: list[_SweepPath] = []
    seen_axes: set[tuple[int, int]] = set()
    for index, point in enumerate(polygon):
        edge = _subtract(polygon[(index + 1) % len(polygon)], point)
        length = _norm(edge)
        if length <= _EPSILON:
            continue
        axis = (edge[0] / length, edge[1] / length)
        # 正反方向视为同一条扫描轴，量化后去重平行边。
        if axis[0] < -_EPSILON or (abs(axis[0]) <= _EPSILON and axis[1] < 0):
            axis = (-axis[0], -axis[1])
        axis_key = (round(axis[0] * 1_000_000_000), round(axis[1] * 1_000_000_000))
        if axis_key in seen_axes:
            continue
        seen_axes.add(axis_key)
        candidates.append(_build_sweep_path(polygon, radius_km, axis))
    if not candidates:
        raise ValueError("search polygon has no usable edge")
    return min(candidates, key=lambda item: (item.length_km, item.lane_count, item.bearing_degrees))


def _build_sweep_path(polygon: tuple[_XY, ...], radius_km: float, axis: _XY) -> _SweepPath:
    """沿 axis 生成往返扫描折线，所有水平量使用 km。"""
    # 法向投影给出扫描宽度；按两倍半径分配线数，再均匀放置扫描中心。
    normal = (-axis[1], axis[0])
    normal_values = [_dot(point, normal) for point in polygon]
    lower, upper = min(normal_values), max(normal_values)
    width = upper - lower
    lane_count = max(1, ceil(width / (2.0 * radius_km) - _EPSILON))
    spacing = width / lane_count

    # 先裁剪出每条扫描线覆盖的带状区域，再取轴向两端；奇偶线反向相连。
    path: list[_XY] = []
    for lane_index in range(lane_count):
        offset = lower + (lane_index + 0.5) * spacing
        band = _clip_to_band(polygon, normal, offset - radius_km, offset + radius_km)
        if not band:
            raise RuntimeError("coverage lane does not intersect its polygon band")
        axial_values = [_dot(point, axis) for point in band]
        start = _add(_scale(axis, min(axial_values)), _scale(normal, offset))
        end = _add(_scale(axis, max(axial_values)), _scale(normal, offset))
        path.extend((start, end) if lane_index % 2 == 0 else (end, start))

    length = _polyline_length(path)
    # 方位角以北向为零，正反轴等价，因此统计范围为 [0, 180) 度。
    bearing = (degrees(atan2(axis[0], axis[1])) + 360.0) % 180.0
    return _SweepPath(tuple(path), spacing, lane_count, bearing, length)


def _clip_to_band(polygon: tuple[_XY, ...], normal: _XY, lower: float, upper: float) -> tuple[_XY, ...]:
    """依次用上下两个半平面裁出法向坐标处于 [lower, upper] 的部分。"""
    below_upper = _clip_half_plane(polygon, normal, upper, keep_below=True)
    return _clip_half_plane(below_upper, normal, lower, keep_below=False)


def _visible_bearings(origin: _XY, polygon: tuple[_XY, ...], count: int) -> tuple[float, ...]:
    """删除顶点方向中最大的空角域，在剩余最小包围角域均匀取弧度角。"""
    angles = sorted((atan2(point[1] - origin[1], point[0] - origin[0]) % (2.0 * pi)) for point in polygon)
    gaps = [
        (angles[(index + 1) % len(angles)] - angles[index]) % (2.0 * pi)
        for index in range(len(angles))
    ]
    gap_index = max(range(len(gaps)), key=gaps.__getitem__)
    start = angles[(gap_index + 1) % len(angles)]
    span = 2.0 * pi - gaps[gap_index]
    if count == 1:
        return ((start + span / 2.0) % (2.0 * pi),)
    return tuple((start + span * index / (count - 1)) % (2.0 * pi) for index in range(count))


def _point_in_convex_polygon(point: _XY, polygon: tuple[_XY, ...]) -> bool:
    """检查点相对各条有向边的叉积符号，边界点也算区域内。"""
    signs: set[int] = set()
    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
        side = _cross(_subtract(end, start), _subtract(point, start))
        if abs(side) > _EPSILON:
            signs.add(1 if side > 0 else -1)
    return len(signs) <= 1


def _ray_polygon_exit_distance(origin: _XY, direction: _XY, polygon: tuple[_XY, ...]) -> float:
    """求正向射线离开凸多边形的最远交点距离，direction 为单位向量。"""
    intersections: list[float] = []
    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
        edge = _subtract(end, start)
        denominator = _cross(direction, edge)
        if abs(denominator) <= _EPSILON:
            continue
        offset = _subtract(start, origin)
        ray_distance = _cross(offset, edge) / denominator
        edge_fraction = _cross(offset, direction) / denominator
        # 射线参数需非负，边参数需落在该边线段内，容差包含端点。
        if ray_distance >= -_EPSILON and -_EPSILON <= edge_fraction <= 1.0 + _EPSILON:
            intersections.append(max(0.0, ray_distance))
    if not intersections:
        raise RuntimeError("radial bearing does not intersect the convex boundary")
    return max(intersections)


def _clip_half_plane(
    polygon: Sequence[_XY],
    normal: _XY,
    threshold: float,
    *,
    keep_below: bool,
) -> tuple[_XY, ...]:
    """按一个半平面裁剪凸多边形，保留边界内顶点和跨界交点。"""
    if not polygon:
        return ()

    # Sutherland–Hodgman 单边裁剪：用法向点积判断顶点是否落在保留侧。
    def inside(point: _XY) -> bool:
        signed = _dot(point, normal) - threshold
        return signed <= _EPSILON if keep_below else signed >= -_EPSILON

    output: list[_XY] = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        # 一条边进出半平面时，先插入其与裁剪边界的交点。
        if current_inside != previous_inside:
            direction = _subtract(current, previous)
            denominator = _dot(direction, normal)
            if abs(denominator) > _EPSILON:
                fraction = (threshold - _dot(previous, normal)) / denominator
                output.append(_add(previous, _scale(direction, fraction)))
        if current_inside:
            output.append(current)
        previous, previous_inside = current, current_inside
    return tuple(output)


def _densify_path(points: tuple[_XY, ...], radius_km: float, platform_count: int) -> tuple[_XY, ...]:
    """在扫描折线上插点，平衡分段精度和分配算法的计算规模。"""
    # 通常按半个半径或每机约八段细分，过长路径再以边数尺度限制精度。
    path_length = _polyline_length(points)
    desired_resolution = min(radius_km / 2.0, path_length / max(1, platform_count * 8))
    resolution = max(path_length / _MAX_ALLOCATION_EDGES, desired_resolution)
    dense: list[_XY] = [points[0]]
    for start, end in zip(points, points[1:]):
        distance = _distance(start, end)
        pieces = max(1, ceil(distance / resolution))
        for piece in range(1, pieces + 1):
            fraction = piece / pieces
            dense.append(
                (
                    start[0] + (end[0] - start[0]) * fraction,
                    start[1] + (end[1] - start[1]) * fraction,
                )
            )
    return tuple(dense)


def _allocate_path(
    points: tuple[_XY, ...],
    platforms: tuple[tuple[int, _XY], ...],
) -> tuple[tuple[tuple[int, tuple[_XY, ...], float, float], ...], tuple[int, ...]]:
    """切分扫描折线并匹配平台，返回每机二维路线与空闲编号。"""
    # 初始按边数平均切段，前缀长度用于常数时间读取每段搜索距离。
    edge_count = len(points) - 1
    segment_count = min(len(platforms), edge_count)
    cuts = tuple(index * edge_count // segment_count for index in range(segment_count)) + (edge_count,)
    prefix_lengths = _prefix_lengths(points)
    allocation = _evaluate_allocation(cuts, points, prefix_lengths, platforms)

    # 最多做 64 轮局部调整；只移动最重路段与相邻段之间的边界。
    for _ in range(min(64, edge_count)):
        best = allocation
        maximum = allocation.objective[0]
        overloaded_segments = [
            index
            for index, total in enumerate(allocation.total_by_segment)
            if abs(total - maximum) <= _EPSILON
        ]
        for segment_index in overloaded_segments:
            moves: list[tuple[int, int]] = []
            if segment_index > 0:
                boundary = segment_index
                transfer = max(
                    0.0,
                    (maximum - allocation.total_by_segment[segment_index - 1]) / 2.0,
                )
                target_length = prefix_lengths[allocation.cuts[boundary]] + transfer
                moves.append((boundary, bisect_left(prefix_lengths, target_length)))
            if segment_index + 1 < segment_count:
                boundary = segment_index + 1
                transfer = max(
                    0.0,
                    (maximum - allocation.total_by_segment[segment_index + 1]) / 2.0,
                )
                target_length = prefix_lengths[allocation.cuts[boundary]] - transfer
                moves.append((boundary, bisect_left(prefix_lengths, target_length)))

            # 把最长路段与邻段的负担差约一半转移过去，检查附近三个离散切点。
            for boundary, proposed_index in moves:
                for candidate_index in (proposed_index - 1, proposed_index, proposed_index + 1):
                    moved = list(allocation.cuts)
                    moved[boundary] = candidate_index
                    if moved[boundary] <= moved[boundary - 1] or moved[boundary] >= moved[boundary + 1]:
                        continue
                    candidate = _evaluate_allocation(tuple(moved), points, prefix_lengths, platforms)
                    if _objective_is_better(candidate.objective, best.objective):
                        best = candidate
        if not _objective_is_better(best.objective, allocation.objective):
            break
        allocation = best

    # 平台多于可分路径段时保留空闲清单；其余平台各承担一条连续路线。
    assigned_platform_indexes = set(allocation.platform_by_segment)
    idle = tuple(
        platform_id
        for index, (platform_id, _) in enumerate(platforms)
        if index not in assigned_platform_indexes
    )
    routes: list[tuple[int, tuple[_XY, ...], float, float]] = []
    for segment_index, platform_index in enumerate(allocation.platform_by_segment):
        start_index = allocation.cuts[segment_index]
        end_index = allocation.cuts[segment_index + 1]
        route_points = points[start_index : end_index + 1]
        platform_id, platform_start = platforms[platform_index]
        distance_to_start = _distance(platform_start, route_points[0])
        distance_to_end = _distance(platform_start, route_points[-1])
        # 同一段可以从任一端进入，选择较近端并反转航点顺序。
        if distance_to_end + _EPSILON < distance_to_start:
            route_points = tuple(reversed(route_points))
        search_distance = prefix_lengths[end_index] - prefix_lengths[start_index]
        routes.append(
            (
                platform_id,
                route_points,
                allocation.preparation_by_segment[segment_index],
                search_distance,
            )
        )
    return tuple(routes), idle


def _evaluate_allocation(
    cuts: tuple[int, ...],
    points: tuple[_XY, ...],
    prefix_lengths: tuple[float, ...],
    platforms: tuple[tuple[int, _XY], ...],
) -> _Allocation:
    """对一组切点计算平台—路段代价，再以最小总成本进行一一匹配。"""
    segment_count = len(cuts) - 1
    costs: list[list[float]] = []
    preparations: list[list[float]] = []
    # costs[平台索引][路段索引] = 接近较近端点的距离 + 路段自身长度。
    for _, platform_start in platforms:
        platform_costs: list[float] = []
        platform_preparations: list[float] = []
        for segment_index in range(segment_count):
            start_index, end_index = cuts[segment_index], cuts[segment_index + 1]
            preparation = min(
                _distance(platform_start, points[start_index]),
                _distance(platform_start, points[end_index]),
            )
            search_distance = prefix_lengths[end_index] - prefix_lengths[start_index]
            platform_preparations.append(preparation)
            platform_costs.append(preparation + search_distance)
        preparations.append(platform_preparations)
        costs.append(platform_costs)

    platform_by_segment = list(_minimum_cost_platforms(costs))
    preparation_by_segment = [0.0] * segment_count
    total_by_segment = [0.0] * segment_count
    for segment_index, platform_index in enumerate(platform_by_segment):
        preparation_by_segment[segment_index] = preparations[platform_index][segment_index]
        total_by_segment[segment_index] = costs[platform_index][segment_index]
    return _Allocation(
        cuts=cuts,
        platform_by_segment=tuple(platform_by_segment),
        preparation_by_segment=tuple(preparation_by_segment),
        total_by_segment=tuple(total_by_segment),
    )


def _minimum_cost_platforms(costs: list[list[float]]) -> tuple[int, ...]:
    """用最短增广路形式的匈牙利算法，将每个路段匹配给不同平台。"""

    platform_count = len(costs)
    segment_count = len(costs[0])
    # 算法为每行寻找一个匹配，故转置为“路段是行，数量更多的平台是列”。
    matrix = [
        [costs[platform_index][segment_index] for platform_index in range(platform_count)]
        for segment_index in range(segment_count)
    ]
    # 第零列作为增广路径起点；势函数和最小松弛量用于逐步降低剩余代价。
    row_potential = [0.0] * (segment_count + 1)
    column_potential = [0.0] * (platform_count + 1)
    matched_row = [0] * (platform_count + 1)
    previous_column = [0] * (platform_count + 1)

    for row in range(1, segment_count + 1):
        matched_row[0] = row
        minimum_slack = [float("inf")] * (platform_count + 1)
        used = [False] * (platform_count + 1)
        column = 0
        # 沿最小松弛列扩展交替树，直至遇到尚未匹配的平台列。
        while True:
            used[column] = True
            active_row = matched_row[column]
            delta = float("inf")
            next_column = 0
            for candidate_column in range(1, platform_count + 1):
                if used[candidate_column]:
                    continue
                reduced_cost = (
                    matrix[active_row - 1][candidate_column - 1]
                    - row_potential[active_row]
                    - column_potential[candidate_column]
                )
                if reduced_cost < minimum_slack[candidate_column]:
                    minimum_slack[candidate_column] = reduced_cost
                    previous_column[candidate_column] = column
                if minimum_slack[candidate_column] < delta:
                    delta = minimum_slack[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(platform_count + 1):
                if used[candidate_column]:
                    row_potential[matched_row[candidate_column]] += delta
                    column_potential[candidate_column] -= delta
                else:
                    minimum_slack[candidate_column] -= delta
            column = next_column
            if matched_row[column] == 0:
                break

        # 回溯 predecessor 链，翻转路径上的匹配关系，新增一行的分配。
        while True:
            prior_column = previous_column[column]
            matched_row[column] = matched_row[prior_column]
            column = prior_column
            if column == 0:
                break

    platform_by_segment = [-1] * segment_count
    for column in range(1, platform_count + 1):
        if matched_row[column]:
            platform_by_segment[matched_row[column] - 1] = column - 1
    return tuple(platform_by_segment)


def _objective_is_better(candidate: tuple[float, float], current: tuple[float, float]) -> bool:
    """按最大路程优先、平方和次之比较分段方案，并忽略容差内差异。"""
    if candidate[0] < current[0] - _EPSILON:
        return True
    return abs(candidate[0] - current[0]) <= _EPSILON and candidate[1] < current[1] - _EPSILON


def _prefix_lengths(points: tuple[_XY, ...]) -> tuple[float, ...]:
    """返回每个航点之前累计的折线长度，单位 km。"""
    lengths = [0.0]
    for start, end in zip(points, points[1:]):
        lengths.append(lengths[-1] + _distance(start, end))
    return tuple(lengths)


def _polyline_length(points: Sequence[_XY]) -> float:
    """汇总二维折线相邻点间的欧氏距离，单位 km。"""
    return sum(_distance(start, end) for start, end in zip(points, points[1:]))


# 以下向量小工具统一作用于二维局部 km 坐标；叉积返回有符号标量。
def _add(left: _XY, right: _XY) -> _XY:
    return left[0] + right[0], left[1] + right[1]


def _subtract(left: _XY, right: _XY) -> _XY:
    return left[0] - right[0], left[1] - right[1]


def _scale(point: _XY, scalar: float) -> _XY:
    return point[0] * scalar, point[1] * scalar


def _dot(left: _XY, right: _XY) -> float:
    return left[0] * right[0] + left[1] * right[1]


def _cross(left: _XY, right: _XY) -> float:
    return left[0] * right[1] - left[1] * right[0]


def _norm(point: _XY) -> float:
    return hypot(point[0], point[1])


def _distance(left: _XY, right: _XY) -> float:
    return hypot(left[0] - right[0], left[1] - right[1])
