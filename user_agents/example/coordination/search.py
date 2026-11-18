"""在 R7 进攻任务分配上增加清道夫三机静态搜索任务。

首次决策从当前 BaselineObservation 中挑选低空平台，结合公开区域和 9600
哨卫位置安排编队目的地。已分配的搜索机继续执行航点任务；当收到 9500
目标记录且距离满足条件时，再以目标编号提出改打申请。其余平台复用 R7
进攻任务分配。几何计算使用场景 Position 字段，水平距离按局部投影折算为 km。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from itertools import combinations, permutations
from math import ceil, hypot, sqrt

from ..common.baselines import (
    Assignment,
    BaselineObservation,
    BaselineRules,
    PlatformState,
    R7StrikePackagePolicy,
    TargetPrior,
    distance_km,
)
from ..common.contracts import Position
from .coverage import radial_fan_endpoints


logger = logging.getLogger(__name__)

# 在朝向区域的角域上采样 73 个方向，为不同三机组挑选路线候选。
_CANDIDATE_GROUP_CENTERS = 73
# 以下均为水平距离阈值；Position 仍沿用场景的引擎坐标字段。
_MINIMUM_SEARCH_ROUTE_KM = 12.0  # 缩短候选搜索航线时保留的最小长度。
_SEARCH_ROUTE_LENGTH_STEP_KM = 2.5  # 每次由边界向内缩短的步长。
_DIRECT_RETARGET_RANGE_KM = 10.0  # 搜索机可直接认领已收到暗巢目标的距离。
_SEARCH_DANGER_RADIUS_KM = 10.0  # 起点及直达路线避开的公开哨卫半径。


@dataclass(frozen=True)
class SearchGroup:
    """一个三机搜索组：平台编号顺序与三个 Position 目的地逐项对应。"""

    group_id: int
    platform_ids: tuple[int, int, int]
    destinations: tuple[Position, Position, Position]


class R7StaticSearchPolicy(R7StrikePackagePolicy):
    """组合三机搜索路线与 R7 的 H/M 进攻任务分配。

    路线构造读取公开区域、多平台起点和已知哨卫位置；改打读取调用方传入的
    observed_targets。实例缓存首次搜索计划、已认领目标和已改打平台，
    同一实例首次规划后的搜索分组保持不变。
    """

    def __init__(self, rules: BaselineRules, search_polygon: tuple[Position, ...]) -> None:
        """接收本通信群的规则与公开搜索边界，准备本局搜索缓存。"""
        super().__init__(rules)
        if rules.static_search_group_size != 3:
            raise ValueError("R7 static search currently requires three-platform groups")
        if len(search_polygon) < 3:
            raise ValueError("R7 static search requires a public map polygon")
        # 几何实现固定为三机编队；群内策略实例持有自己的计划及认领记录。
        self.search_polygon = search_polygon
        self.search_groups: tuple[SearchGroup, ...] = ()
        self._search_planned = False
        self._claimed_ship_ids: set[int] = set()  # 避免多台搜索机认领同一暗巢。
        self._retargeted_platform_ids: set[int] = set()  # 已从搜索转入进攻的机号。

    @property
    def search_platform_ids(self) -> frozenset[int]:
        """从当前分组提取全部搜索机编号，供执行与重新分配时判断身份。"""
        return frozenset(
            platform_id
            for group in self.search_groups
            for platform_id in group.platform_ids
        )

    def decide(self, observation: BaselineObservation) -> tuple[Assignment, ...]:
        """首步建立搜索任务，此后将仍在搜索的机号从普通 R7 分配中排除。"""
        # 即使首轮找不到合适路线，也记录规划已执行，后续直接使用普通 R7。
        if not self._search_planned:
            self._search_planned = True
            search_assignments = self._build_initial_search(observation)
            if search_assignments:
                logger.info(
                    "R7 static search assigned %d groups (%d platforms)",
                    len(self.search_groups),
                    len(search_assignments),
                )
            else:
                logger.warning(
                    "R7 static search found no safe connected group; using ordinary R7 immediately"
                )
        else:
            search_assignments = ()

        # 仍在搜索的机号保留原航点任务；已改打机与其余机型进入 R7 普通分配。
        searching_ids = self.search_platform_ids - self._retargeted_platform_ids
        strike_observation = replace(
            observation,
            platforms=tuple(item for item in observation.platforms if item.entity_id not in searching_ids),
        )
        return search_assignments + super().decide(strike_observation)

    def claim_retarget(
        self,
        platform: PlatformState,
        observed_targets: tuple[TargetPrior, ...],
    ) -> TargetPrior | None:
        """为尚未改打的搜索机认领一个已接收、10 km 内且未被认领的暗巢。"""

        # 搜索任务身份和认领表属于本策略实例，成功认领后本局不再重复转向。
        if (
            platform.entity_id not in self.search_platform_ids
            or platform.entity_id in self._retargeted_platform_ids
        ):
            return None
        # 保留 observed_targets 的传入顺序；此处取首个符合条件的 9500 记录。
        target = next(
            (
                item
                for item in observed_targets
                if item.entity_type == 9500 and item.entity_id not in self._claimed_ship_ids
                and distance_km(platform.position, item.position) <= _DIRECT_RETARGET_RANGE_KM
            ),
            None,
        )
        if target is None:
            return None
        self._retargeted_platform_ids.add(platform.entity_id)
        self._claimed_ship_ids.add(target.entity_id)
        logger.info(
            "R7 static search platform %d legally detected and retargeted ship %d",
            platform.entity_id,
            target.entity_id,
        )
        return target

    def _best_target(
        self,
        platform: PlatformState,
        targets: tuple[TargetPrior, ...],
        counts: dict[int, int],
    ) -> TargetPrior | None:
        """低空平台优先打尚无分配的已知暗巢，其余情况沿用 R7 目标规则。"""
        if platform.kind == "L":
            detected_ships = [
                item
                for item in targets
                if item.entity_type == 9500 and counts.get(item.entity_id, 0) < 1
            ]
            if detected_ships:
                return min(
                    detected_ships,
                    key=lambda item: (distance_km(platform.position, item.position), item.entity_id),
                )
        return super()._best_target(platform, targets, counts)

    def _build_initial_search(
        self,
        observation: BaselineObservation,
    ) -> tuple[Assignment, ...]:
        """按预算筛选低空三机组，挑选避开哨卫的路线并生成搜索分配。"""
        # 编号排序保证候选生成顺序稳定；平台来自调用方本次提供的活动清单。
        low_platforms = sorted(
            (item for item in observation.active_platforms() if item.kind == "L"),
            key=lambda item: item.entity_id,
        )
        if len(low_platforms) < self.rules.static_search_group_size:
            return ()

        # 比例需求向上凑整到三的倍数，再同时受 probe 预算和可用机数约束。
        low_count = len(low_platforms)
        desired = max(
            self.rules.static_search_min_platforms,
            ceil(
                low_count
                * self.rules.static_search_low_fraction
                / self.rules.static_search_group_size
            )
            * self.rules.static_search_group_size,
        )
        probe_budget = self._reassignment_budget(observation, "probe")
        available = min(desired, probe_budget, len(low_platforms))
        search_count = available - available % self.rules.static_search_group_size
        if search_count == 0:
            return ()

        # 起点先避开公开 9600 哨卫 10 km 范围，再按期望间距生成候选三机组。
        danger_sites = tuple(
            target.position
            for target in observation.targets
            if target.entity_type == 9600
        )
        group_count = search_count // self.rules.static_search_group_size
        eligible_platforms = [
            platform
            for platform in low_platforms
            if all(
                distance_km(platform.position, site)
                >= _SEARCH_DANGER_RADIUS_KM
                for site in danger_sites
            )
        ]
        platform_groups = _candidate_platform_groups(
            eligible_platforms,
            self.rules.static_search_group_spacing_km,
        )
        if not platform_groups:
            logger.info(
                "R7 static search has no communication-safe triple outside the %.0f km danger zones",
                _SEARCH_DANGER_RADIUS_KM,
            )
            return ()

        # 候选按几何间距质量排序；逐组检查路线，同一平台至多进入一个组。
        routed_platforms: list[PlatformState] = []
        routed_destinations: list[Position] = []
        groups: list[SearchGroup] = []
        selected_platform_ids: set[int] = set()
        for platforms in platform_groups:
            platform_ids = {item.entity_id for item in platforms}
            if platform_ids & selected_platform_ids:
                continue
            # 每组三个目的地与平台逐项匹配，路线可行后才占用成员编号。
            destinations = self._safe_destination_group(
                platforms,
                danger_sites,
                len(groups),
                group_count,
            )
            if destinations is None:
                continue
            group_id = len(groups) + 1
            group_platform_ids = tuple(item.entity_id for item in platforms)
            groups.append(SearchGroup(group_id, group_platform_ids, destinations))
            logger.info(
                "R7 static search group %d platforms=%s starts=%s destinations=%s",
                group_id,
                group_platform_ids,
                tuple(
                    (round(item.position.x, 4), round(item.position.y, 4))
                    for item in platforms
                ),
                tuple(
                    (round(item.x, 4), round(item.y, 4))
                    for item in destinations
                ),
            )
            routed_platforms.extend(platforms)
            routed_destinations.extend(destinations)
            selected_platform_ids.update(platform_ids)
            if len(groups) == group_count:
                break
        if not groups:
            logger.info(
                "R7 static search rejected all %d connected triples because their routes entered a danger zone",
                len(platform_groups),
            )
            return ()

        # 搜索 Assignment 使用 destination 和 search_group_id，target_id 为空。
        # 下游单机执行组件据此产生搜索航点动作，而不是目标编号攻击动作。
        assignments = tuple(
            Assignment(
                platform_id=platform.entity_id,
                target_id=None,
                score=0.0,
                destination=destination,
                search_group_id=1 + index // self.rules.static_search_group_size,
            )
            for index, (platform, destination) in enumerate(
                zip(routed_platforms, routed_destinations)
            )
        )
        self.search_groups = tuple(groups)
        return assignments

    def _safe_destination_group(
        self,
        platforms: tuple[PlatformState, PlatformState, PlatformState],
        danger_sites: tuple[Position, ...],
        group_index: int,
        group_count: int,
    ) -> tuple[Position, Position, Position] | None:
        """为三机组寻找区域内、向东且避开哨卫的编队目的地。"""
        # 用三机初始位置均值作为组中心，向区域边界采样扇形路线。
        origin = Position(
            x=sum(item.position.x for item in platforms) / 3.0,
            y=sum(item.position.y for item in platforms) / 3.0,
            z=sum(item.position.z for item in platforms) / 3.0,
        )
        centers = radial_fan_endpoints(
            origin,
            self.search_polygon,
            count=_CANDIDATE_GROUP_CENTERS,
            max_range_km=self.rules.static_search_max_range_km,
        )
        # 每个方向从最长候选开始逐步缩短，使三角编队有机会完整落入区域。
        candidates: list[tuple[Position, Position, Position]] = []
        for boundary_center in centers:
            maximum_distance = distance_km(origin, boundary_center)
            for route_distance in _descending_route_distances(maximum_distance):
                center = _interpolate_position(
                    origin,
                    boundary_center,
                    route_distance / maximum_distance,
                )
                formation = _equilateral_destinations(
                    origin,
                    center,
                    self.rules.static_search_group_spacing_km,
                )
                # 同时检查编队三个终点和每机起点到终点的直线段。
                if not all(_inside_convex_polygon(item, self.search_polygon) for item in formation):
                    continue
                destinations = _safe_destination_order(
                    platforms,
                    formation,
                    danger_sites,
                    _SEARCH_DANGER_RADIUS_KM,
                )
                if destinations is None:
                    continue
                if any(
                    destination.x <= platform.position.x
                    for platform, destination in zip(platforms, destinations)
                ):
                    continue
                # 每个方向保留第一条可行路线，即该方向当前找到的最长向东路线。
                candidates.append(destinations)
                break

        if not candidates:
            return None
        # 多组沿候选角域分散选择方向；只有一组时使用候选序列的中间方向。
        preferred_index = (
            len(candidates) // 2
            if group_count == 1
            else round(group_index * (len(candidates) - 1) / (group_count - 1))
        )
        candidate_indexes = sorted(
            range(len(candidates)),
            key=lambda index: (abs(index - preferred_index), index),
        )
        return candidates[candidate_indexes[0]]


def _descending_route_distances(maximum_distance_km: float) -> tuple[float, ...]:
    """按 2.5 km 步长从最远距离缩短到 12 km，补齐精确最短端点。"""
    if maximum_distance_km < _MINIMUM_SEARCH_ROUTE_KM:
        return ()
    count = int(
        (maximum_distance_km - _MINIMUM_SEARCH_ROUTE_KM)
        // _SEARCH_ROUTE_LENGTH_STEP_KM
    )
    distances = tuple(
        maximum_distance_km - index * _SEARCH_ROUTE_LENGTH_STEP_KM
        for index in range(count + 1)
    )
    if distances[-1] > _MINIMUM_SEARCH_ROUTE_KM + 1e-6:
        return distances + (_MINIMUM_SEARCH_ROUTE_KM,)
    return distances


def _interpolate_position(start: Position, end: Position, fraction: float) -> Position:
    """逐分量插值 Position，供候选路线沿起点到边界线段缩短。"""
    return Position(
        x=start.x + fraction * (end.x - start.x),
        y=start.y + fraction * (end.y - start.y),
        z=start.z + fraction * (end.z - start.z),
    )


def _candidate_platform_groups(
    platforms: list[PlatformState],
    preferred_spacing_km: float,
) -> tuple[tuple[PlatformState, PlatformState, PlatformState], ...]:
    """生成两两距离处于阈值内的三机候选，并按接近期望间距排序。

    每个平台作为一次锚点，只与最近似期望间距的前 16 个邻机组合。
    此处保留可重叠候选，调用方随后检查路线并选出成员互不重复的组。
    """

    candidates: dict[
        tuple[int, int, int],
        tuple[
            tuple[float, float, tuple[int, int, int]],
            tuple[PlatformState, PlatformState, PlatformState],
        ],
    ] = {}
    # 这里的两倍期望间距是几何筛选阈值；通信群成员资格由上游观测组织决定。
    communication_limit = 2.0 * preferred_spacing_km
    for anchor in platforms:
        neighbors = sorted(
            (item for item in platforms if item.entity_id != anchor.entity_id),
            key=lambda item: (
                abs(distance_km(anchor.position, item.position) - preferred_spacing_km),
                item.entity_id,
            ),
        )[:16]
        for left, right in combinations(neighbors, 2):
            group = tuple(sorted((anchor, left, right), key=lambda item: item.entity_id))
            group_ids = tuple(item.entity_id for item in group)
            distances = tuple(
                distance_km(first.position, second.position)
                for first, second in combinations(group, 2)
            )
            # 初始位置可以重合；期望间距主要约束飞行终点编队，起点只检查上限。
            if max(distances) > communication_limit:
                continue
            # 先比较最差边的间距误差，再比较总误差，最后用编号稳定消除并列。
            score = (
                max(abs(value - preferred_spacing_km) for value in distances),
                sum(abs(value - preferred_spacing_km) for value in distances),
                group_ids,
            )
            existing = candidates.get(group_ids)
            if existing is None or score < existing[0]:
                candidates[group_ids] = (score, group)
    return tuple(
        group
        for _, group in sorted(candidates.values(), key=lambda item: item[0])
    )


def _safe_destination_order(
    platforms: tuple[PlatformState, PlatformState, PlatformState],
    destinations: tuple[Position, Position, Position],
    danger_sites: tuple[Position, ...],
    danger_radius_km: float,
) -> tuple[Position, Position, Position] | None:
    """枚举三个终点的六种匹配，选择直线路径避开危险圆且总距离最短的一种。"""
    safe_orders: list[tuple[float, tuple[Position, Position, Position]]] = []
    # 用点到线段距离检查整个飞行段，避免只检查终点而漏过途中哨卫范围。
    for destination_order in permutations(destinations):
        if any(
            _point_to_segment_distance_km(site, platform.position, destination)
            < danger_radius_km
            for site in danger_sites
            for platform, destination in zip(platforms, destination_order)
        ):
            continue
        safe_orders.append(
            (
                sum(
                    distance_km(platform.position, destination)
                    for platform, destination in zip(platforms, destination_order)
                ),
                destination_order,
            )
        )
    if not safe_orders:
        return None
    return min(safe_orders, key=lambda item: item[0])[1]


def _equilateral_destinations(
    origin: Position,
    center: Position,
    spacing_km: float,
) -> tuple[Position, Position, Position]:
    """以路线方向生成边长 spacing_km 的三角形终点，沿用 origin 高度。"""
    # center 在此作为左右两个点的中点；后点沿路线反向偏移三角形的高。
    center_xy, scale = _relative_xy(center, origin)
    route_length = hypot(*center_xy)
    if route_length <= 0:
        raise ValueError("search destination must differ from its launch position")
    # 局部 forward/lateral 为正交单位轴，后、左、右三点以 km 偏移构造。
    forward = center_xy[0] / route_length, center_xy[1] / route_length
    lateral = -forward[1], forward[0]
    half_spacing = spacing_km / 2.0
    rear_offset = sqrt(3.0) * half_spacing
    rear = (
        center_xy[0] - rear_offset * forward[0],
        center_xy[1] - rear_offset * forward[1],
    )
    left = (
        center_xy[0] + half_spacing * lateral[0],
        center_xy[1] + half_spacing * lateral[1],
    )
    right = (
        center_xy[0] - half_spacing * lateral[0],
        center_xy[1] - half_spacing * lateral[1],
    )
    return tuple(_position_from_relative_xy(point, origin, scale) for point in (rear, left, right))


def _point_to_segment_distance_km(point: Position, start: Position, end: Position) -> float:
    """将三点投影到同一局部平面，求点到有限线段的最短水平距离，单位 km。"""
    point_xy, scale = _relative_xy(point, start)
    end_xy, _ = _relative_xy(end, start, scale)
    length_squared = end_xy[0] ** 2 + end_xy[1] ** 2
    if length_squared <= 0:
        return hypot(*point_xy)
    # 投影参数截断到 [0, 1]，最近点可以是线段端点。
    fraction = max(
        0.0,
        min(
            1.0,
            (point_xy[0] * end_xy[0] + point_xy[1] * end_xy[1]) / length_squared,
        ),
    )
    return hypot(
        point_xy[0] - fraction * end_xy[0],
        point_xy[1] - fraction * end_xy[1],
    )


def _inside_convex_polygon(point: Position, polygon: tuple[Position, ...]) -> bool:
    """在同一局部投影中检查凸区域有向边符号，允许点落在边界上。"""
    point_xy, scale = _relative_xy(point, polygon[0])
    polygon_xy = tuple(_relative_xy(item, polygon[0], scale)[0] for item in polygon)
    signs: set[int] = set()
    for start, end in zip(polygon_xy, polygon_xy[1:] + polygon_xy[:1]):
        cross = (
            (end[0] - start[0]) * (point_xy[1] - start[1])
            - (end[1] - start[1]) * (point_xy[0] - start[0])
        )
        if abs(cross) > 1e-7:
            signs.add(1 if cross > 0 else -1)
    return len(signs) <= 1


def _relative_xy(
    point: Position,
    origin: Position,
    scale: tuple[float, float] | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """返回相对起点的二维 km 坐标，以及米到千米的换算比例。"""
    scale = (0.001, 0.001) if scale is None else scale
    return (
        (point.x - origin.x) * scale[0],
        (point.y - origin.y) * scale[1],
    ), scale


def _position_from_relative_xy(
    point: tuple[float, float],
    origin: Position,
    scale: tuple[float, float],
) -> Position:
    """将相对二维 km 点恢复为场景 Position，沿用起点高度。"""
    return Position(
        x=origin.x + point[0] / scale[0],
        y=origin.y + point[1] / scale[1],
        z=origin.z,
    )
