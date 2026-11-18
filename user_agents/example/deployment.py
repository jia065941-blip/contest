"""红方独立部署示例：按场景区域生成随机位置，或读取网格数量方案。

deploy 接收管理器给出的受控实体、部署约束与配置，返回初始位置映射。
这里负责计算布局，位置的完整校验、应用和红方单元启动由 AgentManager 完成。
"""

import json
from pathlib import Path
import random

from scenarios.grid_maps.deployment import compile_positions
from user_agents.base import BaseDeployment


# 示例处理的三类红方平台：H 毁灭者、M 突击者、L 清道夫。
RED_PLATFORM_TYPES = (21000, 21001, 21002)


def _point_in_polygon(x: float, y: float, polygon) -> bool:
    """用射线交点奇偶判断平面坐标是否位于二维多边形内，边界点视为在区域内。"""
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = float(previous[0]), float(previous[1])
        x2, y2 = float(current[0]), float(current[1])
        # 先识别位于线段上的点，避免边界位置被后续的交点判断漏掉。
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if (
            abs(cross) <= 1e-12
            and min(x1, x2) - 1e-12 <= x <= max(x1, x2) + 1e-12
            and min(y1, y2) - 1e-12 <= y <= max(y1, y2) + 1e-12
        ):
            return True
        if (y1 > y) != (y2 > y):
            # 边跨过当前水平线时计算交点；交点在点右侧则翻转内外状态。
            intersection = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < intersection:
                inside = not inside
        previous = current
    return inside


def random_point_in_region(coordinates: list[list[float]], rng) -> tuple[float, float]:
    """在单个部署多边形的包围盒内拒绝采样，返回以米为单位的 x、y 坐标。

    rng 由部署对象持有，使多次部署可以延续同一随机流。
    """
    if not coordinates:
        raise ValueError("红方部署区域不能为空")
    if len(coordinates) < 3:
        raise ValueError("红方部署区域至少需要三个顶点")
    xs = [float(point[0]) for point in coordinates]
    ys = [float(point[1]) for point in coordinates]
    # 有限次尝试用于发现过窄或退化的区域，成功时立即返回区域内坐标。
    for _ in range(10_000):
        x = rng.uniform(min(xs), max(xs))
        y = rng.uniform(min(ys), max(ys))
        if _point_in_polygon(x, y, coordinates):
            return x, y
    raise ValueError("无法在红方部署区域内生成合法位置")


def build_deployment_positions(
    entities, coordinates: list[list[float]], rng, plan=None,
) -> dict[int, tuple[float, float]]:
    """按受控实体原始顺序生成随机位置，或编译整队网格数量方案。

    entities 使用实体编号作键，值只需提供 type；三类平台使用同一部署区域。
    输出只包含 x、y，单位为米；管理器应用方案时保留各实体原始高度。
    """
    # 网格编译只需本次受控实体的编号与类型。
    controlled = {
        entity_id: {"type": entity["type"]}
        for entity_id, entity in entities.items()
        if entity.get("type") in RED_PLATFORM_TYPES
    }
    if plan is not None:
        return compile_positions(plan, controlled, coordinates)

    positions = {}
    for entity_id in controlled:
        # H/M/L 均从统一部署区采样，高度由管理器从原生实体读取。
        x, y = random_point_in_region(coordinates, rng)
        positions[entity_id] = (x, y)
    return positions


class Deployment(BaseDeployment):
    """独立部署整队单位，本对象跨局延续自己的随机流。

    内置策略入口导入此类作为 Deployment 导出。普通 Simulator 每次运行局部创建它；
    示例训练 runner 可跨局复用同一对象，从同一随机源中连续抽取布局。
    """

    # 示例可自行指定网格方案文件；未指定时从统一部署区随机采样。
    plan_path = None

    def __init__(self):
        """延后到首次 deploy 取得种子，并缓存已读取的网格方案文件。"""
        self._rng = None
        self._plans: dict[Path, dict] = {}

    def deploy(self, observation: dict) -> dict[int, tuple[float, float]]:
        """读取一次部署观测，返回本局完整布局，供管理器统一检查和应用。"""
        # 首次部署初始化随机源，每次根据当前待部署清单生成新位置。
        if self._rng is None:
            seed = observation.get("seed")
            self._rng = random.Random(1 if seed is None else seed)
        source = self.plan_path
        plan = None
        if source is not None:
            # 同一网格方案只读取一次，方案编译仍使用本次实体与部署范围。
            path = Path(source).expanduser().resolve()
            if path not in self._plans:
                self._plans[path] = json.loads(path.read_text(encoding="utf-8"))
            plan = self._plans[path]
        return build_deployment_positions(
            observation["entities"], observation["coordinates"], self._rng, plan,
        )
