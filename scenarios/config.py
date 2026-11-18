"""一次场景加载产生的物理初态、部署约束和比赛规则。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .cases import RewardPolicy

if TYPE_CHECKING:
    from envengine.sdk.base_struct.profile.profile import Profile


Coordinates = tuple[tuple[tuple[float, float], ...], ...]


@dataclass(frozen=True)
class DeploymentConstraints:
    """本场景的部署区域和可选部署高度，坐标容器统一为嵌套元组。"""

    coordinates: Coordinates
    launch_altitude: float | None = None

    def __post_init__(self) -> None:
        # 单独冻结坐标容器，使其与可变 Profile 中的坐标列表互相独立。
        coordinates = self.coordinates
        if coordinates and coordinates[0] and isinstance(coordinates[0][0], (int, float)):
            coordinates = (coordinates,)
        object.__setattr__(
            self, "coordinates",
            tuple(tuple(tuple(point) for point in polygon) for polygon in coordinates),
        )


@dataclass(frozen=True)
class ScenarioConfig:
    """场景配置；每次加载拥有独立的 Profile 和规则数据。"""

    path: Path
    profile: Profile
    deployment: DeploymentConstraints
    reward_policy: RewardPolicy | None

    @property
    def asset_values(self) -> dict[int, float]:
        """按需提供保护对象价值副本，供蓝方策略初始化。"""
        if self.reward_policy is None:
            return {}
        return dict(self.reward_policy.objective_weights)
