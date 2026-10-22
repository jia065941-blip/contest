"""Official deployment-agent adapter for the pure staggered planner."""

from __future__ import annotations

import numpy as np

from staggered_deployment_skill import (
    DeploymentBounds,
    PlatformDeploymentRequest,
    StaggeredDeploymentSkill,
    SUPPORTED_PLATFORM_TYPES,
    to_deployment_actions,
)
from user_agents.base_agent import AgentType, BaseAgent


def rectangle_bounds(regions) -> tuple[float, float, float, float]:
    """Validate exactly one axis-aligned rectangle and return its bounds."""
    if not isinstance(regions, (list, tuple)) or len(regions) != 1:
        raise ValueError("staggered deployment requires exactly one rectangle")
    polygon = regions[0]
    if not isinstance(polygon, (list, tuple)) or len(polygon) not in (4, 5):
        raise ValueError("deployment polygon must have four rectangle vertices")
    points = [(float(point[0]), float(point[1])) for point in polygon]
    if len(points) == 5:
        if points[0] != points[-1]:
            raise ValueError("five-point deployment rectangle must be closed")
        points.pop()
    lon_min, lon_max = min(x for x, _ in points), max(x for x, _ in points)
    lat_min, lat_max = min(y for _, y in points), max(y for _, y in points)
    corners = {(lon_min, lat_min), (lon_min, lat_max), (lon_max, lat_min), (lon_max, lat_max)}
    if len(set(points)) != 4 or set(points) != corners:
        raise ValueError("deployment polygon is not an axis-aligned rectangle")
    return lon_min, lon_max, lat_min, lat_max


class StaggeredDeployAgent(BaseAgent):
    def __init__(self, coordinates, coordinates_hm, *, config, batch_size: int, launch_altitude):
        super().__init__(-1, -1, AgentType.DEPLOY, {})
        bounds = rectangle_bounds(coordinates)
        hm_bounds = rectangle_bounds(coordinates_hm or coordinates)
        if bounds != hm_bounds:
            raise ValueError("L and H/M deployment rectangles must be identical")
        self._raw_bounds = bounds
        self._altitude = launch_altitude
        self._skill = StaggeredDeploymentSkill(config)
        self._batch_size = int(batch_size)
        self._actions: tuple[tuple[int, int, float, float, float], ...] | None = None
        self._index = 0
        self.latest_observation = None

    def set_observation(self, observation: dict) -> None:
        self.latest_observation = observation

    def _plan(self) -> None:
        if self.latest_observation is None:
            raise RuntimeError("deployment observation has not been provided")
        entities = self.latest_observation.get("entities", {})
        selected = tuple(
            (int(entity_id), entity)
            for entity_id, entity in sorted(entities.items(), key=lambda item: int(item[0]))
            if int(entity.get("type", -1)) in SUPPORTED_PLATFORM_TYPES
        )
        requests = tuple(
            PlatformDeploymentRequest(entity_id, int(entity["type"]))
            for entity_id, entity in selected
        )
        if not requests:
            raise ValueError("deployment observation contains no H/M/L platforms")
        altitude = self._altitude
        if altitude is None:
            values = {float(entity["position"]["alt"]) for _, entity in selected}
            if len(values) != 1:
                raise ValueError("platform launch altitudes differ and no launch altitude was supplied")
            altitude = values.pop()
        plan = self._skill.plan(DeploymentBounds(*self._raw_bounds, float(altitude)), requests)
        rows = to_deployment_actions(plan)
        if len(rows) != len(requests) or {row[1] for row in rows} != {item.entity_id for item in requests}:
            raise RuntimeError("deployment planner did not return every platform exactly once")
        self._actions = rows

    def get_action(self) -> np.ndarray:
        if self._actions is None:
            self._plan()
        assert self._actions is not None
        rows = self._actions[self._index:self._index + self._batch_size]
        self._index += len(rows)
        if not rows:
            return np.asarray([[1, -1, 0, 0, 0]], dtype=np.float64)
        return np.asarray(rows, dtype=np.float64).reshape(-1, 5)

    def reset(self):
        super().reset()
        self._actions = None
        self._index = 0
        self.latest_observation = None
