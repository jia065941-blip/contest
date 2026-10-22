"""Pure deterministic geometry for staggered H/M/L deployment."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


H_PLATFORM_TYPE = 21000
M_PLATFORM_TYPE = 21001
L_PLATFORM_TYPE = 21002
SUPPORTED_PLATFORM_TYPES = frozenset(
    (H_PLATFORM_TYPE, M_PLATFORM_TYPE, L_PLATFORM_TYPE)
)

_KM_PER_LATITUDE_DEGREE = 110.574
_KM_PER_LONGITUDE_DEGREE_AT_EQUATOR = 111.320
_EPSILON_KM = 1e-9


def _finite(value: float, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _positive(value: float, field: str) -> float:
    result = _finite(value, field)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive")
    return result


@dataclass(frozen=True)
class DeploymentBounds:
    """Axis-aligned WGS84 deployment rectangle and common altitude."""

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    altitude: float

    def __post_init__(self) -> None:
        for name in ("lon_min", "lon_max", "lat_min", "lat_max", "altitude"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if not -180.0 <= self.lon_min < self.lon_max <= 180.0:
            raise ValueError("longitude bounds must satisfy -180 <= min < max <= 180")
        if not -90.0 <= self.lat_min < self.lat_max <= 90.0:
            raise ValueError("latitude bounds must satisfy -90 <= min < max <= 90")
        if self.altitude < 0.0:
            raise ValueError("altitude must be non-negative")
        if self.km_per_longitude_degree <= _EPSILON_KM:
            raise ValueError("deployment bounds are too close to a pole")

    @property
    def midpoint_latitude(self) -> float:
        return (self.lat_min + self.lat_max) / 2.0

    @property
    def km_per_longitude_degree(self) -> float:
        return _KM_PER_LONGITUDE_DEGREE_AT_EQUATOR * math.cos(
            math.radians(self.midpoint_latitude)
        )

    @property
    def width_km(self) -> float:
        return (self.lon_max - self.lon_min) * self.km_per_longitude_degree

    @property
    def height_km(self) -> float:
        return (self.lat_max - self.lat_min) * _KM_PER_LATITUDE_DEGREE


@dataclass(frozen=True)
class PlatformDeploymentRequest:
    """Identity and supported H/M/L type for one platform."""

    entity_id: int
    entity_type: int

    def __post_init__(self) -> None:
        if type(self.entity_id) is not int or self.entity_id < 0:
            raise ValueError("entity_id must be a non-negative integer")
        if type(self.entity_type) is not int or self.entity_type not in SUPPORTED_PLATFORM_TYPES:
            raise ValueError("entity_type must be one of 21000, 21001, or 21002")


@dataclass(frozen=True)
class StaggeredDeploymentConfig:
    """Geometry configuration expressed in local tangent-plane kilometres."""

    front_axis: str = "east"
    deployment_gap_km: float = 30.0
    l_depth_km: float = 5.0
    hm_depth_km: float = 10.0
    lateral_spacing_km: float = 1.0

    def __post_init__(self) -> None:
        axis = str(self.front_axis).lower()
        if axis not in {"east", "west", "north", "south"}:
            raise ValueError("front_axis must be east, west, north, or south")
        object.__setattr__(self, "front_axis", axis)
        for name in (
            "deployment_gap_km",
            "l_depth_km",
            "hm_depth_km",
            "lateral_spacing_km",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))


@dataclass(frozen=True)
class DeploymentPoint:
    """Semantic position for one platform in the front or rear band."""

    entity_id: int
    entity_type: int
    longitude: float
    latitude: float
    altitude: float
    band: str
    column: int
    row: int
    forward_km: float
    lateral_km: float

    def __post_init__(self) -> None:
        if type(self.entity_id) is not int or self.entity_id < 0:
            raise ValueError("entity_id must be a non-negative integer")
        if type(self.entity_type) is not int or self.entity_type not in SUPPORTED_PLATFORM_TYPES:
            raise ValueError("entity_type must be one of 21000, 21001, or 21002")
        if self.band not in {"front", "rear"}:
            raise ValueError("band must be front or rear")
        if type(self.column) is not int or self.column < 0:
            raise ValueError("column must be a non-negative integer")
        if type(self.row) is not int or self.row < 0:
            raise ValueError("row must be a non-negative integer")
        for name in (
            "longitude",
            "latitude",
            "altitude",
            "forward_km",
            "lateral_km",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))


@dataclass(frozen=True)
class DeploymentPlan:
    """Complete immutable deployment result ordered by entity ID."""

    bounds: DeploymentBounds
    config: StaggeredDeploymentConfig
    points: tuple[DeploymentPoint, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.bounds, DeploymentBounds):
            raise ValueError("bounds must be DeploymentBounds")
        if not isinstance(self.config, StaggeredDeploymentConfig):
            raise ValueError("config must be StaggeredDeploymentConfig")
        points = tuple(self.points)
        if any(not isinstance(point, DeploymentPoint) for point in points):
            raise ValueError("points must contain only DeploymentPoint values")
        ids = [point.entity_id for point in points]
        if len(ids) != len(set(ids)):
            raise ValueError("deployment point entity IDs must be unique")
        object.__setattr__(self, "points", points)


class StaggeredDeploymentSkill:
    """Pack L ahead of H/M while maintaining a configured empty-axis gap."""

    def __init__(self, config: StaggeredDeploymentConfig | None = None) -> None:
        if config is not None and not isinstance(config, StaggeredDeploymentConfig):
            raise ValueError("config must be StaggeredDeploymentConfig")
        self.config = config or StaggeredDeploymentConfig()

    def plan(
        self,
        bounds: DeploymentBounds,
        requests: Iterable[PlatformDeploymentRequest],
    ) -> DeploymentPlan:
        if not isinstance(bounds, DeploymentBounds):
            raise ValueError("bounds must be DeploymentBounds")
        try:
            normalized = tuple(requests)
        except TypeError as exc:
            raise ValueError("requests must be an iterable of deployment requests") from exc
        if any(not isinstance(item, PlatformDeploymentRequest) for item in normalized):
            raise ValueError("requests must contain only PlatformDeploymentRequest values")
        ids = [item.entity_id for item in normalized]
        if len(ids) != len(set(ids)):
            raise ValueError("deployment request entity IDs must be unique")

        axis_length, lateral_length = self._dimensions(bounds)
        l_low = axis_length - self.config.l_depth_km
        hm_high = l_low - self.config.deployment_gap_km
        hm_low = max(0.0, hm_high - self.config.hm_depth_km)
        if l_low <= _EPSILON_KM or hm_high <= _EPSILON_KM:
            required = self.config.l_depth_km + self.config.deployment_gap_km
            raise ValueError(
                f"deployment rectangle is too narrow on {self.config.front_axis} axis; "
                f"requires more than {required:g} km"
            )
        if hm_high > l_low or l_low - hm_high + _EPSILON_KM < self.config.deployment_gap_km:
            raise ValueError("front and rear deployment bands overlap")

        front = tuple(
            sorted(
                (item for item in normalized if item.entity_type == L_PLATFORM_TYPE),
                key=lambda item: item.entity_id,
            )
        )
        rear = tuple(
            sorted(
                (item for item in normalized if item.entity_type != L_PLATFORM_TYPE),
                key=lambda item: item.entity_id,
            )
        )
        points = self._place_group(bounds, front, "front", l_low, axis_length, lateral_length)
        points += self._place_group(bounds, rear, "rear", hm_low, hm_high, lateral_length)
        return DeploymentPlan(bounds, self.config, tuple(sorted(points, key=lambda item: item.entity_id)))

    def _dimensions(self, bounds: DeploymentBounds) -> tuple[float, float]:
        if self.config.front_axis in {"east", "west"}:
            return bounds.width_km, bounds.height_km
        return bounds.height_km, bounds.width_km

    def _place_group(
        self,
        bounds: DeploymentBounds,
        requests: tuple[PlatformDeploymentRequest, ...],
        band: str,
        band_low: float,
        band_high: float,
        lateral_length: float,
    ) -> tuple[DeploymentPoint, ...]:
        if not requests:
            return ()
        spacing = self.config.lateral_spacing_km
        band_depth = band_high - band_low
        row_capacity = math.floor((lateral_length + _EPSILON_KM) / spacing) + 1
        column_capacity = math.floor((band_depth + _EPSILON_KM) / spacing) + 1
        capacity = row_capacity * column_capacity
        if len(requests) > capacity:
            raise ValueError(
                f"{band} band capacity {capacity} is smaller than {len(requests)} requests"
            )

        column_count = math.ceil(len(requests) / row_capacity)
        used_axis_span = (column_count - 1) * spacing
        axis_start = (band_low + band_high - used_axis_span) / 2.0
        result: list[DeploymentPoint] = []
        for column in range(column_count):
            first = column * row_capacity
            column_requests = requests[first : first + row_capacity]
            used_lateral_span = (len(column_requests) - 1) * spacing
            lateral_start = (lateral_length - used_lateral_span) / 2.0
            forward = axis_start + column * spacing
            for row, request in enumerate(column_requests):
                lateral = lateral_start + row * spacing
                longitude, latitude = self._to_geographic(
                    bounds, forward, lateral
                )
                result.append(
                    DeploymentPoint(
                        entity_id=request.entity_id,
                        entity_type=request.entity_type,
                        longitude=longitude,
                        latitude=latitude,
                        altitude=bounds.altitude,
                        band=band,
                        column=column,
                        row=row,
                        forward_km=forward,
                        lateral_km=lateral,
                    )
                )
        return tuple(result)

    def _to_geographic(
        self, bounds: DeploymentBounds, forward: float, lateral: float
    ) -> tuple[float, float]:
        axis = self.config.front_axis
        if axis == "east":
            east, north = forward, lateral
        elif axis == "west":
            east, north = bounds.width_km - forward, lateral
        elif axis == "north":
            east, north = lateral, forward
        else:
            east, north = lateral, bounds.height_km - forward
        longitude = bounds.lon_min + east / bounds.km_per_longitude_degree
        latitude = bounds.lat_min + north / _KM_PER_LATITUDE_DEGREE
        return (
            min(bounds.lon_max, max(bounds.lon_min, longitude)),
            min(bounds.lat_max, max(bounds.lat_min, latitude)),
        )


__all__ = [
    "DeploymentBounds",
    "DeploymentPlan",
    "DeploymentPoint",
    "H_PLATFORM_TYPE",
    "L_PLATFORM_TYPE",
    "M_PLATFORM_TYPE",
    "PlatformDeploymentRequest",
    "SUPPORTED_PLATFORM_TYPES",
    "StaggeredDeploymentConfig",
    "StaggeredDeploymentSkill",
]
