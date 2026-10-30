"""Pure deterministic geometry for staggered H/M/L deployment."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Iterable


H_PLATFORM_TYPE = 21000
M_PLATFORM_TYPE = 21001
L_PLATFORM_TYPE = 21002
SUPPORTED_PLATFORM_TYPES = frozenset(
    (H_PLATFORM_TYPE, M_PLATFORM_TYPE, L_PLATFORM_TYPE)
)
PUBLIC_TARGET_TYPES = frozenset((9400, 9600))
DEPLOYMENT_ROLES = frozenset(("core", "reserve_h", "reserve_l"))
_RESERVE_H_DEPTH_KM = 5.0

_KM_PER_LATITUDE_DEGREE = 110.574
_KM_PER_LONGITUDE_DEGREE_AT_EQUATOR = 111.320
_EARTH_MEAN_RADIUS_KM = 6371.0088
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


def _nonnegative(value: float, field: str) -> float:
    result = _finite(value, field)
    if result < 0.0:
        raise ValueError(f"{field} must be non-negative")
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
    """Identity, supported H/M/L type, and optional public target."""

    entity_id: int
    entity_type: int
    target_lon: float | None = None
    target_lat: float | None = None
    target_id: int | None = None
    target_type: int | None = None
    deployment_role: str = "core"

    def __post_init__(self) -> None:
        if type(self.entity_id) is not int or self.entity_id < 0:
            raise ValueError("entity_id must be a non-negative integer")
        if type(self.entity_type) is not int or self.entity_type not in SUPPORTED_PLATFORM_TYPES:
            raise ValueError("entity_type must be one of 21000, 21001, or 21002")
        deployment_role = str(self.deployment_role).lower()
        if deployment_role not in DEPLOYMENT_ROLES:
            raise ValueError("deployment_role must be core, reserve_h, or reserve_l")
        if deployment_role == "reserve_h" and self.entity_type != H_PLATFORM_TYPE:
            raise ValueError("reserve_h deployment role requires an H platform")
        if deployment_role == "reserve_l" and self.entity_type != L_PLATFORM_TYPE:
            raise ValueError("reserve_l deployment role requires an L platform")
        object.__setattr__(self, "deployment_role", deployment_role)
        if (self.target_lon is None) != (self.target_lat is None):
            raise ValueError("target_lon and target_lat must be provided together")
        if self.target_lon is not None:
            object.__setattr__(self, "target_lon", _finite(self.target_lon, "target_lon"))
            object.__setattr__(self, "target_lat", _finite(self.target_lat, "target_lat"))
            if not -180.0 <= self.target_lon <= 180.0:
                raise ValueError("target_lon must be between -180 and 180")
            if not -90.0 <= self.target_lat <= 90.0:
                raise ValueError("target_lat must be between -90 and 90")
        if (self.target_id is None) != (self.target_type is None):
            raise ValueError("target_id and target_type must be provided together")
        if self.target_id is not None:
            if type(self.target_id) is not int or self.target_id < 0:
                raise ValueError("target_id must be a non-negative integer")
            if type(self.target_type) is not int or self.target_type not in PUBLIC_TARGET_TYPES:
                raise ValueError("target_type must be public 9400 or 9600")
            if deployment_role != "core":
                raise ValueError("reserve coordinates are provisional, not an assignment")
            if self.target_lon is None:
                raise ValueError("target coordinates are required with target_id/target_type")
            if self.entity_type not in {H_PLATFORM_TYPE, M_PLATFORM_TYPE}:
                raise ValueError("only H/M requests may carry public target identity")


@dataclass(frozen=True)
class StaggeredDeploymentConfig:
    """Geometry configuration expressed in local tangent-plane kilometres."""

    front_axis: str = "east"
    deployment_gap_km: float = 15.0
    l_depth_km: float = 5.0
    hm_depth_km: float = 10.0
    lateral_spacing_km: float = 1.0
    l_first_wave_fraction: float = 0.70
    h_depth_km: float = 5.0
    m_depth_km: float = 10.0
    hm_follow_gap_km: float = 0.0
    mother_crossfire_angle_degrees: float = 35.0
    mother_crossfire_max_offset_km: float = 35.0
    sentry_m_lead_km: float = 7.5
    sentry_rendezvous_mode: str = "adaptive"
    h_speed_mps: float = 200.0
    m_speed_mps: float = 120.0
    desired_m_arrival_lead_seconds: float = 5.0
    min_sentry_m_lead_km: float = 10.0
    max_sentry_m_lead_km: float | None = None
    l_reserve_gap_km: float = 8.0
    l_reserve_depth_km: float = 5.0

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
            "h_depth_km",
            "m_depth_km",
            "l_reserve_gap_km",
            "l_reserve_depth_km",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        object.__setattr__(
            self,
            "hm_follow_gap_km",
            _nonnegative(self.hm_follow_gap_km, "hm_follow_gap_km"),
        )
        # Keep the former rear-depth option as an alias for m_depth_km. New
        # callers win when they provide a non-default m_depth_km explicitly.
        if self.hm_depth_km != 10.0 and self.m_depth_km == 10.0:
            object.__setattr__(self, "m_depth_km", self.hm_depth_km)
        else:
            object.__setattr__(self, "hm_depth_km", self.m_depth_km)
        fraction = _finite(self.l_first_wave_fraction, "l_first_wave_fraction")
        if not 0.0 < fraction <= 1.0:
            raise ValueError("l_first_wave_fraction must be in (0, 1]")
        object.__setattr__(self, "l_first_wave_fraction", fraction)
        angle = _finite(
            self.mother_crossfire_angle_degrees,
            "mother_crossfire_angle_degrees",
        )
        if not 30.0 <= angle < 180.0:
            raise ValueError("mother_crossfire_angle_degrees must be in [30, 180)")
        object.__setattr__(self, "mother_crossfire_angle_degrees", angle)
        object.__setattr__(
            self,
            "mother_crossfire_max_offset_km",
            _positive(
                self.mother_crossfire_max_offset_km,
                "mother_crossfire_max_offset_km",
            ),
        )
        sentry_lead = _finite(self.sentry_m_lead_km, "sentry_m_lead_km")
        if not 0.0 <= sentry_lead <= 15.0:
            raise ValueError("sentry_m_lead_km must be in [0, 15]")
        object.__setattr__(self, "sentry_m_lead_km", sentry_lead)
        rendezvous_mode = str(self.sentry_rendezvous_mode).lower()
        if rendezvous_mode not in {"fixed", "adaptive"}:
            raise ValueError("sentry_rendezvous_mode must be fixed or adaptive")
        object.__setattr__(self, "sentry_rendezvous_mode", rendezvous_mode)
        object.__setattr__(
            self, "h_speed_mps", _positive(self.h_speed_mps, "h_speed_mps")
        )
        object.__setattr__(
            self, "m_speed_mps", _positive(self.m_speed_mps, "m_speed_mps")
        )
        object.__setattr__(
            self,
            "desired_m_arrival_lead_seconds",
            _nonnegative(
                self.desired_m_arrival_lead_seconds,
                "desired_m_arrival_lead_seconds",
            ),
        )
        minimum_lead = _nonnegative(
            self.min_sentry_m_lead_km, "min_sentry_m_lead_km"
        )
        object.__setattr__(self, "min_sentry_m_lead_km", minimum_lead)
        if self.max_sentry_m_lead_km is not None:
            maximum_lead = _nonnegative(
                self.max_sentry_m_lead_km, "max_sentry_m_lead_km"
            )
            if maximum_lead < minimum_lead:
                raise ValueError(
                    "max_sentry_m_lead_km must be at least min_sentry_m_lead_km"
                )
            object.__setattr__(self, "max_sentry_m_lead_km", maximum_lead)


@dataclass(frozen=True)
class DeploymentPoint:
    """Semantic position and wave role for one deployed platform."""

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
    wave_role: str | None = None
    target_id: int | None = None
    target_type: int | None = None
    crossfire_role: str | None = None
    crossfire_angle: float | None = None
    formation_role: str | None = None
    desired_separation_km: float | None = None
    actual_separation_km: float | None = None
    requested_lead: float | None = None
    actual_lead: float | None = None
    predicted_m_arrival_delta_seconds: float | None = None
    predicted_separation_at_m_20km: float | None = None
    geometry_clamped: bool | None = None
    clamp_reason: str | None = None
    deployment_role: str = "core"

    def __post_init__(self) -> None:
        if type(self.entity_id) is not int or self.entity_id < 0:
            raise ValueError("entity_id must be a non-negative integer")
        if type(self.entity_type) is not int or self.entity_type not in SUPPORTED_PLATFORM_TYPES:
            raise ValueError("entity_type must be one of 21000, 21001, or 21002")
        deployment_role = str(self.deployment_role).lower()
        if deployment_role not in DEPLOYMENT_ROLES:
            raise ValueError("deployment_role must be core, reserve_h, or reserve_l")
        if deployment_role == "reserve_h" and self.entity_type != H_PLATFORM_TYPE:
            raise ValueError("reserve_h deployment role requires an H platform")
        if deployment_role == "reserve_l" and self.entity_type != L_PLATFORM_TYPE:
            raise ValueError("reserve_l deployment role requires an L platform")
        object.__setattr__(self, "deployment_role", deployment_role)
        if self.band not in {"front", "second", "rear"}:
            raise ValueError("band must be front, second, or rear")
        wave_role = self.wave_role
        if wave_role is None:
            wave_role = {
                "front": "front",
                "second": "second",
            }.get(self.band, "rear")
        if wave_role not in {
            "front", "second", "reserve_l", "h_lead", "m_follow", "rear"
        }:
            raise ValueError(
                "wave_role must be front, second, reserve_l, h_lead, m_follow, or rear"
            )
        object.__setattr__(self, "wave_role", wave_role)
        if (self.target_id is None) != (self.target_type is None):
            raise ValueError("target_id and target_type must be provided together")
        if self.target_id is not None:
            if type(self.target_id) is not int or self.target_id < 0:
                raise ValueError("target_id must be a non-negative integer")
            if type(self.target_type) is not int or self.target_type not in PUBLIC_TARGET_TYPES:
                raise ValueError("target_type must be public 9400 or 9600")
            if deployment_role != "core":
                raise ValueError("reserve coordinates are provisional, not an assignment")
        if self.crossfire_role not in {None, "left", "center", "right"}:
            raise ValueError("crossfire_role must be left, center, right, or None")
        if (self.crossfire_role is None) != (self.crossfire_angle is None):
            raise ValueError("crossfire_role and crossfire_angle must be provided together")
        if self.crossfire_role is not None and (
            self.entity_type != H_PLATFORM_TYPE or self.target_type != 9400
        ):
            raise ValueError("crossfire diagnostics require an H assigned to a 9400 target")
        if self.crossfire_angle is not None:
            angle = _finite(self.crossfire_angle, "crossfire_angle")
            if not 0.0 <= angle <= 180.0:
                raise ValueError("crossfire_angle must be between 0 and 180")
            object.__setattr__(self, "crossfire_angle", angle)
        if self.formation_role not in {None, "m_lead", "h_follow"}:
            raise ValueError("formation_role must be m_lead, h_follow, or None")
        formation_diagnostics = (
            self.formation_role,
            self.desired_separation_km,
            self.actual_separation_km,
        )
        rendezvous_diagnostics = (
            self.requested_lead,
            self.actual_lead,
            self.predicted_m_arrival_delta_seconds,
            self.predicted_separation_at_m_20km,
            self.geometry_clamped,
        )
        if any(value is None for value in formation_diagnostics) != all(
            value is None for value in formation_diagnostics
        ):
            raise ValueError(
                "formation_role and separation diagnostics must be provided together"
            )
        if any(value is None for value in rendezvous_diagnostics) != all(
            value is None for value in rendezvous_diagnostics
        ):
            raise ValueError("rendezvous diagnostics must be provided together")
        if self.formation_role is None and not all(
            value is None for value in rendezvous_diagnostics
        ):
            raise ValueError("rendezvous diagnostics require formation_role")
        if self.requested_lead is None and self.clamp_reason is not None:
            raise ValueError("clamp_reason requires rendezvous diagnostics")
        if self.formation_role is not None:
            expected_type = (
                M_PLATFORM_TYPE if self.formation_role == "m_lead" else H_PLATFORM_TYPE
            )
            if self.target_type != 9600 or self.entity_type != expected_type:
                raise ValueError(
                    "formation diagnostics require a matching H/M assigned to a 9600 target"
            )
            desired = _finite(self.desired_separation_km, "desired_separation_km")
            actual = _finite(self.actual_separation_km, "actual_separation_km")
            if desired < 0.0:
                raise ValueError("desired_separation_km must be non-negative")
            if actual < 0.0:
                raise ValueError("actual_separation_km must be non-negative")
            object.__setattr__(self, "desired_separation_km", desired)
            object.__setattr__(self, "actual_separation_km", actual)
        if self.requested_lead is not None:
            requested = _finite(self.requested_lead, "requested_lead")
            actual_lead = _nonnegative(self.actual_lead, "actual_lead")
            arrival_delta = _finite(
                self.predicted_m_arrival_delta_seconds,
                "predicted_m_arrival_delta_seconds",
            )
            separation_at_20 = _finite(
                self.predicted_separation_at_m_20km,
                "predicted_separation_at_m_20km",
            )
            if type(self.geometry_clamped) is not bool:
                raise ValueError("geometry_clamped must be a boolean")
            if self.clamp_reason not in {None, "boundary", "capacity", "corridor"}:
                raise ValueError(
                    "clamp_reason must be boundary, capacity, corridor, or None"
                )
            if self.geometry_clamped != (self.clamp_reason is not None):
                raise ValueError(
                    "clamp_reason must be present exactly when geometry is clamped"
                )
            object.__setattr__(self, "requested_lead", requested)
            object.__setattr__(self, "actual_lead", actual_lead)
            object.__setattr__(
                self, "predicted_m_arrival_delta_seconds", arrival_delta
            )
            object.__setattr__(
                self, "predicted_separation_at_m_20km", separation_at_20
            )
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
    """Deploy front L and target-aware H/M formations before second-wave L."""

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
        target_catalogue: dict[int, tuple[int, float, float]] = {}
        for item in normalized:
            if item.target_id is None:
                continue
            target = (item.target_type, item.target_lon, item.target_lat)
            known = target_catalogue.setdefault(item.target_id, target)
            if known != target:
                raise ValueError(
                    f"target {item.target_id} must have one consistent type and position"
                )

        axis_length, lateral_length = self._dimensions(bounds)
        required = max(
            self.config.l_depth_km
            + self.config.deployment_gap_km
            + self.config.h_depth_km
            + self.config.hm_follow_gap_km
            + self.config.m_depth_km,
            self.config.l_depth_km / 2.0
            + self.config.l_reserve_gap_km
            + self.config.l_reserve_depth_km / 2.0,
        )
        if axis_length + _EPSILON_KM < required:
            raise ValueError(
                f"deployment rectangle is too narrow on {self.config.front_axis} axis; "
                f"requires at least {required:g} km"
            )

        front_high = axis_length
        front_low = front_high - self.config.l_depth_km
        h_high = front_low - self.config.deployment_gap_km
        h_low = h_high - self.config.h_depth_km
        follow_high = h_low - self.config.hm_follow_gap_km
        follow_low = max(0.0, follow_high - self.config.m_depth_km)
        reserve_l_center = (
            (front_low + front_high) / 2.0
            - self.config.l_reserve_gap_km
        )
        reserve_l_low = reserve_l_center - self.config.l_reserve_depth_km / 2.0
        reserve_l_high = reserve_l_center + self.config.l_reserve_depth_km / 2.0

        l_requests = tuple(
            sorted(
                (item for item in normalized if item.entity_type == L_PLATFORM_TYPE),
                key=lambda item: item.entity_id,
            )
        )
        reserve_l_requests = tuple(
            item for item in l_requests if item.deployment_role == "reserve_l"
        )
        core_l_requests = tuple(
            item for item in l_requests if item.deployment_role == "core"
        )
        if reserve_l_requests:
            first_wave = core_l_requests
            second_wave = ()
        else:
            first_wave_count = math.ceil(
                len(core_l_requests) * self.config.l_first_wave_fraction
            )
            first_wave = core_l_requests[:first_wave_count]
            second_wave = core_l_requests[first_wave_count:]
        reserve_h_requests = tuple(
            sorted(
                (
                    item
                    for item in normalized
                    if item.deployment_role == "reserve_h"
                ),
                key=lambda item: item.entity_id,
            )
        )
        h_requests = tuple(
            sorted(
                (
                    item
                    for item in normalized
                    if item.entity_type == H_PLATFORM_TYPE
                    and item.deployment_role == "core"
                ),
                key=lambda item: item.entity_id,
            )
        )
        m_requests = tuple(
            sorted(
                (item for item in normalized if item.entity_type == M_PLATFORM_TYPE),
                key=lambda item: item.entity_id,
            )
        )
        reserve_h_points = self._place_reserve_h(
            bounds,
            reserve_h_requests,
            lateral_length,
        )
        reserve_coordinates = {
            (point.forward_km, point.lateral_km)
            for point in reserve_h_points
        }
        minimum_core_forward = (
            0.0
            if not reserve_h_points
            else max(point.forward_km for point in reserve_h_points)
            + self.config.lateral_spacing_km
        )
        h_points, m_points = self._place_rear_groups(
            bounds,
            h_requests,
            m_requests,
            h_low,
            h_high,
            follow_low,
            follow_high,
            axis_length,
            lateral_length,
            reserve_coordinates,
            minimum_core_forward,
        )
        hm_points = reserve_h_points + h_points + m_points
        l_points = self._place_l_groups(
            bounds,
            first_wave,
            second_wave,
            reserve_l_requests,
            front_low,
            front_high,
            follow_low,
            follow_high,
            reserve_l_low,
            reserve_l_high,
            axis_length,
            lateral_length,
            {(point.forward_km, point.lateral_km) for point in hm_points},
        )
        points = hm_points + l_points
        coordinates = {(point.longitude, point.latitude) for point in points}
        if len(coordinates) != len(points):
            raise ValueError("deployment geometry produced overlapping coordinates")
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
        wave_role: str | None = None,
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
                        wave_role=wave_role,
                        target_id=request.target_id,
                        target_type=request.target_type,
                        deployment_role=request.deployment_role,
                    )
                )
        return tuple(result)

    def _place_reserve_h(
        self,
        bounds: DeploymentBounds,
        requests: tuple[PlatformDeploymentRequest, ...],
        lateral_length: float,
    ) -> tuple[DeploymentPoint, ...]:
        if not requests:
            return ()
        spacing = self.config.lateral_spacing_km
        row_capacity = math.floor((lateral_length + _EPSILON_KM) / spacing) + 1
        column_count = math.ceil(len(requests) / row_capacity)
        used_forward_span = (column_count - 1) * spacing
        if used_forward_span > _RESERVE_H_DEPTH_KM + _EPSILON_KM:
            capacity = row_capacity * (
                math.floor(_RESERVE_H_DEPTH_KM / spacing) + 1
            )
            raise ValueError(
                f"reserve_h band capacity {capacity} is smaller than "
                f"{len(requests)} requests"
            )
        forward_start = (_RESERVE_H_DEPTH_KM - used_forward_span) / 2.0
        rows = tuple(
            (row, min(lateral_length, row * spacing))
            for row in range(row_capacity)
        )
        result: list[DeploymentPoint] = []
        for column in range(column_count):
            column_requests = requests[
                column * row_capacity:(column + 1) * row_capacity
            ]
            available = dict(rows)
            untargeted_laterals = iter(self._uniform_lateral_positions(
                sum(request.target_lon is None for request in column_requests),
                lateral_length,
            ))
            for request in column_requests:
                target_lateral = (
                    next(untargeted_laterals)
                    if request.target_lon is None
                    else self._target_lateral(bounds, request, lateral_length)
                )
                row = min(
                    available,
                    key=lambda candidate: (
                        abs(available[candidate] - target_lateral),
                        candidate,
                    ),
                )
                lateral = available.pop(row)
                forward = forward_start + column * spacing
                longitude, latitude = self._to_geographic(
                    bounds, forward, lateral
                )
                result.append(DeploymentPoint(
                    entity_id=request.entity_id,
                    entity_type=request.entity_type,
                    longitude=longitude,
                    latitude=latitude,
                    altitude=bounds.altitude,
                    band="rear",
                    column=column,
                    row=row,
                    forward_km=forward,
                    lateral_km=lateral,
                    wave_role="rear",
                    deployment_role="reserve_h",
                ))
        return tuple(result)

    def _grid_slots(
        self,
        band_low: float,
        band_high: float,
        lateral_length: float,
    ) -> tuple[tuple[int, int, float, float], ...]:
        spacing = self.config.lateral_spacing_km
        band_depth = band_high - band_low
        row_capacity = math.floor((lateral_length + _EPSILON_KM) / spacing) + 1
        column_capacity = math.floor((band_depth + _EPSILON_KM) / spacing) + 1
        used_axis_span = (column_capacity - 1) * spacing
        axis_start = (band_low + band_high - used_axis_span) / 2.0
        used_lateral_span = (row_capacity - 1) * spacing
        lateral_start = (lateral_length - used_lateral_span) / 2.0
        return tuple(
            (column, row, axis_start + column * spacing, lateral_start + row * spacing)
            for column in range(column_capacity)
            for row in range(row_capacity)
        )

    def _place_l_groups(
        self,
        bounds: DeploymentBounds,
        first_wave: tuple[PlatformDeploymentRequest, ...],
        second_wave: tuple[PlatformDeploymentRequest, ...],
        reserve_l: tuple[PlatformDeploymentRequest, ...],
        front_low: float,
        front_high: float,
        follow_low: float,
        follow_high: float,
        reserve_low: float,
        reserve_high: float,
        axis_length: float,
        lateral_length: float,
        occupied: set[tuple[float, float]],
    ) -> tuple[DeploymentPoint, ...]:
        all_slots = self._grid_slots(0.0, axis_length, lateral_length)
        available = {
            (slot[0], slot[1]): slot
            for slot in all_slots
            if self._coordinates_clear((slot,), occupied)
        }
        total_l = len(first_wave) + len(second_wave) + len(reserve_l)
        if total_l > len(available):
            raise ValueError(
                f"L deployment capacity {len(available)} is smaller than "
                f"{total_l} requests"
            )

        result: list[DeploymentPoint] = []
        spatial_order = tuple(sorted(
            first_wave + second_wave + reserve_l,
            key=lambda request: request.entity_id,
        ))
        target_lateral_by_id = dict(zip(
            (request.entity_id for request in spatial_order),
            self._uniform_lateral_positions(len(spatial_order), lateral_length),
            strict=True,
        ))
        for requests, band, wave_role, preferred_low, preferred_high in (
            (first_wave, "front", "front", front_low, front_high),
            (second_wave, "second", "second", follow_low, follow_high),
            (reserve_l, "second", "reserve_l", reserve_low, reserve_high),
        ):
            preferred_center = (preferred_low + preferred_high) / 2.0
            target_laterals = (
                tuple(
                    target_lateral_by_id[request.entity_id]
                    for request in requests
                )
                if reserve_l
                else self._uniform_lateral_positions(
                    len(requests), lateral_length
                )
            )
            for request, target_lateral in zip(
                requests, target_laterals, strict=True
            ):
                key = min(
                    available,
                    key=lambda item: (
                        not (
                            preferred_low - _EPSILON_KM
                            <= available[item][2]
                            <= preferred_high + _EPSILON_KM
                        ),
                        abs(available[item][3] - target_lateral),
                        abs(available[item][2] - preferred_center),
                        item[0],
                        item[1],
                    ),
                )
                slot = available.pop(key)
                result.append(
                    self._point_from_slot(
                        bounds, request, slot, band, wave_role
                    )
                )
        return tuple(result)

    def _point_from_slot(
        self,
        bounds: DeploymentBounds,
        request: PlatformDeploymentRequest,
        slot: tuple[int, int, float, float],
        band: str,
        wave_role: str,
        formation_role: str | None = None,
        desired_separation_km: float | None = None,
        actual_separation_km: float | None = None,
        requested_lead: float | None = None,
        actual_lead: float | None = None,
        predicted_m_arrival_delta_seconds: float | None = None,
        predicted_separation_at_m_20km: float | None = None,
        geometry_clamped: bool | None = None,
        clamp_reason: str | None = None,
    ) -> DeploymentPoint:
        column, row, forward, lateral = slot
        longitude, latitude = self._to_geographic(bounds, forward, lateral)
        return DeploymentPoint(
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
            wave_role=wave_role,
            target_id=request.target_id,
            target_type=request.target_type,
            formation_role=formation_role,
            desired_separation_km=desired_separation_km,
            actual_separation_km=actual_separation_km,
            requested_lead=requested_lead,
            actual_lead=actual_lead,
            predicted_m_arrival_delta_seconds=(
                predicted_m_arrival_delta_seconds
            ),
            predicted_separation_at_m_20km=(
                predicted_separation_at_m_20km
            ),
            geometry_clamped=geometry_clamped,
            clamp_reason=clamp_reason,
            deployment_role=request.deployment_role,
        )

    def _place_rear_groups(
        self,
        bounds: DeploymentBounds,
        h_requests: tuple[PlatformDeploymentRequest, ...],
        m_requests: tuple[PlatformDeploymentRequest, ...],
        h_low: float,
        h_high: float,
        follow_low: float,
        follow_high: float,
        axis_length: float,
        lateral_length: float,
        initially_blocked: set[tuple[float, float]],
        minimum_core_forward: float,
    ) -> tuple[tuple[DeploymentPoint, ...], tuple[DeploymentPoint, ...]]:
        h_slots = self._grid_slots(h_low, h_high, lateral_length)
        follow_slots = self._grid_slots(follow_low, follow_high, lateral_length)

        h_available = {
            (slot[0], slot[1]): slot
            for slot in h_slots
            if slot[2] + _EPSILON_KM >= minimum_core_forward
            and self._coordinates_clear((slot,), initially_blocked)
        }
        follow_available = {
            (slot[0], slot[1]): slot
            for slot in follow_slots
            if slot[2] + _EPSILON_KM >= minimum_core_forward
            and self._coordinates_clear((slot,), initially_blocked)
        }
        packages: dict[
            tuple[object, ...],
            dict[int, list[PlatformDeploymentRequest]],
        ] = {}
        for request in h_requests + m_requests:
            if request.target_lon is None or request.target_lat is None:
                continue
            target = (
                ("target", request.target_id)
                if request.target_id is not None
                else ("coordinate", request.target_lon, request.target_lat)
            )
            packages.setdefault(
                target, {H_PLATFORM_TYPE: [], M_PLATFORM_TYPE: []}
            )[request.entity_type].append(request)

        h_result: list[DeploymentPoint] = []
        m_result: list[DeploymentPoint] = []
        ordered_packages = sorted(
            packages.items(),
            key=lambda item: (
                self._target_lateral(
                    bounds,
                    item[1][H_PLATFORM_TYPE][0]
                    if item[1][H_PLATFORM_TYPE]
                    else item[1][M_PLATFORM_TYPE][0],
                    lateral_length,
                ),
                item[0],
            ),
        )
        crossfire_h_ids: set[int] = set()
        occupied_h_coordinates: set[tuple[float, float]] = set()
        for _target, members_by_type in ordered_packages:
            h_members = tuple(sorted(
                members_by_type[H_PLATFORM_TYPE],
                key=lambda item: item.entity_id,
            ))
            if (
                len(h_members) not in {2, 3}
                or h_members[0].target_type != 9400
            ):
                continue
            crossfire_points, reserved_keys = self._place_mother_crossfire(
                bounds,
                h_members,
                h_available,
                h_low,
                h_high,
                lateral_length,
                occupied_h_coordinates,
            )
            for key in reserved_keys:
                h_available.pop(key)
            h_result.extend(crossfire_points)
            crossfire_h_ids.update(item.entity_id for item in h_members)
            occupied_h_coordinates.update(
                (item.forward_km, item.lateral_km) for item in crossfire_points
            )

        blocked_coordinates = set(initially_blocked)
        blocked_coordinates.update(occupied_h_coordinates)
        blocked_coordinates.update(
            (slot[2], slot[3])
            for slot in h_slots
            if (slot[0], slot[1]) not in h_available
        )
        used_corridor_rows: set[int] = set()
        row_count = max((slot[1] for slot in h_slots), default=-1) + 1
        row_laterals = {
            slot[1]: slot[3]
            for slot in h_slots
            if slot[0] == h_slots[0][0]
        } if h_slots else {}
        corridor_required = any(
            members_by_type[M_PLATFORM_TYPE]
            or any(
                item.entity_id not in crossfire_h_ids
                for item in members_by_type[H_PLATFORM_TYPE]
            )
            for members_by_type in packages.values()
        )
        if corridor_required and row_count < 3:
            raise ValueError("target corridor requires at least three lateral grid rows")

        for target, members_by_type in ordered_packages:
            h_members = tuple(
                sorted(
                    (
                        item for item in members_by_type[H_PLATFORM_TYPE]
                        if item.entity_id not in crossfire_h_ids
                    ),
                    key=lambda item: item.entity_id,
                )
            )
            m_members = tuple(
                sorted(
                    members_by_type[M_PLATFORM_TYPE],
                    key=lambda item: item.entity_id,
                )
            )
            if not h_members and not m_members:
                continue
            sample = h_members[0] if h_members else m_members[0]
            sentry_formation = (
                sample.target_type == 9600
                and bool(h_members)
                and bool(m_members)
            )
            target_lateral = self._target_lateral(bounds, sample, lateral_length)
            corridor_rows = sorted(
                (row for row in range(1, row_count - 1)
                 if row not in used_corridor_rows),
                key=lambda row: (
                    abs(row_laterals[row] - target_lateral),
                    row,
                ),
            )
            adaptive_sentry = (
                sentry_formation
                and self.config.sentry_rendezvous_mode == "adaptive"
            )
            if adaptive_sentry:
                adaptive = self._adaptive_sentry_placement(
                    bounds,
                    h_members,
                    m_members,
                    corridor_rows,
                    row_laterals,
                    h_high,
                    axis_length,
                    lateral_length,
                    blocked_coordinates,
                    minimum_core_forward,
                )
                if adaptive is None:
                    raise ValueError(
                        f"target corridor capacity cannot place target {target!r}"
                    )
                corridor_row, h_selected_slots, m_selected_slots, diagnostics = adaptive
                used_corridor_rows.add(corridor_row)
                (
                    requested_lead,
                    actual_lead,
                    arrival_delta,
                    separation_at_20,
                    geometry_clamped,
                    placement_requested,
                    clamp_reason,
                ) = diagnostics
                for request, slot in zip(h_members, h_selected_slots, strict=True):
                    h_result.append(self._point_from_slot(
                        bounds,
                        request,
                        slot,
                        "rear",
                        "h_lead",
                        "h_follow",
                        placement_requested,
                        actual_lead,
                        requested_lead,
                        actual_lead,
                        arrival_delta,
                        separation_at_20,
                        geometry_clamped,
                        clamp_reason,
                    ))
                for request, slot in zip(m_members, m_selected_slots, strict=True):
                    m_result.append(self._point_from_slot(
                        bounds,
                        request,
                        slot,
                        "rear",
                        "m_follow",
                        "m_lead",
                        placement_requested,
                        actual_lead,
                        requested_lead,
                        actual_lead,
                        arrival_delta,
                        separation_at_20,
                        geometry_clamped,
                        clamp_reason,
                    ))
                placed_coordinates = {
                    (slot[2], slot[3])
                    for slot in h_selected_slots + m_selected_slots
                }
                blocked_coordinates.update(placed_coordinates)
                self._remove_coordinates_from_regions(
                    placed_coordinates, h_available, follow_available
                )
                continue

            selected: tuple[
                int,
                tuple[tuple[int, int], ...],
                tuple[tuple[int, int], ...],
            ] | None = None
            if sentry_formation:
                lead_center, follow_center = self._sentry_formation_centers(
                    h_low,
                    h_high,
                    follow_low,
                    follow_high,
                )
            else:
                lead_center = (h_low + h_high) / 2.0
                follow_center = (follow_low + follow_high) / 2.0
            for corridor_row in corridor_rows:
                if sentry_formation:
                    h_keys = self._package_slot_keys(
                        len(h_members), follow_slots, corridor_row,
                        follow_center, False,
                    )
                    m_keys = self._package_slot_keys(
                        len(m_members), h_slots, corridor_row,
                        lead_center, True,
                    )
                    h_region = follow_available
                    m_region = h_available
                else:
                    h_keys = self._package_slot_keys(
                        len(h_members), h_slots, corridor_row,
                        lead_center, True,
                    )
                    m_keys = self._package_slot_keys(
                        len(m_members), follow_slots, corridor_row,
                        follow_center, False,
                    )
                    h_region = h_available
                    m_region = follow_available
                if all(key in h_region for key in h_keys) and all(
                    key in m_region for key in m_keys
                ):
                    selected = corridor_row, h_keys, m_keys
                    break
            if selected is None:
                raise ValueError(
                    f"target corridor capacity cannot place target {target!r}"
                )
            corridor_row, h_keys, m_keys = selected
            used_corridor_rows.add(corridor_row)
            if sentry_formation:
                h_region = follow_available
                m_region = h_available
                h_selected_slots = tuple(h_region[key] for key in h_keys)
                m_selected_slots = tuple(m_region[key] for key in m_keys)
                actual_separation = (
                    sum(slot[2] for slot in m_selected_slots) / len(m_selected_slots)
                    - sum(slot[2] for slot in h_selected_slots) / len(h_selected_slots)
                )
                if not 0.0 <= actual_separation <= 15.0 + _EPSILON_KM:
                    raise ValueError(
                        f"same-target sentry H/M formation for {target!r} "
                        "cannot satisfy forward separation limits"
                    )
                h_forward = sum(
                    slot[2] for slot in h_selected_slots
                ) / len(h_selected_slots)
                h_lateral = sum(
                    slot[3] for slot in h_selected_slots
                ) / len(h_selected_slots)
                h_longitude, h_latitude = self._to_geographic(
                    bounds, h_forward, h_lateral
                )
                distance_to_target = self._surface_distance_km(
                    h_longitude,
                    h_latitude,
                    sample.target_lon,
                    sample.target_lat,
                )
                diagnostics = self._sentry_diagnostics(
                    distance_to_target,
                    self.config.sentry_m_lead_km,
                    self.config.sentry_m_lead_km,
                    actual_separation,
                )
            else:
                h_region = h_available
                m_region = follow_available
                actual_separation = None
                diagnostics = (None, None, None, None, None, None, None)
            (
                requested_lead,
                actual_lead,
                arrival_delta,
                separation_at_20,
                geometry_clamped,
                placement_requested,
                clamp_reason,
            ) = diagnostics
            placed_coordinates: set[tuple[float, float]] = set()
            for request, key in zip(h_members, h_keys):
                slot = h_region.pop(key)
                placed_coordinates.add((slot[2], slot[3]))
                h_result.append(self._point_from_slot(
                    bounds,
                    request,
                    slot,
                    "rear",
                    "h_lead",
                    "h_follow" if sentry_formation else None,
                    placement_requested if sentry_formation else None,
                    actual_separation,
                    requested_lead,
                    actual_lead,
                    arrival_delta,
                    separation_at_20,
                    geometry_clamped,
                    clamp_reason,
                ))
            for request, key in zip(m_members, m_keys):
                slot = m_region.pop(key)
                placed_coordinates.add((slot[2], slot[3]))
                m_result.append(self._point_from_slot(
                    bounds,
                    request,
                    slot,
                    "rear",
                    "m_follow",
                    "m_lead" if sentry_formation else None,
                    placement_requested if sentry_formation else None,
                    actual_separation,
                    requested_lead,
                    actual_lead,
                    arrival_delta,
                    separation_at_20,
                    geometry_clamped,
                    clamp_reason,
                ))
            blocked_coordinates.update(placed_coordinates)
            self._remove_coordinates_from_regions(
                placed_coordinates, h_available, follow_available
            )

        untargeted_h = tuple(
            request for request in h_requests if request.target_lon is None
        )
        untargeted_m = tuple(
            request for request in m_requests if request.target_lon is None
        )
        for request, slot in zip(
            untargeted_h,
            self._take_uniform_slots(
                len(untargeted_h), h_available, lateral_length,
                (h_low + h_high) / 2.0, "h_lead",
            ),
        ):
            h_result.append(
                self._point_from_slot(bounds, request, slot, "rear", "h_lead")
            )
        for request, slot in zip(
            untargeted_m,
            self._take_uniform_slots(
                len(untargeted_m), follow_available, lateral_length,
                (follow_low + follow_high) / 2.0, "m_follow",
            ),
        ):
            m_result.append(
                self._point_from_slot(bounds, request, slot, "rear", "m_follow")
            )

        self._validate_package_centroids(packages, h_result, m_result)

        return tuple(h_result), tuple(m_result)

    def _sentry_formation_centers(
        self,
        lead_low: float,
        lead_high: float,
        follow_low: float,
        follow_high: float,
    ) -> tuple[float, float]:
        desired = self.config.sentry_m_lead_km
        # A positive grid step is the closest non-overlapping placement for a
        # zero or sub-grid request when both groups use the same corridor rows.
        minimum_separation = max(0.0, lead_low - follow_high)
        maximum_separation = lead_high - follow_low
        placement_separation = min(
            maximum_separation,
            max(
                desired,
                minimum_separation,
                min(self.config.lateral_spacing_km, maximum_separation),
            ),
        )
        minimum_lead = max(lead_low, follow_low + placement_separation)
        maximum_lead = min(lead_high, follow_high + placement_separation)
        nominal_lead = (
            (lead_low + lead_high) / 2.0
            + (follow_low + follow_high) / 2.0
            + placement_separation
        ) / 2.0
        lead_center = min(maximum_lead, max(minimum_lead, nominal_lead))
        return lead_center, lead_center - placement_separation

    def _adaptive_sentry_placement(
        self,
        bounds: DeploymentBounds,
        h_members: tuple[PlatformDeploymentRequest, ...],
        m_members: tuple[PlatformDeploymentRequest, ...],
        corridor_rows: list[int],
        row_laterals: dict[int, float],
        h_high: float,
        axis_length: float,
        lateral_length: float,
        blocked: set[tuple[float, float]],
        minimum_core_forward: float,
    ) -> tuple[
        int,
        tuple[tuple[int, int, float, float], ...],
        tuple[tuple[int, int, float, float], ...],
        tuple[float, float, float, float, bool, float, str | None],
    ] | None:
        h_offsets = self._formation_offsets(len(h_members))
        m_offsets = self._formation_offsets(len(m_members))
        h_min_center = max(
            -min(item[2] for item in h_offsets),
            minimum_core_forward - min(item[2] for item in h_offsets),
        )
        h_max_center = h_high - max(item[2] for item in h_offsets)
        if h_min_center > h_max_center + _EPSILON_KM:
            return None

        h_centers = self._interval_candidates(
            h_min_center,
            h_max_center,
            h_min_center,
        )
        for corridor_row in corridor_rows:
            corridor_lateral = row_laterals[corridor_row]
            for h_center in h_centers:
                h_slots = self._formation_slots(
                    h_offsets,
                    h_center,
                    corridor_lateral,
                    corridor_row,
                    h_high,
                    lateral_length,
                )
                if h_slots is None or not self._coordinates_clear(h_slots, blocked):
                    continue

                h_longitude, h_latitude = self._to_geographic(
                    bounds, h_center, corridor_lateral
                )
                distance_to_target = self._surface_distance_km(
                    h_longitude,
                    h_latitude,
                    h_members[0].target_lon,
                    h_members[0].target_lat,
                )
                raw_requested = (
                    (1.0 - self.config.m_speed_mps / self.config.h_speed_mps)
                    * distance_to_target
                    + self.config.m_speed_mps
                    / 1000.0
                    * self.config.desired_m_arrival_lead_seconds
                )
                placement_requested = max(
                    raw_requested, self.config.min_sentry_m_lead_km
                )
                if self.config.max_sentry_m_lead_km is not None:
                    placement_requested = min(
                        placement_requested, self.config.max_sentry_m_lead_km
                    )

                legal_low = max(
                    -min(item[2] for item in m_offsets), h_center
                )
                legal_high = axis_length - max(item[2] for item in m_offsets)
                if legal_low > legal_high + _EPSILON_KM:
                    continue
                policy_low = max(
                    legal_low,
                    h_center + self.config.min_sentry_m_lead_km,
                )
                policy_high = legal_high
                if self.config.max_sentry_m_lead_km is not None:
                    policy_high = min(
                        policy_high,
                        h_center + self.config.max_sentry_m_lead_km,
                    )
                if policy_low <= policy_high + _EPSILON_KM:
                    m_low, m_high = policy_low, policy_high
                else:
                    # A small legal rectangle can make the configured minimum
                    # impossible. Retain the farthest legal forward placement.
                    m_low, m_high = legal_low, legal_high
                desired_m_center = min(
                    m_high,
                    max(m_low, h_center + placement_requested),
                )
                boundary_clamped = not math.isclose(
                    desired_m_center,
                    h_center + placement_requested,
                    rel_tol=0.0,
                    abs_tol=_EPSILON_KM,
                )
                locally_blocked = blocked | {
                    (slot[2], slot[3]) for slot in h_slots
                }
                for m_center in self._interval_candidates(
                    m_low, m_high, desired_m_center
                ):
                    m_slots = self._formation_slots(
                        m_offsets,
                        m_center,
                        corridor_lateral,
                        corridor_row,
                        axis_length,
                        lateral_length,
                    )
                    if m_slots is None or not self._coordinates_clear(
                        m_slots, locally_blocked
                    ):
                        continue
                    actual_lead = m_center - h_center
                    diagnostics = self._sentry_diagnostics(
                        distance_to_target,
                        raw_requested,
                        placement_requested,
                        actual_lead,
                        "boundary"
                        if boundary_clamped
                        else "capacity"
                        if not math.isclose(
                            m_center,
                            desired_m_center,
                            rel_tol=0.0,
                            abs_tol=_EPSILON_KM,
                        )
                        else None,
                    )
                    return (
                        corridor_row,
                        h_slots,
                        m_slots,
                        diagnostics,
                    )
        return None

    def _formation_offsets(
        self, count: int
    ) -> tuple[tuple[int, int, float, float], ...]:
        """Return a compact, centroid-zero formation with three lateral lanes."""
        if count <= 0:
            return ()
        column_count = math.ceil(count / 3)
        base, extra = divmod(count, column_count)
        raw: list[tuple[int, int, float, float]] = []
        for column in range(column_count):
            column_size = base + (column < extra)
            lanes = {
                1: (0,),
                2: (-1, 1),
                3: (-1, 0, 1),
            }[column_size]
            raw.extend(
                (column, lane, float(column), float(lane)) for lane in lanes
            )
        mean_forward = sum(item[2] for item in raw) / count
        mean_lateral = sum(item[3] for item in raw) / count
        spacing = self.config.lateral_spacing_km
        return tuple(
            (
                column,
                lane,
                (forward - mean_forward) * spacing,
                (lateral - mean_lateral) * spacing,
            )
            for column, lane, forward, lateral in raw
        )

    @staticmethod
    def _formation_slots(
        offsets: tuple[tuple[int, int, float, float], ...],
        center_forward: float,
        center_lateral: float,
        corridor_row: int,
        forward_high: float,
        lateral_high: float,
    ) -> tuple[tuple[int, int, float, float], ...] | None:
        slots = tuple(
            (
                max(0, int(round(center_forward + forward_offset))),
                max(0, corridor_row + lane),
                center_forward + forward_offset,
                center_lateral + lateral_offset,
            )
            for _column, lane, forward_offset, lateral_offset in offsets
        )
        if any(
            slot[2] < -_EPSILON_KM
            or slot[2] > forward_high + _EPSILON_KM
            or slot[3] < -_EPSILON_KM
            or slot[3] > lateral_high + _EPSILON_KM
            for slot in slots
        ):
            return None
        return slots

    def _interval_candidates(
        self, low: float, high: float, preferred: float
    ) -> tuple[float, ...]:
        preferred = min(high, max(low, preferred))
        values = {preferred, low, high}
        spacing = self.config.lateral_spacing_km
        step_count = math.floor((high - low + _EPSILON_KM) / spacing)
        values.update(low + index * spacing for index in range(step_count + 1))
        return tuple(
            sorted(
                values,
                key=lambda value: (abs(value - preferred), -value),
            )
        )

    @staticmethod
    def _coordinates_clear(
        slots: tuple[tuple[int, int, float, float], ...],
        blocked: set[tuple[float, float]],
    ) -> bool:
        coordinates = tuple((slot[2], slot[3]) for slot in slots)
        if len(set(coordinates)) != len(coordinates):
            return False
        return not any(
            math.hypot(forward - occupied_forward, lateral - occupied_lateral)
            <= _EPSILON_KM
            for forward, lateral in coordinates
            for occupied_forward, occupied_lateral in blocked
        )

    @staticmethod
    def _remove_coordinates_from_regions(
        coordinates: set[tuple[float, float]],
        *regions: dict[tuple[int, int], tuple[int, int, float, float]],
    ) -> None:
        for region in regions:
            for key, slot in tuple(region.items()):
                if any(
                    math.hypot(
                        slot[2] - occupied_forward,
                        slot[3] - occupied_lateral,
                    ) <= _EPSILON_KM
                    for occupied_forward, occupied_lateral in coordinates
                ):
                    region.pop(key)

    def _sentry_diagnostics(
        self,
        distance_to_target: float,
        raw_requested: float,
        placement_requested: float,
        actual_lead: float,
        clamp_reason: str | None = None,
    ) -> tuple[float, float, float, float, bool, float, str | None]:
        h_speed_kmps = self.config.h_speed_mps / 1000.0
        m_speed_kmps = self.config.m_speed_mps / 1000.0
        arrival_delta = (
            distance_to_target / h_speed_kmps
            - (distance_to_target - actual_lead) / m_speed_kmps
        )
        separation_at_20 = (
            distance_to_target
            - self.config.h_speed_mps
            / self.config.m_speed_mps
            * (distance_to_target - actual_lead - 20.0)
            - 20.0
        )
        geometry_clamped = not math.isclose(
            actual_lead,
            placement_requested,
            rel_tol=0.0,
            abs_tol=_EPSILON_KM,
        )
        if geometry_clamped and clamp_reason is None:
            clamp_reason = "capacity"
        if not geometry_clamped:
            clamp_reason = None
        return (
            raw_requested,
            actual_lead,
            arrival_delta,
            separation_at_20,
            geometry_clamped,
            placement_requested,
            clamp_reason,
        )

    @staticmethod
    def _surface_distance_km(
        first_lon: float,
        first_lat: float,
        second_lon: float | None,
        second_lat: float | None,
    ) -> float:
        assert second_lon is not None and second_lat is not None
        first_lat_radians = math.radians(first_lat)
        second_lat_radians = math.radians(second_lat)
        latitude_delta = second_lat_radians - first_lat_radians
        longitude_delta = math.radians(second_lon - first_lon)
        haversine = (
            math.sin(latitude_delta / 2.0) ** 2
            + math.cos(first_lat_radians)
            * math.cos(second_lat_radians)
            * math.sin(longitude_delta / 2.0) ** 2
        )
        return 2.0 * _EARTH_MEAN_RADIUS_KM * math.asin(
            min(1.0, math.sqrt(haversine))
        )

    def _place_mother_crossfire(
        self,
        bounds: DeploymentBounds,
        requests: tuple[PlatformDeploymentRequest, ...],
        available: dict[tuple[int, int], tuple[int, int, float, float]],
        band_low: float,
        band_high: float,
        lateral_length: float,
        occupied: set[tuple[float, float]],
    ) -> tuple[tuple[DeploymentPoint, ...], tuple[tuple[int, int], ...]]:
        target_forward, target_lateral = self._target_local(bounds, requests[0])
        clamped_lateral = min(lateral_length, max(0.0, target_lateral))
        desired_angle = self.config.mother_crossfire_angle_degrees
        tangent = math.tan(math.radians(desired_angle / 2.0))
        band_center = (band_low + band_high) / 2.0
        columns = sorted({slot[0]: slot[2] for slot in available.values()}.items())
        roles = ("left", "right") if len(requests) == 2 else (
            "left", "center", "right"
        )
        best: tuple[
            tuple[float, ...],
            tuple[float, ...],
            tuple[tuple[int, int], ...],
            float,
            int,
            float,
        ] | None = None

        for column, forward in columns:
            column_slots = [
                (key, slot) for key, slot in available.items()
                if slot[0] == column
            ]
            if len(column_slots) < len(requests):
                continue
            forward_distance = abs(target_forward - forward)
            desired_offset = forward_distance * tangent
            symmetric_limit = min(
                self.config.mother_crossfire_max_offset_km,
                clamped_lateral,
                lateral_length - clamped_lateral,
            )
            if symmetric_limit > _EPSILON_KM:
                center = clamped_lateral
                offset = min(desired_offset, symmetric_limit)
                shifted_from_edge = False
            else:
                max_offset = min(
                    self.config.mother_crossfire_max_offset_km,
                    lateral_length / 2.0,
                )
                offset = min(desired_offset, max_offset)
                if offset <= _EPSILON_KM:
                    offset = min(
                        max_offset,
                        self.config.lateral_spacing_km / 2.0,
                    )
                center = (
                    offset
                    if target_lateral <= lateral_length / 2.0
                    else lateral_length - offset
                )
                shifted_from_edge = True
            if offset <= _EPSILON_KM:
                continue

            lateral = (
                (center - offset, center + offset)
                if len(requests) == 2
                else (center - offset, center, center + offset)
            )
            achieved_angle = self._crossfire_angle(
                target_forward, target_lateral, forward, lateral[0], lateral[-1]
            )
            if shifted_from_edge and achieved_angle + _EPSILON_KM < 30.0:
                offset = min(
                    self.config.mother_crossfire_max_offset_km,
                    lateral_length / 2.0,
                )
                center = (
                    offset
                    if target_lateral <= lateral_length / 2.0
                    else lateral_length - offset
                )
                lateral = (
                    (center - offset, center + offset)
                    if len(requests) == 2
                    else (center - offset, center, center + offset)
                )
                achieved_angle = self._crossfire_angle(
                    target_forward, target_lateral, forward, lateral[0], lateral[-1]
                )
            coordinates = tuple((forward, value) for value in lateral)
            if any(item in occupied for item in coordinates):
                continue

            remaining = list(column_slots)
            reserved: list[tuple[int, int]] = []
            for value in lateral:
                key, _slot = min(
                    remaining,
                    key=lambda item: (
                        abs(item[1][3] - value),
                        item[1][1],
                    ),
                )
                remaining.remove((key, _slot))
                reserved.append(key)
            score = (
                float(achieved_angle + _EPSILON_KM < 30.0),
                abs(desired_angle - achieved_angle)
                if achieved_angle + _EPSILON_KM >= 30.0
                else -achieved_angle,
                abs(forward - band_center),
                float(column),
            )
            candidate = (
                score,
                lateral,
                tuple(reserved),
                achieved_angle,
                column,
                forward,
            )
            if best is None or candidate[0] < best[0]:
                best = candidate

        if best is None:
            raise ValueError(
                f"mother crossfire cannot place target {requests[0].target_id}"
            )

        _score, lateral, reserved, achieved_angle, column, forward = best
        points = []
        for request, role, value, reservation in zip(
            requests, roles, lateral, reserved, strict=True
        ):
            longitude, latitude = self._to_geographic(bounds, forward, value)
            points.append(DeploymentPoint(
                entity_id=request.entity_id,
                entity_type=request.entity_type,
                longitude=longitude,
                latitude=latitude,
                altitude=bounds.altitude,
                band="rear",
                column=column,
                row=reservation[1],
                forward_km=forward,
                lateral_km=value,
                wave_role="h_lead",
                target_id=request.target_id,
                target_type=request.target_type,
                crossfire_role=role,
                crossfire_angle=achieved_angle,
                deployment_role=request.deployment_role,
            ))
        return tuple(points), reserved

    @staticmethod
    def _crossfire_angle(
        target_forward: float,
        target_lateral: float,
        deployment_forward: float,
        low_lateral: float,
        high_lateral: float,
    ) -> float:
        first = (
            deployment_forward - target_forward,
            low_lateral - target_lateral,
        )
        second = (
            deployment_forward - target_forward,
            high_lateral - target_lateral,
        )
        first_length = math.hypot(*first)
        second_length = math.hypot(*second)
        if first_length <= _EPSILON_KM or second_length <= _EPSILON_KM:
            return 0.0
        cross = abs(first[0] * second[1] - first[1] * second[0])
        dot = first[0] * second[0] + first[1] * second[1]
        return math.degrees(math.atan2(cross, dot))

    def _package_slot_keys(
        self,
        count: int,
        slots: tuple[tuple[int, int, float, float], ...],
        corridor_row: int,
        band_center: float,
        prefer_front: bool,
    ) -> tuple[tuple[int, int], ...]:
        if count == 0:
            return ()
        columns = sorted({slot[0]: slot[2] for slot in slots}.items())
        minimum_columns = 2 if count > 1 else 1
        column_count = min(
            len(columns), max(minimum_columns, math.ceil(count / 3))
        )
        if count > column_count * 3:
            raise ValueError(
                f"target package capacity {column_count * 3} is smaller than "
                f"{count} requests"
            )

        base, extra = divmod(count, column_count)
        best: tuple[tuple[float, ...], tuple[tuple[int, int], ...]] | None = None
        for selected_columns in combinations(columns, column_count):
            for extra_indexes in combinations(range(column_count), extra):
                extras = set(extra_indexes)
                weighted_sum = sum(
                    forward * (base + (index in extras))
                    for index, (_, forward) in enumerate(selected_columns)
                )
                key = (
                    abs(weighted_sum / count - band_center),
                    -weighted_sum / count if prefer_front else weighted_sum / count,
                    max(abs(forward - band_center) for _, forward in selected_columns),
                    selected_columns[-1][1] - selected_columns[0][1],
                    *(forward for _, forward in selected_columns),
                )
                allocation = tuple(
                    (column, base + (index in extras))
                    for index, (column, _) in enumerate(selected_columns)
                )
                if best is None or key < best[0]:
                    best = key, allocation
        assert best is not None

        quotient, remainder = divmod(count, 3)
        lane_remaining = {
            -1: quotient + (remainder == 2),
            0: quotient + (remainder == 1),
            1: quotient + (remainder == 2),
        }
        result: list[tuple[int, int]] = []
        for column, column_size in best[1]:
            used_offsets: set[int] = set()
            for _ in range(column_size):
                offset = min(
                    (
                        item for item in (-1, 1, 0)
                        if lane_remaining[item] and item not in used_offsets
                    ),
                    key=lambda item: (-lane_remaining[item], (-1, 1, 0).index(item)),
                )
                lane_remaining[offset] -= 1
                used_offsets.add(offset)
                result.append((column, corridor_row + offset))
        return tuple(result)

    def _take_uniform_slots(
        self,
        count: int,
        available: dict[tuple[int, int], tuple[int, int, float, float]],
        lateral_length: float,
        band_center: float,
        region_name: str,
    ) -> tuple[tuple[int, int, float, float], ...]:
        if count > len(available):
            raise ValueError(
                f"{region_name} region capacity {len(available)} is smaller than "
                f"{count} requests"
            )
        result: list[tuple[int, int, float, float]] = []
        for target_lateral in self._uniform_lateral_positions(count, lateral_length):
            key = min(
                available,
                key=lambda item: (
                    abs(available[item][3] - target_lateral),
                    abs(available[item][2] - band_center),
                    item[0],
                    item[1],
                ),
            )
            result.append(available.pop(key))
        return tuple(result)

    def _validate_package_centroids(
        self,
        packages: dict[
            tuple[object, ...],
            dict[int, list[PlatformDeploymentRequest]],
        ],
        h_points: list[DeploymentPoint],
        m_points: list[DeploymentPoint],
    ) -> None:
        h_by_id = {point.entity_id: point for point in h_points}
        m_by_id = {point.entity_id: point for point in m_points}
        for target, members_by_type in packages.items():
            sample = (
                members_by_type[H_PLATFORM_TYPE]
                or members_by_type[M_PLATFORM_TYPE]
            )[0]
            if sample.target_type == 9400:
                continue
            if (
                not members_by_type[H_PLATFORM_TYPE]
                or not members_by_type[M_PLATFORM_TYPE]
            ):
                continue
            target_h = [
                h_by_id[item.entity_id]
                for item in members_by_type[H_PLATFORM_TYPE]
            ]
            target_m = [
                m_by_id[item.entity_id]
                for item in members_by_type[M_PLATFORM_TYPE]
            ]
            h_forward = sum(item.forward_km for item in target_h) / len(target_h)
            h_lateral = sum(item.lateral_km for item in target_h) / len(target_h)
            m_forward = sum(item.forward_km for item in target_m) / len(target_m)
            m_lateral = sum(item.lateral_km for item in target_m) / len(target_m)
            lateral_gap = abs(h_lateral - m_lateral)
            distance = math.hypot(h_forward - m_forward, lateral_gap)
            wrong_sentry_order = (
                sample.target_type == 9600
                and m_forward + _EPSILON_KM < h_forward
            )
            adaptive_sentry = (
                sample.target_type == 9600
                and self.config.sentry_rendezvous_mode == "adaptive"
            )
            if (
                wrong_sentry_order
                or lateral_gap > 3.0 + _EPSILON_KM
                or (
                    not adaptive_sentry
                    and distance > 15.0 + _EPSILON_KM
                )
            ):
                raise ValueError(
                    f"same-target H/M centroid link for {target!r} exceeds limits"
                )

    def _target_lateral(
        self,
        bounds: DeploymentBounds,
        request: PlatformDeploymentRequest,
        lateral_length: float,
    ) -> float:
        _forward, lateral = self._target_local(bounds, request)
        return min(lateral_length, max(0.0, lateral))

    def _target_local(
        self,
        bounds: DeploymentBounds,
        request: PlatformDeploymentRequest,
    ) -> tuple[float, float]:
        if request.target_lon is None or request.target_lat is None:
            raise ValueError("target coordinates are required for target-aware placement")
        east = (request.target_lon - bounds.lon_min) * bounds.km_per_longitude_degree
        north = (request.target_lat - bounds.lat_min) * _KM_PER_LATITUDE_DEGREE
        if self.config.front_axis == "east":
            return east, north
        if self.config.front_axis == "west":
            return bounds.width_km - east, north
        if self.config.front_axis == "north":
            return north, east
        return bounds.height_km - north, east

    @staticmethod
    def _uniform_lateral_positions(count: int, lateral_length: float) -> tuple[float, ...]:
        if count <= 0:
            return ()
        if count == 1:
            return (lateral_length / 2.0,)
        step = lateral_length / (count - 1)
        return tuple(index * step for index in range(count))

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
    "DEPLOYMENT_ROLES",
    "H_PLATFORM_TYPE",
    "L_PLATFORM_TYPE",
    "M_PLATFORM_TYPE",
    "PlatformDeploymentRequest",
    "PUBLIC_TARGET_TYPES",
    "SUPPORTED_PLATFORM_TYPES",
    "StaggeredDeploymentConfig",
    "StaggeredDeploymentSkill",
]
