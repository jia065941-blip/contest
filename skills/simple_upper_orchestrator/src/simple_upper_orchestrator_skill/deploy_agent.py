"""Official deployment-agent adapter for the pure staggered planner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import numpy as np

from reserve_echelon_skill import ReservePackageSnapshot

from hm_public_strike_packages_skill import (
    HMMissileSnapshot,
    HMPublicStrikePackagesSkill,
)

from staggered_deployment_skill import (
    DeploymentBounds,
    PlatformDeploymentRequest,
    StaggeredDeploymentSkill,
    SUPPORTED_PLATFORM_TYPES,
    to_deployment_actions,
)
from user_agents.base_agent import AgentType, BaseAgent


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class PreMissionPlan:
    """Immutable information legally available during fleet deployment."""

    hm_assignments: Mapping[int, int]
    entity_types: Mapping[int, int]
    entity_core: Mapping[int, Mapping]
    l_roster: tuple[int, ...]
    reserve_packages: tuple[ReservePackageSnapshot, ...] = ()
    reserve_target_regions: Mapping[int, str] = field(default_factory=dict)
    l_roles: Mapping[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assignments = {
            int(owner): int(target)
            for owner, target in sorted(self.hm_assignments.items())
        }
        entity_types = {
            int(entity_id): int(entity_type)
            for entity_id, entity_type in sorted(self.entity_types.items())
        }
        entity_core = {
            int(entity_id): _freeze(value)
            for entity_id, value in sorted(self.entity_core.items())
        }
        l_roster = tuple(int(entity_id) for entity_id in self.l_roster)
        l_roles = {
            int(entity_id): str(role).lower()
            for entity_id, role in sorted(self.l_roles.items())
        }
        reserve_packages = tuple(
            sorted(self.reserve_packages, key=lambda package: package.package_id)
        )
        reserve_target_regions = {
            int(target_id): str(region)
            for target_id, region in sorted(self.reserve_target_regions.items())
        }
        if len(l_roster) != len(set(l_roster)):
            raise ValueError("l_roster must contain unique entity IDs")
        if set(entity_core) != set(entity_types):
            raise ValueError("entity_core and entity_types must have identical IDs")
        if any(entity_type not in SUPPORTED_PLATFORM_TYPES for entity_type in entity_types.values()):
            raise ValueError("mission plan contains an unsupported platform type")
        if any(entity_types.get(owner) not in (21000, 21001) for owner in assignments):
            raise ValueError("H/M assignments may contain only 21000/21001 owners")
        expected_l = tuple(
            entity_id
            for entity_id, entity_type in entity_types.items()
            if entity_type == 21002
        )
        if set(l_roster) != set(expected_l) or len(l_roster) != len(expected_l):
            raise ValueError("l_roster must contain every 21002 entity exactly once")
        if l_roles and set(l_roles) != set(l_roster):
            raise ValueError("l_roles must contain every L roster entity exactly once")
        if any(role not in {"core", "reserve_l"} for role in l_roles.values()):
            raise ValueError("L roles must be core or reserve_l")
        if any(
            not isinstance(package, ReservePackageSnapshot)
            for package in reserve_packages
        ):
            raise ValueError("reserve_packages must contain ReservePackageSnapshot values")
        reserve_ids = tuple(
            member_id
            for package in reserve_packages
            for member_id in package.member_ids
        )
        if len(reserve_ids) != len(set(reserve_ids)):
            raise ValueError("a reserve member may belong to only one package")
        for package in reserve_packages:
            if package.state != "STAGING" or package.dispatch_count != 0:
                raise ValueError("pre-mission reserve packages must be STAGING")
            member_types = {entity_types.get(member_id) for member_id in package.member_ids}
            if member_types not in ({21000}, {21002}):
                raise ValueError("pre-mission reserve packages must contain homogeneous H or L members")
            if any(member_id in assignments for member_id in package.member_ids):
                raise ValueError("reserve members must not have core H/M assignments")
            if member_types == {21000}:
                if any(
                    reserve_target_regions.get(target_id)
                    != package.responsibility_region
                    for target_id in (
                        package.provisional_target_id,
                        package.fallback_target_id,
                    )
                ):
                    raise ValueError("reserve H package targets must match its region")
            elif package.responsibility_region != "l" or any(
                l_roles.get(member_id) != "reserve_l"
                for member_id in package.member_ids
            ):
                raise ValueError("reserve L packages require reserve_l roster members")
        object.__setattr__(self, "hm_assignments", MappingProxyType(assignments))
        object.__setattr__(self, "entity_types", MappingProxyType(entity_types))
        object.__setattr__(self, "entity_core", MappingProxyType(entity_core))
        object.__setattr__(self, "l_roster", l_roster)
        object.__setattr__(self, "l_roles", MappingProxyType(l_roles))
        object.__setattr__(self, "reserve_packages", reserve_packages)
        object.__setattr__(
            self,
            "reserve_target_regions",
            MappingProxyType(reserve_target_regions),
        )


def _even_chunks(values: tuple, count: int) -> tuple[tuple, ...]:
    quotient, remainder = divmod(len(values), count)
    chunks = []
    start = 0
    for index in range(count):
        size = quotient + (index < remainder)
        chunks.append(values[start:start + size])
        start += size
    return tuple(chunks)


def _reserve_h_packages(
    reserve_ids: tuple[int, ...], public_targets: tuple,
) -> tuple[tuple[ReservePackageSnapshot, ...], dict[int, str]]:
    """Build deterministic latitude responsibility regions for reserve H."""
    if not reserve_ids or len(public_targets) < 2:
        return (), {}
    region_count = min(3, len(reserve_ids), len(public_targets) // 2)
    labels = {
        1: ("center",),
        2: ("south", "north"),
        3: ("south", "center", "north"),
    }[region_count]
    target_chunks = _even_chunks(
        tuple(sorted(public_targets, key=lambda target: (target.lat, target.target_id))),
        region_count,
    )
    reserve_chunks = _even_chunks(tuple(sorted(reserve_ids)), region_count)
    packages = []
    target_regions = {}
    for label, targets, members in zip(labels, target_chunks, reserve_chunks):
        preferred = tuple(sorted(
            targets,
            key=lambda target: (target.target_type != 9400, target.target_id),
        ))
        provisional, fallback = preferred[:2]
        for target in targets:
            target_regions[target.target_id] = label
        packages.append(ReservePackageSnapshot(
            package_id=members[0],
            member_ids=members,
            member_types=(21000,) * len(members),
            state="STAGING",
            provisional_target_id=provisional.target_id,
            fallback_target_id=fallback.target_id,
            responsibility_region=label,
            dispatch_count=0,
        ))
    return tuple(packages), target_regions


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
    def __init__(
        self, coordinates, coordinates_hm, *, config, batch_size: int,
        launch_altitude, on_plan=None, on_mission_plan=None,
        public_targets=(), hm_config=None, l_first_wave_per_target=0,
        expected_l_target_count=None,
    ):
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
        self._on_plan = on_plan
        self._on_mission_plan = on_mission_plan
        self._public_targets = tuple(public_targets)
        self._hm_config = hm_config
        self._l_first_wave_per_target = int(l_first_wave_per_target)
        self._expected_l_target_count = (
            None
            if expected_l_target_count is None
            else int(expected_l_target_count)
        )
        if self._l_first_wave_per_target < 0:
            raise ValueError("l_first_wave_per_target must be nonnegative")
        if (
            self._l_first_wave_per_target > 0
            and (
                self._expected_l_target_count is None
                or self._expected_l_target_count <= 0
            )
        ):
            raise ValueError(
                "positive l_first_wave_per_target requires expected L target count"
            )
        self._mission_plan: PreMissionPlan | None = None

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
        if not selected:
            raise ValueError("deployment observation contains no H/M/L platforms")
        entity_types = {
            entity_id: int(entity["type"])
            for entity_id, entity in selected
        }
        entity_l_ids = tuple(
            entity_id
            for entity_id, entity in selected
            if int(entity["type"]) == 21002
        )
        altitude = self._altitude
        if altitude is None:
            values = {float(entity["position"]["alt"]) for _, entity in selected}
            if len(values) != 1:
                raise ValueError(
                    "platform launch altitudes differ and no launch altitude was supplied"
                )
            altitude = values.pop()
        deployment_bounds = DeploymentBounds(
            *self._raw_bounds, float(altitude)
        )
        if self._mission_plan is None:
            hm_members = tuple(
                HMMissileSnapshot(entity_id, int(entity["type"]))
                for entity_id, entity in selected
                if int(entity["type"]) in (21000, 21001)
            )
            target_by_member = {}
            if hm_members and self._public_targets:
                planner = HMPublicStrikePackagesSkill(
                    self._hm_config, public_targets=self._public_targets,
                )
                decision = planner.decide(step=0, members=hm_members)
                target_by_member = {
                    member_id: package.target.target_id
                    for package in decision.packages
                    for member_id in package.member_ids
                }
                reserve_h_ids = tuple(
                    member_id
                    for member_id in decision.unassigned_member_ids
                    if entity_types[member_id] == 21000
                )
                (
                    reserve_packages,
                    reserve_target_regions,
                ) = _reserve_h_packages(reserve_h_ids, self._public_targets)
            else:
                reserve_packages = ()
                reserve_target_regions = {}
            target_by_id = {
                target.target_id: target for target in self._public_targets
            }
            reserve_target_by_member = {
                member_id: target_by_id[package.provisional_target_id]
                for package in reserve_packages
                for member_id in package.member_ids
            }
            provisional_requests = tuple(
                PlatformDeploymentRequest(
                    entity_id,
                    int(entity["type"]),
                    target_lon=(
                        None
                        if (
                            target := reserve_target_by_member.get(entity_id)
                            or target_by_id.get(target_by_member.get(entity_id))
                        ) is None
                        else target.lon
                    ),
                    target_lat=None if target is None else target.lat,
                    target_id=(
                        None
                        if target is None or entity_id in reserve_target_by_member
                        else target.target_id
                    ),
                    target_type=(
                        None
                        if target is None or entity_id in reserve_target_by_member
                        else target.target_type
                    ),
                    deployment_role=(
                        "reserve_h"
                        if entity_id in reserve_target_by_member
                        else "core"
                    ),
                )
                for entity_id, entity in selected
            )
            spatial_skill = StaggeredDeploymentSkill(replace(
                self._skill.config, l_first_wave_fraction=1.0
            ))
            provisional_plan = spatial_skill.plan(
                deployment_bounds, provisional_requests
            )
            l_roster = tuple(
                int(point.entity_id)
                for point in sorted(
                    (
                        point
                        for point in provisional_plan.points
                        if int(point.entity_type) == 21002
                    ),
                    key=lambda point: (point.lateral_km, point.entity_id),
                )
            )
            core_count = (
                len(l_roster)
                if self._l_first_wave_per_target == 0
                else min(
                    len(l_roster),
                    self._l_first_wave_per_target
                    * self._expected_l_target_count,
                )
            )
            l_roles = {
                entity_id: "core" if rank < core_count else "reserve_l"
                for rank, entity_id in enumerate(l_roster)
            }
            self._mission_plan = PreMissionPlan(
                hm_assignments=target_by_member,
                entity_types=entity_types,
                entity_core={
                    entity_id: {**entity, "entity_id": entity_id}
                    for entity_id, entity in selected
                },
                l_roster=l_roster,
                l_roles=l_roles,
                reserve_packages=reserve_packages,
                reserve_target_regions=reserve_target_regions,
            )
        elif (
            self._mission_plan.entity_types != entity_types
            or set(self._mission_plan.l_roster) != set(entity_l_ids)
        ):
            raise RuntimeError("deployment reset changed the immutable entity roster")
        target_by_member = self._mission_plan.hm_assignments
        target_by_id = {target.target_id: target for target in self._public_targets}
        reserve_target_by_member = {
            member_id: target_by_id[package.provisional_target_id]
            for package in self._mission_plan.reserve_packages
            for member_id in package.member_ids
        }
        requests = []
        for entity_id, entity in selected:
            reserve_target = reserve_target_by_member.get(entity_id)
            target = (
                reserve_target
                if reserve_target is not None
                else target_by_id.get(target_by_member.get(entity_id))
            )
            requests.append(PlatformDeploymentRequest(
                entity_id,
                int(entity["type"]),
                target_lon=None if target is None else target.lon,
                target_lat=None if target is None else target.lat,
                target_id=(
                    None
                    if target is None or reserve_target is not None
                    else target.target_id
                ),
                target_type=(
                    None
                    if target is None or reserve_target is not None
                    else target.target_type
                ),
                deployment_role=(
                    "reserve_h"
                    if reserve_target is not None
                    else self._mission_plan.l_roles.get(entity_id, "core")
                ),
            ))
        requests = tuple(requests)
        plan = self._skill.plan(deployment_bounds, requests)
        if self._on_mission_plan is not None:
            self._on_mission_plan(self._mission_plan)
        if self._on_plan is not None:
            self._on_plan(plan)
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
