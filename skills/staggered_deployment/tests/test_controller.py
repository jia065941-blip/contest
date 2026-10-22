from dataclasses import FrozenInstanceError
import math

import pytest

from staggered_deployment_skill import (
    DeploymentBounds,
    PlatformDeploymentRequest,
    StaggeredDeploymentConfig,
    StaggeredDeploymentSkill,
    route_deployment_actions,
    to_deployment_actions,
)


def bounds(width_km=50.0, height_km=40.0, midpoint_latitude=30.0):
    km_per_lon_degree = 111.320 * math.cos(math.radians(midpoint_latitude))
    lon_span = width_km / km_per_lon_degree
    lat_span = height_km / 110.574
    return DeploymentBounds(
        120.0,
        120.0 + lon_span,
        midpoint_latitude - lat_span / 2.0,
        midpoint_latitude + lat_span / 2.0,
        1200.0,
    )


def request(entity_id, entity_type):
    return PlatformDeploymentRequest(entity_id, entity_type)


@pytest.mark.parametrize("width_km", (40.0, 50.0, 60.0))
def test_l_is_east_of_hm_with_at_least_the_configured_gap(width_km):
    area = bounds(width_km=width_km)
    plan = StaggeredDeploymentSkill().plan(
        area,
        (request(3, 21002), request(1, 21000), request(2, 21001)),
    )
    front = [point for point in plan.points if point.entity_type == 21002]
    rear = [point for point in plan.points if point.entity_type in (21000, 21001)]
    separation = (
        min(point.longitude for point in front)
        - max(point.longitude for point in rear)
    ) * area.km_per_longitude_degree
    assert separation >= 30.0 - 1e-9


def test_every_point_is_inside_bounds_and_uses_requested_altitude():
    area = bounds()
    plan = StaggeredDeploymentSkill().plan(
        area,
        tuple(request(index, 21000 + index % 3) for index in range(30)),
    )
    assert all(area.lon_min <= point.longitude <= area.lon_max for point in plan.points)
    assert all(area.lat_min <= point.latitude <= area.lat_max for point in plan.points)
    assert {point.altitude for point in plan.points} == {1200.0}


def test_result_is_deterministic_and_independent_of_request_order():
    requests = tuple(
        request(entity_id, entity_type)
        for entity_id, entity_type in (
            (9, 21002),
            (3, 21001),
            (7, 21000),
            (1, 21002),
        )
    )
    skill = StaggeredDeploymentSkill()
    assert skill.plan(bounds(), requests) == skill.plan(bounds(), reversed(requests))


def test_large_counts_expand_to_multiple_columns_without_overflow():
    area = bounds(width_km=60.0, height_km=40.0)
    requests = tuple(request(index, 21002) for index in range(160)) + tuple(
        request(index, 21000 if index % 2 else 21001)
        for index in range(1000, 1160)
    )
    plan = StaggeredDeploymentSkill().plan(area, requests)
    front = [point for point in plan.points if point.band == "front"]
    rear = [point for point in plan.points if point.band == "rear"]
    assert len({point.column for point in front}) > 1
    assert len({point.column for point in rear}) > 1
    assert all(area.lon_min <= point.longitude <= area.lon_max for point in plan.points)
    assert all(area.lat_min <= point.latitude <= area.lat_max for point in plan.points)
    assert min(point.forward_km for point in front) - max(
        point.forward_km for point in rear
    ) >= 30.0 - 1e-9


def test_north_axis_uses_latitude_as_the_front_direction():
    plan = StaggeredDeploymentSkill(
        StaggeredDeploymentConfig(front_axis="north")
    ).plan(bounds(width_km=40.0, height_km=50.0), (request(1, 21000), request(2, 21002)))
    by_id = {point.entity_id: point for point in plan.points}
    assert by_id[2].latitude > by_id[1].latitude
    assert (by_id[2].latitude - by_id[1].latitude) * 110.574 >= 30.0


def test_narrow_front_axis_is_rejected():
    with pytest.raises(ValueError, match="too narrow"):
        StaggeredDeploymentSkill().plan(
            bounds(width_km=35.0),
            (request(1, 21000), request(2, 21002)),
        )


def test_grid_capacity_is_rejected_instead_of_crossing_bounds():
    requests = tuple(request(index, 21002) for index in range(300))
    with pytest.raises(ValueError, match="capacity"):
        StaggeredDeploymentSkill().plan(bounds(width_km=40.0), requests)


@pytest.mark.parametrize("entity_type", (20999, 21003, "21000"))
def test_only_h_m_l_types_are_accepted(entity_type):
    with pytest.raises(ValueError, match="entity_type"):
        PlatformDeploymentRequest(1, entity_type)


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        StaggeredDeploymentSkill().plan(
            bounds(),
            (request(1, 21000), request(1, 21002)),
        )


def test_dtos_are_frozen():
    area = bounds()
    with pytest.raises(FrozenInstanceError):
        area.altitude = 10.0


def test_runtime_actions_have_expected_shape_ids_and_owner_routes():
    plan = StaggeredDeploymentSkill().plan(
        bounds(),
        (request(8, 21002), request(2, 21000), request(5, 21001)),
    )
    actions = to_deployment_actions(plan)
    assert tuple(action[1] for action in actions) == (2, 5, 8)
    assert all(len(action) == 5 and action[0] == 0 for action in actions)
    assert all(isinstance(action[1], int) for action in actions)
    routed = route_deployment_actions(plan, {2: "west", 5: "west", 8: "front"})
    assert tuple(action[1] for action in routed["west"]) == (2, 5)
    assert tuple(action[1] for action in routed["front"]) == (8,)


def test_missing_runtime_owner_is_rejected():
    plan = StaggeredDeploymentSkill().plan(bounds(), (request(1, 21000),))
    with pytest.raises(ValueError, match="missing owner"):
        route_deployment_actions(plan, {})
