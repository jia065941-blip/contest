from dataclasses import FrozenInstanceError
import math

import pytest

from staggered_deployment_skill import (
    DeploymentBounds,
    PlatformDeploymentRequest,
    StaggeredDeploymentConfig,
    StaggeredDeploymentSkill,
    __version__,
    route_deployment_actions,
    to_deployment_actions,
)


EASY_E01_BOUNDS = DeploymentBounds(
    11.381719629926717,
    11.74134166473247,
    1.9625,
    2.6975,
    3000.0,
)
EASY_E01_MOTHERS = (
    (12.287003356448002, 2.4210636112609896),
    (12.261216545924443, 2.26370176033798),
    (12.384653356448002, 2.0459988887390104),
    (12.187477933388573, 2.6719665296108697),
)
EASY_E01_SENTRIES = (
    (12.109626545924442, 2.6480357396620198),
    (12.110847092737613, 2.5845269914299798),
    (12.018206187475833, 2.432034930630495),
    (12.006143356448002, 2.2426113887390104),
    (12.099143356448002, 2.05886138873901),
    (12.30853650496727, 2.2783984360510026),
    (12.2712582655112, 2.0458445675074217),
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


def request(
    entity_id,
    entity_type,
    target_lon=None,
    target_lat=None,
    target_id=None,
    target_type=None,
):
    return PlatformDeploymentRequest(
        entity_id,
        entity_type,
        target_lon,
        target_lat,
        target_id,
        target_type,
    )


def easy_e01_requests():
    h_ids = iter(range(2576, 2593))
    m_ids = iter(range(2611, 2635))
    result = []
    for target_id, target in enumerate(EASY_E01_MOTHERS, start=9401):
        result.extend(
            request(next(h_ids), 21000, *target, target_id, 9400)
            for _ in range(2)
        )
    for target_id, (target, h_count, m_count) in enumerate(
        zip(
            EASY_E01_SENTRIES,
            (2, 2, 1, 1, 1, 1, 1),
            (4, 4, 4, 3, 3, 3, 3),
        ),
        start=9601,
    ):
        result.extend(
            request(next(h_ids), 21000, *target, target_id, 9600)
            for _ in range(h_count)
        )
        result.extend(
            request(next(m_ids), 21001, *target, target_id, 9600)
            for _ in range(m_count)
        )
    result.extend(request(entity_id, 21002) for entity_id in range(2660, 2745))
    assert next(h_ids, None) is None
    assert next(m_ids, None) is None
    return tuple(result)


def target_ahead(area, axis, distance_km=60.0, lateral_fraction=0.5):
    east = area.width_km * lateral_fraction
    north = area.height_km * lateral_fraction
    if axis == "east":
        east = area.width_km + distance_km
    elif axis == "west":
        east = -distance_km
    elif axis == "north":
        north = area.height_km + distance_km
    else:
        north = -distance_km
    return (
        area.lon_min + east / area.km_per_longitude_degree,
        area.lat_min + north / 110.574,
    )


def target_local(area, axis, target):
    east = (target[0] - area.lon_min) * area.km_per_longitude_degree
    north = (target[1] - area.lat_min) * 110.574
    if axis == "east":
        return east, north
    if axis == "west":
        return area.width_km - east, north
    if axis == "north":
        return north, east
    return area.height_km - north, east


def incidence_angle(target_forward, target_lateral, points):
    ordered = sorted(points, key=lambda item: item.lateral_km)
    low, high = ordered[0], ordered[-1]
    first = (
        low.forward_km - target_forward,
        low.lateral_km - target_lateral,
    )
    second = (
        high.forward_km - target_forward,
        high.lateral_km - target_lateral,
    )
    cross = abs(first[0] * second[1] - first[1] * second[0])
    dot = first[0] * second[0] + first[1] * second[1]
    return math.degrees(math.atan2(cross, dot))


def surface_distance_km(first, second):
    radius_km = 6371.0088
    first_lat = math.radians(first[1])
    second_lat = math.radians(second[1])
    latitude_delta = second_lat - first_lat
    longitude_delta = math.radians(second[0] - first[0])
    haversine = (
        math.sin(latitude_delta / 2.0) ** 2
        + math.cos(first_lat)
        * math.cos(second_lat)
        * math.sin(longitude_delta / 2.0) ** 2
    )
    return 2.0 * radius_km * math.asin(math.sqrt(haversine))


@pytest.mark.parametrize("width_km", (40.0, 50.0, 60.0))
def test_first_l_wave_is_east_of_rear_with_at_least_the_configured_gap(width_km):
    area = bounds(width_km=width_km)
    plan = StaggeredDeploymentSkill().plan(
        area,
        tuple(request(index, 21002) for index in range(10))
        + (request(100, 21000), request(101, 21001)),
    )
    front = [point for point in plan.points if point.band == "front"]
    rear = [point for point in plan.points if point.band != "front"]
    separation = (
        min(point.longitude for point in front)
        - max(point.longitude for point in rear)
    ) * area.km_per_longitude_degree
    assert separation >= 15.0 - 1e-9


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
    area = bounds()
    requests = tuple(
        request(entity_id, entity_type, target_lon, target_lat)
        for entity_id, entity_type, target_lon, target_lat in (
            (9, 21002, None, None),
            (3, 21001, area.lon_min, area.lat_max),
            (7, 21000, area.lon_max, area.lat_min),
            (1, 21002, None, None),
        )
    )
    skill = StaggeredDeploymentSkill()
    assert skill.plan(area, requests) == skill.plan(area, reversed(requests))


def test_untargeted_hm_placement_is_deterministic_and_laterally_uniform():
    area = bounds(width_km=40.0, height_km=40.0)
    requests = tuple(request(index, 21000) for index in range(5)) + tuple(
        request(100 + index, 21001) for index in range(8)
    )
    skill = StaggeredDeploymentSkill()
    plan = skill.plan(area, requests)

    assert plan == skill.plan(area, reversed(requests))
    for entity_type in (21000, 21001):
        lateral = [
            point.lateral_km
            for point in plan.points
            if point.entity_type == entity_type
        ]
        assert min(lateral) <= plan.config.lateral_spacing_km
        assert max(lateral) >= area.height_km - plan.config.lateral_spacing_km


def test_large_counts_expand_to_multiple_columns_without_overflow():
    area = bounds(width_km=60.0, height_km=40.0)
    requests = tuple(request(index, 21002) for index in range(160)) + tuple(
        request(index, 21000 if index % 2 else 21001)
        for index in range(1000, 1160)
    )
    plan = StaggeredDeploymentSkill().plan(area, requests)
    front = [point for point in plan.points if point.band == "front"]
    rear = [point for point in plan.points if point.band != "front"]
    assert len({point.column for point in front}) > 1
    assert len({point.column for point in rear}) > 1
    assert all(area.lon_min <= point.longitude <= area.lon_max for point in plan.points)
    assert all(area.lat_min <= point.latitude <= area.lat_max for point in plan.points)
    assert min(point.forward_km for point in front) - max(
        point.forward_km for point in rear
    ) >= 15.0 - 1e-9
    assert len({(point.longitude, point.latitude) for point in plan.points}) == len(
        plan.points
    )


def test_north_axis_uses_latitude_as_the_front_direction():
    plan = StaggeredDeploymentSkill(
        StaggeredDeploymentConfig(front_axis="north")
    ).plan(bounds(width_km=40.0, height_km=50.0), (request(1, 21000), request(2, 21002)))
    by_id = {point.entity_id: point for point in plan.points}
    assert by_id[2].latitude > by_id[1].latitude
    assert (by_id[2].latitude - by_id[1].latitude) * 110.574 >= 15.0


def test_narrow_front_axis_is_rejected():
    with pytest.raises(ValueError, match="too narrow"):
        StaggeredDeploymentSkill().plan(
            bounds(width_km=20.0),
            (request(1, 21000), request(2, 21002)),
        )


def test_grid_capacity_is_rejected_instead_of_crossing_bounds():
    requests = tuple(request(index, 21002) for index in range(2000))
    with pytest.raises(ValueError, match="capacity"):
        StaggeredDeploymentSkill().plan(bounds(width_km=40.0), requests)


def test_default_config_and_seventy_percent_l_split():
    plan = StaggeredDeploymentSkill().plan(
        bounds(), tuple(request(index, 21002) for index in range(10))
    )
    assert plan.config.deployment_gap_km == 15.0
    assert plan.config.h_depth_km == 5.0
    assert plan.config.m_depth_km == 10.0
    assert plan.config.hm_follow_gap_km == 0.0
    assert plan.config.l_first_wave_fraction == 0.70
    assert plan.config.mother_crossfire_angle_degrees == 35.0
    assert plan.config.mother_crossfire_max_offset_km == 35.0
    assert plan.config.sentry_m_lead_km == 7.5
    assert plan.config.sentry_rendezvous_mode == "adaptive"
    assert plan.config.h_speed_mps == 200.0
    assert plan.config.m_speed_mps == 120.0
    assert plan.config.desired_m_arrival_lead_seconds == 5.0
    assert plan.config.min_sentry_m_lead_km == 10.0
    assert plan.config.max_sentry_m_lead_km is None
    assert [point.entity_id for point in plan.points if point.band == "front"] == list(
        range(7)
    )
    assert [point.entity_id for point in plan.points if point.band == "second"] == [
        7,
        8,
        9,
    ]


@pytest.mark.parametrize("fraction", (0.0, -0.1, 1.0001, math.inf, math.nan, True))
def test_first_wave_fraction_must_be_in_strict_range(fraction):
    with pytest.raises(ValueError, match="l_first_wave_fraction"):
        StaggeredDeploymentConfig(l_first_wave_fraction=fraction)


def test_second_l_and_m_share_follow_grid_without_coordinate_overlap():
    area = bounds()
    plan = StaggeredDeploymentSkill().plan(
        area,
        tuple(request(index, 21002) for index in range(10))
        + (request(100, 21000), request(101, 21001)),
    )
    front = [point for point in plan.points if point.band == "front"]
    second = [point for point in plan.points if point.band == "second"]
    h = [point for point in plan.points if point.wave_role == "h_lead"]
    m = [point for point in plan.points if point.wave_role == "m_follow"]
    coordinates = {(point.longitude, point.latitude) for point in plan.points}

    assert len(coordinates) == len(plan.points)
    assert {
        (point.column, point.row) for point in second
    }.isdisjoint((point.column, point.row) for point in m)
    assert min(point.forward_km for point in h) > max(
        point.forward_km for point in m
    )
    assert max(point.forward_km for point in second) < min(
        point.forward_km for point in front
    )
    assert max(point.lateral_km for point in second) - min(
        point.lateral_km for point in second
    ) >= area.height_km - plan.config.lateral_spacing_km - 1e-9


@pytest.mark.parametrize("axis", ("east", "west", "north", "south"))
def test_all_front_axes_keep_bounds_and_strict_gap(axis):
    area = bounds(width_km=40.0, height_km=40.0)
    plan = StaggeredDeploymentSkill(
        StaggeredDeploymentConfig(front_axis=axis)
    ).plan(
        area,
        tuple(request(index, 21002) for index in range(10))
        + (request(100, 21000),),
    )
    front = [point for point in plan.points if point.band == "front"]
    rear = [point for point in plan.points if point.band != "front"]

    assert min(point.forward_km for point in front) - max(
        point.forward_km for point in rear
    ) >= plan.config.deployment_gap_km - 1e-9
    assert all(area.lon_min <= point.longitude <= area.lon_max for point in plan.points)
    assert all(area.lat_min <= point.latitude <= area.lat_max for point in plan.points)


@pytest.mark.parametrize("axis", ("east", "west", "north", "south"))
def test_easy_40km_has_ordered_bands_and_same_target_hm_link(axis):
    area = bounds(width_km=40.0, height_km=40.0)
    target = (
        area.lon_min + (area.lon_max - area.lon_min) * 0.37,
        area.lat_min + (area.lat_max - area.lat_min) * 0.63,
    )
    plan = StaggeredDeploymentSkill(
        StaggeredDeploymentConfig(front_axis=axis)
    ).plan(
        area,
        tuple(request(index, 21002) for index in range(10))
        + (request(100, 21000, *target), request(101, 21001, *target)),
    )
    front = [point for point in plan.points if point.wave_role == "front"]
    h = next(point for point in plan.points if point.wave_role == "h_lead")
    m = next(point for point in plan.points if point.wave_role == "m_follow")
    second = [point for point in plan.points if point.wave_role == "second"]
    coordinates = {(point.longitude, point.latitude) for point in plan.points}
    forward_gap = h.forward_km - m.forward_km
    communication_distance = math.hypot(
        forward_gap, h.lateral_km - m.lateral_km
    )

    assert min(point.forward_km for point in front) > h.forward_km
    assert h.forward_km > max(point.forward_km for point in second + [m])
    assert 7.5 <= forward_gap <= 10.0
    assert communication_distance <= 15.0
    assert communication_distance < 20.0
    assert abs(h.lateral_km - m.lateral_km) <= 1.0
    assert len(coordinates) == len(plan.points)
    assert all(area.lon_min <= point.longitude <= area.lon_max for point in plan.points)
    assert all(area.lat_min <= point.latitude <= area.lat_max for point in plan.points)


@pytest.mark.parametrize("axis", ("east", "west", "north", "south"))
def test_easy_e01_actual_target_packages_keep_hm_centroids_and_reserved_grid(axis):
    requests = easy_e01_requests()
    skill = StaggeredDeploymentSkill(StaggeredDeploymentConfig(front_axis=axis))
    plan = skill.plan(EASY_E01_BOUNDS, requests)
    hm_only = skill.plan(
        EASY_E01_BOUNDS,
        tuple(item for item in requests if item.entity_type != 21002),
    )
    by_id = {point.entity_id: point for point in plan.points}
    hm_only_by_id = {point.entity_id: point for point in hm_only.points}
    request_by_id = {item.entity_id: item for item in requests}
    targets = EASY_E01_MOTHERS + EASY_E01_SENTRIES

    assert len([point for point in plan.points if point.entity_type == 21000]) == 17
    assert len([point for point in plan.points if point.entity_type == 21001]) == 24
    assert len([point for point in plan.points if point.entity_type == 21002]) == 85
    assert {
        entity_id: (point.forward_km, point.lateral_km)
        for entity_id, point in by_id.items()
        if point.entity_type in (21000, 21001)
    } == {
        entity_id: (point.forward_km, point.lateral_km)
        for entity_id, point in hm_only_by_id.items()
    }

    for target in targets:
        h = [
            by_id[entity_id]
            for entity_id, item in request_by_id.items()
            if item.entity_type == 21000
            and (item.target_lon, item.target_lat) == target
        ]
        m = [
            by_id[entity_id]
            for entity_id, item in request_by_id.items()
            if item.entity_type == 21001
            and (item.target_lon, item.target_lat) == target
        ]
        sample = next(
            item for item in request_by_id.values()
            if (item.target_lon, item.target_lat) == target
        )
        if not h or not m or sample.target_type != 9600:
            continue
        h_forward = sum(point.forward_km for point in h) / len(h)
        h_lateral = sum(point.lateral_km for point in h) / len(h)
        m_forward = sum(point.forward_km for point in m) / len(m)
        m_lateral = sum(point.lateral_km for point in m) / len(m)
        lateral_gap = abs(h_lateral - m_lateral)
        assert lateral_gap <= 3.0
        assert m_forward - h_forward >= plan.config.min_sentry_m_lead_km
        assert {point.formation_role for point in h} == {"h_follow"}
        assert {point.formation_role for point in m} == {"m_lead"}
        assert len({point.requested_lead for point in h + m}) == 1
        assert len({point.actual_lead for point in h + m}) == 1
        assert all(
            point.actual_separation_km == pytest.approx(m_forward - h_forward)
            for point in h + m
        )
        assert all(
            point.actual_lead == pytest.approx(m_forward - h_forward)
            for point in h + m
        )
        if not h[0].geometry_clamped:
            assert h[0].predicted_m_arrival_delta_seconds == pytest.approx(5.0)
            assert h[0].predicted_separation_at_m_20km == pytest.approx(
                14.0 + 1.0 / 3.0
            )

    for target_id, target in enumerate(EASY_E01_MOTHERS, start=9401):
        h = [
            point for point in plan.points
            if point.entity_type == 21000 and point.target_id == target_id
        ]
        target_forward, target_lateral = target_local(
            EASY_E01_BOUNDS, axis, target
        )
        achieved = incidence_angle(target_forward, target_lateral, h)
        assert {point.crossfire_role for point in h} == {"left", "right"}
        assert all(point.crossfire_angle == pytest.approx(achieved) for point in h)
        assert len({(point.longitude, point.latitude) for point in h}) == 2

    m_coordinates = {
        (point.longitude, point.latitude)
        for point in plan.points
        if point.entity_type == 21001
    }
    l_coordinates = {
        (point.longitude, point.latitude)
        for point in plan.points
        if point.entity_type == 21002
    }
    assert m_coordinates.isdisjoint(l_coordinates)
    assert len({(point.longitude, point.latitude) for point in plan.points}) == len(
        plan.points
    )


@pytest.mark.parametrize("axis", ("east", "west", "north", "south"))
@pytest.mark.parametrize("h_count", (2, 3))
def test_mother_crossfire_reaches_configured_angle_on_all_axes(axis, h_count):
    area = bounds(width_km=80.0, height_km=80.0)
    target = target_ahead(area, axis)
    plan = StaggeredDeploymentSkill(
        StaggeredDeploymentConfig(front_axis=axis)
    ).plan(
        area,
        tuple(
            request(
                index,
                21000,
                *target,
                target_id=9401,
                target_type=9400,
            )
            for index in range(h_count)
        ),
    )
    h = list(plan.points)
    target_forward, target_lateral = target_local(area, axis, target)
    achieved = incidence_angle(target_forward, target_lateral, h)
    laterals = sorted(point.lateral_km for point in h)
    forward = h[0].forward_km
    expected_offset = abs(target_forward - forward) * math.tan(
        math.radians(plan.config.mother_crossfire_angle_degrees / 2.0)
    )

    assert len({point.forward_km for point in h}) == 1
    assert sum(laterals) / len(laterals) == pytest.approx(target_lateral)
    assert laterals[-1] - target_lateral == pytest.approx(expected_offset)
    assert achieved == pytest.approx(35.0)
    assert all(point.crossfire_angle == pytest.approx(achieved) for point in h)
    assert {point.crossfire_role for point in h} == (
        {"left", "right"}
        if h_count == 2
        else {"left", "center", "right"}
    )
    assert {point.target_id for point in h} == {9401}
    assert {point.target_type for point in h} == {9400}


@pytest.mark.parametrize("axis", ("east", "west", "north", "south"))
def test_mother_crossfire_at_lateral_boundary_stays_unique(axis):
    area = bounds(width_km=80.0, height_km=80.0)
    target = target_ahead(area, axis, lateral_fraction=0.0)
    plan = StaggeredDeploymentSkill(
        StaggeredDeploymentConfig(front_axis=axis)
    ).plan(
        area,
        tuple(
            request(
                index,
                21000,
                *target,
                target_id=9401,
                target_type=9400,
            )
            for index in range(3)
        ),
    )
    target_forward, target_lateral = target_local(area, axis, target)
    achieved = incidence_angle(target_forward, target_lateral, plan.points)

    assert len({(point.longitude, point.latitude) for point in plan.points}) == 3
    assert min(point.lateral_km for point in plan.points) >= 0.0
    assert max(point.lateral_km for point in plan.points) <= 80.0 + 1e-9
    assert achieved >= 30.0
    assert all(
        point.crossfire_angle == pytest.approx(achieved)
        for point in plan.points
    )


def test_mother_crossfire_reports_angle_when_max_offset_cannot_reach_thirty():
    area = bounds(width_km=80.0, height_km=80.0)
    target = target_ahead(area, "east", distance_km=300.0)
    config = StaggeredDeploymentConfig(mother_crossfire_max_offset_km=5.0)
    plan = StaggeredDeploymentSkill(config).plan(
        area,
        tuple(
            request(
                index,
                21000,
                *target,
                target_id=9401,
                target_type=9400,
            )
            for index in range(2)
        ),
    )
    h = list(plan.points)
    target_forward, target_lateral = target_local(area, "east", target)
    achieved = incidence_angle(target_forward, target_lateral, h)

    assert (max(point.lateral_km for point in h) - min(
        point.lateral_km for point in h
    )) / 2.0 == pytest.approx(5.0)
    assert achieved < 30.0
    assert all(point.crossfire_angle == pytest.approx(achieved) for point in h)


def test_9600_hm_target_identity_keeps_corridor_limits_without_crossfire():
    area = bounds(width_km=40.0, height_km=40.0)
    target = target_ahead(area, "east", distance_km=20.0, lateral_fraction=0.3)
    plan = StaggeredDeploymentSkill().plan(
        area,
        tuple(
            request(
                index,
                21000,
                *target,
                target_id=9601,
                target_type=9600,
            )
            for index in range(2)
        )
        + tuple(
            request(
                100 + index,
                21001,
                *target,
                target_id=9601,
                target_type=9600,
            )
            for index in range(6)
        ),
    )
    h = [point for point in plan.points if point.entity_type == 21000]
    m = [point for point in plan.points if point.entity_type == 21001]
    h_forward = sum(point.forward_km for point in h) / len(h)
    h_lateral = sum(point.lateral_km for point in h) / len(h)
    m_forward = sum(point.forward_km for point in m) / len(m)
    m_lateral = sum(point.lateral_km for point in m) / len(m)
    actual_separation = m_forward - h_forward

    assert abs(h_lateral - m_lateral) <= 3.0
    assert actual_separation >= plan.config.min_sentry_m_lead_km
    assert all(point.crossfire_role is None for point in h + m)
    assert all(point.crossfire_angle is None for point in h + m)
    assert {point.formation_role for point in h} == {"h_follow"}
    assert {point.formation_role for point in m} == {"m_lead"}
    assert len({point.requested_lead for point in h + m}) == 1
    assert all(
        point.actual_separation_km == pytest.approx(actual_separation)
        for point in h + m
    )
    assert all(
        point.actual_lead == pytest.approx(actual_separation)
        for point in h + m
    )


@pytest.mark.parametrize("axis", ("east", "west", "north", "south"))
def test_9600_sentry_m_leads_h_symmetrically_on_every_axis(axis):
    area = bounds(width_km=40.0, height_km=40.0)
    target = target_ahead(area, axis, distance_km=20.0, lateral_fraction=0.4)
    plan = StaggeredDeploymentSkill(
        StaggeredDeploymentConfig(front_axis=axis)
    ).plan(
        area,
        tuple(
            request(index, 21000, *target, target_id=9601, target_type=9600)
            for index in range(2)
        )
        + tuple(
            request(100 + index, 21001, *target, target_id=9601, target_type=9600)
            for index in range(4)
        )
        + tuple(request(200 + index, 21002) for index in range(10)),
    )
    h = [point for point in plan.points if point.entity_type == 21000]
    m = [point for point in plan.points if point.entity_type == 21001]
    h_forward = sum(point.forward_km for point in h) / len(h)
    m_forward = sum(point.forward_km for point in m) / len(m)
    h_lateral = sum(point.lateral_km for point in h) / len(h)
    m_lateral = sum(point.lateral_km for point in m) / len(m)

    assert m_forward - h_forward >= plan.config.min_sentry_m_lead_km
    assert abs(m_lateral - h_lateral) <= 3.0
    assert len({(point.longitude, point.latitude) for point in plan.points}) == len(
        plan.points
    )
    assert all(area.lon_min <= point.longitude <= area.lon_max for point in plan.points)
    assert all(area.lat_min <= point.latitude <= area.lat_max for point in plan.points)


@pytest.mark.parametrize("axis", ("east", "west", "north", "south"))
def test_adaptive_sentry_uses_target_distance_and_predicts_rendezvous(axis):
    area = bounds(width_km=60.0, height_km=60.0)
    target = target_ahead(area, axis, distance_km=20.0, lateral_fraction=0.4)
    plan = StaggeredDeploymentSkill(
        StaggeredDeploymentConfig(front_axis=axis)
    ).plan(
        area,
        tuple(
            request(index, 21000, *target, target_id=9601, target_type=9600)
            for index in range(2)
        )
        + tuple(
            request(100 + index, 21001, *target, target_id=9601, target_type=9600)
            for index in range(4)
        ),
    )
    h = [point for point in plan.points if point.entity_type == 21000]
    m = [point for point in plan.points if point.entity_type == 21001]
    h_position = (
        sum(point.longitude for point in h) / len(h),
        sum(point.latitude for point in h) / len(h),
    )
    distance = surface_distance_km(h_position, target)
    expected_lead = 0.4 * distance + 0.6
    actual_lead = (
        sum(point.forward_km for point in m) / len(m)
        - sum(point.forward_km for point in h) / len(h)
    )

    assert expected_lead > 20.0
    assert actual_lead == pytest.approx(expected_lead)
    assert all(point.requested_lead == pytest.approx(expected_lead) for point in h + m)
    assert all(point.actual_lead == pytest.approx(actual_lead) for point in h + m)
    assert all(point.predicted_m_arrival_delta_seconds == pytest.approx(5.0) for point in h + m)
    assert all(
        point.predicted_separation_at_m_20km == pytest.approx(14.0 + 1.0 / 3.0)
        for point in h + m
    )
    assert {point.geometry_clamped for point in h + m} == {False}
    assert abs(
        sum(point.lateral_km for point in m) / len(m)
        - sum(point.lateral_km for point in h) / len(h)
    ) <= 3.0


@pytest.mark.parametrize(
    ("difficulty", "axis_length", "target_distance", "clamped"),
    (
        ("Easy", 40.0, 20.0, False),
        ("Medium", 60.0, 20.0, False),
        ("Hard", 80.0, 40.0, False),
    ),
)
def test_adaptive_sentry_easy_medium_hard_geometry(
    difficulty, axis_length, target_distance, clamped
):
    area = bounds(width_km=axis_length, height_km=40.0)
    target = target_ahead(area, "east", distance_km=target_distance)
    plan = StaggeredDeploymentSkill().plan(
        area,
        tuple(
            request(index, 21000, *target, target_id=9601, target_type=9600)
            for index in range(2)
        )
        + tuple(
            request(100 + index, 21001, *target, target_id=9601, target_type=9600)
            for index in range(4)
        )
        + tuple(request(200 + index, 21002) for index in range(10)),
    )
    sentry = [point for point in plan.points if point.target_type == 9600]

    assert difficulty in {"Easy", "Medium", "Hard"}
    assert {point.geometry_clamped for point in sentry} == {clamped}
    assert len({(point.longitude, point.latitude) for point in plan.points}) == len(
        plan.points
    )
    assert all(area.lon_min <= point.longitude <= area.lon_max for point in plan.points)
    assert all(area.lat_min <= point.latitude <= area.lat_max for point in plan.points)


def test_easy_e01_adaptive_uses_full_forward_extent_and_exact_diagnostics():
    plan = StaggeredDeploymentSkill().plan(EASY_E01_BOUNDS, easy_e01_requests())
    packages = [
        [point for point in plan.points if point.target_id == target_id]
        for target_id in range(9601, 9608)
    ]
    within_bounds = [
        package for package in packages
        if package[0].requested_lead <= EASY_E01_BOUNDS.width_km
    ]
    beyond_bounds = [
        package for package in packages
        if package[0].requested_lead > EASY_E01_BOUNDS.width_km
    ]

    assert len(within_bounds) >= 4
    assert all(package[0].requested_lead > 28.0 for package in within_bounds)
    assert all(package[0].actual_lead > 20.0 for package in within_bounds)
    assert all(
        package[0].actual_lead == pytest.approx(package[0].requested_lead)
        for package in within_bounds
    )
    assert all(
        package[0].predicted_m_arrival_delta_seconds == pytest.approx(5.0)
        for package in within_bounds
    )
    assert all(package[0].clamp_reason is None for package in within_bounds)
    assert beyond_bounds
    assert all(package[0].geometry_clamped for package in beyond_bounds)
    assert all(package[0].clamp_reason == "boundary" for package in beyond_bounds)


def test_adaptive_m_reserves_front_range_before_l_uses_global_remainder():
    area = bounds(width_km=40.0, height_km=40.0)
    target = target_ahead(area, "east", distance_km=50.0)
    hm = tuple(
        request(index, 21000, *target, target_id=9601, target_type=9600)
        for index in range(2)
    ) + tuple(
        request(100 + index, 21001, *target, target_id=9601, target_type=9600)
        for index in range(4)
    )
    skill = StaggeredDeploymentSkill()
    hm_only = skill.plan(area, hm)
    full = skill.plan(
        area,
        hm + tuple(request(1000 + index, 21002) for index in range(400)),
    )
    hm_only_positions = {
        point.entity_id: (point.forward_km, point.lateral_km)
        for point in hm_only.points
    }
    full_positions = {
        point.entity_id: (point.forward_km, point.lateral_km)
        for point in full.points
        if point.entity_type != 21002
    }
    m = [point for point in full.points if point.entity_type == 21001]

    assert full_positions == hm_only_positions
    assert sum(point.forward_km for point in m) / len(m) > 35.0
    assert len({(point.longitude, point.latitude) for point in full.points}) == len(
        full.points
    )


def test_adaptive_unique_coordinate_fallback_reports_capacity_clamp():
    area = bounds(width_km=60.0, height_km=40.0)
    target = (
        area.lon_min + 5.0 / area.km_per_longitude_degree,
        area.lat_min + 19.5 / 110.574,
    )
    requests = []
    for target_id, base_id in ((9601, 0), (9602, 100)):
        requests.append(
            request(
                base_id,
                21000,
                *target,
                target_id=target_id,
                target_type=9600,
            )
        )
        requests.extend(
            request(
                base_id + index + 1,
                21001,
                *target,
                target_id=target_id,
                target_type=9600,
            )
            for index in range(3)
        )
    plan = StaggeredDeploymentSkill().plan(area, requests)
    first = [point for point in plan.points if point.target_id == 9601]
    second = [point for point in plan.points if point.target_id == 9602]

    assert {point.actual_lead for point in first} == {10.0}
    assert {point.clamp_reason for point in first} == {None}
    assert {point.actual_lead for point in second} == {11.0}
    assert {point.clamp_reason for point in second} == {"capacity"}
    assert len({(point.longitude, point.latitude) for point in plan.points}) == len(
        plan.points
    )


def test_adaptive_near_target_uses_minimum_lead_without_geometry_clamp():
    area = bounds(width_km=60.0, height_km=40.0)
    target = (
        area.lon_min + 5.0 / area.km_per_longitude_degree,
        (area.lat_min + area.lat_max) / 2.0,
    )
    plan = StaggeredDeploymentSkill().plan(
        area,
        (
            request(1, 21000, *target, target_id=9601, target_type=9600),
            request(2, 21001, *target, target_id=9601, target_type=9600),
        ),
    )

    assert all(point.requested_lead < 10.0 for point in plan.points)
    assert all(point.desired_separation_km == 10.0 for point in plan.points)
    assert all(point.actual_lead == pytest.approx(10.0) for point in plan.points)
    assert {point.geometry_clamped for point in plan.points} == {False}


def test_adaptive_geometry_boundary_and_explicit_maximum_are_diagnosed():
    easy_area = bounds(width_km=35.0, height_km=40.0)
    easy_target = target_ahead(easy_area, "east", distance_km=100.0)
    easy_plan = StaggeredDeploymentSkill().plan(
        easy_area,
        (
            request(1, 21000, *easy_target, target_id=9601, target_type=9600),
            request(2, 21001, *easy_target, target_id=9601, target_type=9600),
        ),
    )
    capped_area = bounds(width_km=80.0, height_km=40.0)
    capped_target = target_ahead(capped_area, "east", distance_km=100.0)
    capped_plan = StaggeredDeploymentSkill(
        StaggeredDeploymentConfig(max_sentry_m_lead_km=15.0)
    ).plan(
        capped_area,
        (
            request(1, 21000, *capped_target, target_id=9601, target_type=9600),
            request(2, 21001, *capped_target, target_id=9601, target_type=9600),
        ),
    )

    assert all(point.geometry_clamped for point in easy_plan.points)
    assert {point.clamp_reason for point in easy_plan.points} == {"boundary"}
    assert all(point.actual_lead < point.requested_lead for point in easy_plan.points)
    assert all(point.requested_lead > 15.0 for point in capped_plan.points)
    assert all(point.actual_lead == pytest.approx(15.0) for point in capped_plan.points)
    assert {point.geometry_clamped for point in capped_plan.points} == {False}


@pytest.mark.parametrize("missing_type", (21000, 21001))
def test_9600_target_without_both_h_and_m_keeps_legacy_placement(missing_type):
    area = bounds(width_km=40.0, height_km=40.0)
    target = target_ahead(area, "east", distance_km=20.0)
    present_type = 21001 if missing_type == 21000 else 21000
    plan = StaggeredDeploymentSkill().plan(
        area,
        (
            request(
                1,
                present_type,
                *target,
                target_id=9601,
                target_type=9600,
            ),
        ),
    )

    assert plan.points[0].formation_role is None
    assert plan.points[0].desired_separation_km is None
    assert plan.points[0].actual_separation_km is None
    if present_type == 21000:
        assert plan.points[0].forward_km >= 15.0
    else:
        assert plan.points[0].forward_km <= 15.0


@pytest.mark.parametrize("h_count", (1, 2, 3))
@pytest.mark.parametrize("m_count", range(4, 13))
def test_target_corridors_expand_package_sizes_across_columns(h_count, m_count):
    area = bounds(width_km=40.0, height_km=40.0)
    target = (area.lon_max, area.lat_min)
    plan = StaggeredDeploymentSkill().plan(
        area,
        tuple(request(index, 21000, *target) for index in range(h_count))
        + tuple(request(100 + index, 21001, *target) for index in range(m_count)),
    )
    h = [point for point in plan.points if point.entity_type == 21000]
    m = [point for point in plan.points if point.entity_type == 21001]
    h_lateral = sum(point.lateral_km for point in h) / len(h)
    m_lateral = sum(point.lateral_km for point in m) / len(m)
    distance = math.hypot(
        sum(point.forward_km for point in h) / len(h)
        - sum(point.forward_km for point in m) / len(m),
        h_lateral - m_lateral,
    )

    assert len({point.column for point in h}) == min(h_count, 2)
    assert len({point.column for point in m}) > 1
    assert max(point.lateral_km for point in h + m) <= area.height_km
    assert min(point.lateral_km for point in h + m) >= 0.0
    assert abs(h_lateral - m_lateral) <= 3.0
    assert distance <= 15.0


@pytest.mark.parametrize("axis", ("east", "west", "north", "south"))
def test_hm_follow_target_transverse_coordinate_for_every_axis(axis):
    area = bounds(width_km=40.0, height_km=40.0)
    if axis in {"east", "west"}:
        low_target = (
            area.lon_min,
            area.lat_min + (area.lat_max - area.lat_min) * 0.25,
        )
        high_target = (
            area.lon_max,
            area.lat_min + (area.lat_max - area.lat_min) * 0.75,
        )
        low_coordinate = low_target[1]
        high_coordinate = high_target[1]
        coordinate = lambda point: point.latitude
        scale = 110.574
    else:
        low_target = (
            area.lon_min + (area.lon_max - area.lon_min) * 0.25,
            area.lat_min,
        )
        high_target = (
            area.lon_min + (area.lon_max - area.lon_min) * 0.75,
            area.lat_max,
        )
        low_coordinate = low_target[0]
        high_coordinate = high_target[0]
        coordinate = lambda point: point.longitude
        scale = area.km_per_longitude_degree

    plan = StaggeredDeploymentSkill(
        StaggeredDeploymentConfig(front_axis=axis)
    ).plan(
        area,
        (
            request(2, 21001, *high_target),
            request(1, 21000, *low_target),
        ),
    )
    by_id = {point.entity_id: point for point in plan.points}
    assert coordinate(by_id[1]) < coordinate(by_id[2])
    assert abs(coordinate(by_id[1]) - low_coordinate) * scale <= 0.5 + 1e-9
    assert abs(coordinate(by_id[2]) - high_coordinate) * scale <= 0.5 + 1e-9


def test_legacy_hm_depth_alias_controls_m_follow_depth():
    legacy = StaggeredDeploymentConfig(hm_depth_km=7.0)
    positional = StaggeredDeploymentConfig("east", 15.0, 5.0, 6.0, 1.0, 0.7)
    current = StaggeredDeploymentConfig(m_depth_km=8.0)

    assert legacy.hm_depth_km == legacy.m_depth_km == 7.0
    assert positional.hm_depth_km == positional.m_depth_km == 6.0
    assert current.hm_depth_km == current.m_depth_km == 8.0


def test_hm_follow_gap_accepts_zero_and_rejects_negative_values():
    assert StaggeredDeploymentConfig(hm_follow_gap_km=0).hm_follow_gap_km == 0.0
    with pytest.raises(ValueError, match="hm_follow_gap_km"):
        StaggeredDeploymentConfig(hm_follow_gap_km=-0.1)


@pytest.mark.parametrize(
    "value",
    (-0.1, 15.1, math.inf, -math.inf, math.nan, True),
)
def test_sentry_m_lead_is_finite_and_bounded(value):
    with pytest.raises(ValueError, match="sentry_m_lead_km"):
        StaggeredDeploymentConfig(sentry_m_lead_km=value)


@pytest.mark.parametrize("value", (0.0, 15.0))
def test_sentry_m_lead_accepts_closed_range_boundaries(value):
    assert StaggeredDeploymentConfig(sentry_m_lead_km=value).sentry_m_lead_km == value


@pytest.mark.parametrize("mode", ("fixed", "adaptive", "FIXED", "ADAPTIVE"))
def test_sentry_rendezvous_mode_accepts_supported_values(mode):
    assert StaggeredDeploymentConfig(
        sentry_rendezvous_mode=mode
    ).sentry_rendezvous_mode == mode.lower()


@pytest.mark.parametrize(
    "kwargs",
    (
        {"sentry_rendezvous_mode": "automatic"},
        {"h_speed_mps": 0.0},
        {"m_speed_mps": -1.0},
        {"desired_m_arrival_lead_seconds": -0.1},
        {"min_sentry_m_lead_km": -0.1},
        {"max_sentry_m_lead_km": -0.1},
        {"min_sentry_m_lead_km": 12.0, "max_sentry_m_lead_km": 10.0},
    ),
)
def test_adaptive_sentry_config_is_strictly_validated(kwargs):
    with pytest.raises(ValueError, match="sentry|speed|arrival"):
        StaggeredDeploymentConfig(**kwargs)


@pytest.mark.parametrize("desired", (0.0, 3.0, 7.5, 12.0, 15.0))
def test_sentry_m_lead_config_controls_achieved_grid_separation(desired):
    area = bounds(width_km=40.0, height_km=40.0)
    target = target_ahead(area, "east", distance_km=20.0)
    plan = StaggeredDeploymentSkill(
        StaggeredDeploymentConfig(
            sentry_rendezvous_mode="fixed",
            sentry_m_lead_km=desired,
        )
    ).plan(
        area,
        tuple(
            request(index, 21000, *target, target_id=9601, target_type=9600)
            for index in range(2)
        )
        + tuple(
            request(100 + index, 21001, *target, target_id=9601, target_type=9600)
            for index in range(4)
        ),
    )
    h = [point for point in plan.points if point.entity_type == 21000]
    m = [point for point in plan.points if point.entity_type == 21001]
    actual = (
        sum(point.forward_km for point in m) / len(m)
        - sum(point.forward_km for point in h) / len(h)
    )

    assert actual >= 0.0
    assert actual <= 15.0
    assert actual == pytest.approx(desired, abs=1.5)
    assert all(point.desired_separation_km == desired for point in h + m)
    assert all(point.actual_separation_km == pytest.approx(actual) for point in h + m)
    assert all(point.requested_lead == desired for point in h + m)
    assert all(point.actual_lead == pytest.approx(actual) for point in h + m)


@pytest.mark.parametrize(
    ("target_lon", "target_lat"),
    ((120.0, None), (None, 30.0), (math.nan, 30.0), (120.0, math.inf)),
)
def test_target_coordinates_must_be_paired_and_finite(target_lon, target_lat):
    with pytest.raises(ValueError, match="target"):
        request(1, 21000, target_lon, target_lat)


@pytest.mark.parametrize(
    "factory",
    (
        lambda: request(1, 21000, 120.0, 30.0, target_id=1),
        lambda: request(1, 21000, 120.0, 30.0, target_type=9400),
        lambda: request(1, 21000, 120.0, 30.0, target_id=True, target_type=9400),
        lambda: request(1, 21000, 120.0, 30.0, target_id=-1, target_type=9400),
        lambda: request(1, 21000, 120.0, 30.0, target_id=1, target_type=9500),
        lambda: request(1, 21000, target_id=1, target_type=9400),
        lambda: request(1, 21002, 120.0, 30.0, target_id=1, target_type=9400),
    ),
)
def test_target_identity_is_strictly_validated(factory):
    with pytest.raises(ValueError, match="target|H/M"):
        factory()


@pytest.mark.parametrize(
    ("target_lon", "target_lat"),
    ((-180.1, 0.0), (180.1, 0.0), (0.0, -90.1), (0.0, 90.1)),
)
def test_target_coordinates_must_be_valid_wgs84(target_lon, target_lat):
    with pytest.raises(ValueError, match="target"):
        request(1, 21000, target_lon, target_lat)


def test_same_target_id_requires_consistent_type_and_position():
    area = bounds()
    first = request(
        1, 21000, area.lon_max, area.lat_min,
        target_id=7, target_type=9400,
    )
    moved = request(
        2, 21000, area.lon_max, area.lat_max,
        target_id=7, target_type=9400,
    )
    changed_type = request(
        2, 21000, area.lon_max, area.lat_min,
        target_id=7, target_type=9600,
    )

    with pytest.raises(ValueError, match="consistent"):
        StaggeredDeploymentSkill().plan(area, (first, moved))
    with pytest.raises(ValueError, match="consistent"):
        StaggeredDeploymentSkill().plan(area, (first, changed_type))


@pytest.mark.parametrize(
    "kwargs",
    (
        {"mother_crossfire_angle_degrees": 29.9},
        {"mother_crossfire_angle_degrees": 180.0},
        {"mother_crossfire_angle_degrees": math.nan},
        {"mother_crossfire_max_offset_km": 0.0},
        {"mother_crossfire_max_offset_km": math.inf},
    ),
)
def test_crossfire_config_is_strictly_validated(kwargs):
    with pytest.raises(ValueError, match="mother_crossfire"):
        StaggeredDeploymentConfig(**kwargs)


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
    assert __version__ == "0.8.0"


def test_missing_runtime_owner_is_rejected():
    plan = StaggeredDeploymentSkill().plan(bounds(), (request(1, 21000),))
    with pytest.raises(ValueError, match="missing owner"):
        route_deployment_actions(plan, {})
