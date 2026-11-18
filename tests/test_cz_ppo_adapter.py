import math

from user_agents.cz_ppo.compat import CoordinateAdapter, NEW_BOUNDS, OLD_BOUNDS


def test_coordinate_adapter_round_trip_all20():
    for map_name in OLD_BOUNDS:
        adapter = CoordinateAdapter(map_name)
        samples = (
            (adapter.lon_min, adapter.lat_min),
            (adapter.lon_max, adapter.lat_max),
            ((adapter.lon_min + adapter.lon_max) / 2, (adapter.lat_min + adapter.lat_max) / 2),
        )
        for old in samples:
            new = adapter.to_new_point(old)
            restored = adapter.to_old_point([new[0], new[1], 0])
            assert math.isclose(old[0], restored[0], abs_tol=1e-12)
            assert math.isclose(old[1], restored[1], abs_tol=1e-12)


def test_coordinate_adapter_matches_official_e01_bounds():
    adapter = CoordinateAdapter("E01")
    assert math.isclose(adapter.x_min, -62747.211386, abs_tol=1e-6)
    assert math.isclose(adapter.x_max, 62747.211386, abs_tol=1e-6)
    assert math.isclose(adapter.y_min, -40636.447751, abs_tol=1e-6)
    assert math.isclose(adapter.y_max, 40636.447751, abs_tol=1e-6)


def test_all20_use_exact_official_cartesian_bounds():
    for map_name, expected in NEW_BOUNDS.items():
        adapter = CoordinateAdapter(map_name)
        assert (adapter.x_min, adapter.x_max, adapter.y_min, adapter.y_max) == expected


def test_public_waypoint_boundary_projection_does_not_change_track_conversion():
    adapter = CoordinateAdapter("M04")
    outside = adapter.to_new_point([11.98503790342878, 2.762623372974882])
    assert outside[1] > adapter.y_max
    submitted = adapter.bound_new_point(outside)
    assert submitted == [outside[0], adapter.y_max]
    # Projection is an output-boundary operation; the raw conversion remains
    # available to the policy's internal target tracker and PPO goal features.
    assert adapter.to_new_point([11.98503790342878, 2.762623372974882]) == outside
