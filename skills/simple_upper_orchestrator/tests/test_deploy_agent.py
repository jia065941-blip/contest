import numpy as np
import pytest

from hm_public_strike_packages_skill import PublicStrikeConfig, PublicTargetTrack
from staggered_deployment_skill import StaggeredDeploymentConfig

from simple_upper_orchestrator_skill.deploy_agent import (
    PreMissionPlan,
    StaggeredDeployAgent,
    rectangle_bounds,
)


RECTANGLE = [[[0, 0], [1, 0], [1, 1], [0, 1]]]


def test_deploy_returns_nx5_batches_then_completion():
    agent = StaggeredDeployAgent(
        RECTANGLE, RECTANGLE,
        config=StaggeredDeploymentConfig(deployment_gap_km=1, l_depth_km=1, hm_depth_km=1),
        batch_size=2, launch_altitude=1000,
    )
    agent.set_observation({"entities": {
        1: {"type": 21000, "position": {"alt": 1000}},
        2: {"type": 21001, "position": {"alt": 1000}},
        3: {"type": 21002, "position": {"alt": 1000}},
        99: {"type": 9202, "position": {"alt": 0}},
    }})
    first, second, complete = agent.get_action(), agent.get_action(), agent.get_action()
    assert first.shape == (2, 5)
    assert second.shape == (1, 5)
    assert np.array_equal(complete, np.array([[1, -1, 0, 0, 0]], dtype=np.float64))
    assert set(np.concatenate((first, second))[:, 1].astype(int)) == {1, 2, 3}


def test_deployment_geometry_fails_closed():
    with pytest.raises(ValueError, match="axis-aligned"):
        rectangle_bounds([[[0, 0], [1, 0], [0.5, 1], [0, 1]]])
    with pytest.raises(ValueError, match="identical"):
        StaggeredDeployAgent(
            RECTANGLE, [[[0, 0], [2, 0], [2, 1], [0, 1]]],
            config=StaggeredDeploymentConfig(), batch_size=10, launch_altitude=1000,
        )


def test_deploy_reports_two_l_waves_and_target_aligns_hm():
    plans = []
    targets = (
        PublicTargetTrack(901, 9400, 0.8, 0.8, 0, 24, 5),
        PublicTargetTrack(902, 9600, 0.8, 0.2, 0, 18, 2),
    )
    agent = StaggeredDeployAgent(
        RECTANGLE, RECTANGLE,
        config=StaggeredDeploymentConfig(
            deployment_gap_km=1, l_depth_km=1, hm_depth_km=1,
            l_first_wave_fraction=0.5,
        ),
        batch_size=20,
        launch_altitude=1000,
        on_plan=plans.append,
        public_targets=targets,
        hm_config=PublicStrikeConfig(),
    )
    entities = {
        **{index: {"type": 21000, "position": {"alt": 1000}} for index in range(1, 5)},
        **{100 + index: {"type": 21002, "position": {"alt": 1000}} for index in range(4)},
    }
    agent.set_observation({"entities": entities})

    rows = agent.get_action()

    assert rows.shape == (8, 5)
    assert len(plans) == 1
    l_bands = [point.band for point in plans[0].points if point.entity_type == 21002]
    assert l_bands.count("front") == 2
    assert l_bands.count("second") == 2
    assert all(point.band == "rear" for point in plans[0].points if point.entity_type == 21000)


def test_mission_callback_is_first_immutable_and_survives_reset():
    events = []
    targets = (
        PublicTargetTrack(901, 9400, 0.8, 0.8, 0, 24, 5),
        PublicTargetTrack(902, 9600, 0.8, 0.2, 0, 18, 2),
    )
    agent = StaggeredDeployAgent(
        RECTANGLE,
        RECTANGLE,
        config=StaggeredDeploymentConfig(
            deployment_gap_km=1,
            l_depth_km=1,
            hm_depth_km=1,
        ),
        batch_size=20,
        launch_altitude=1000,
        on_mission_plan=lambda plan: events.append(("mission", plan)),
        on_plan=lambda plan: events.append(("deployment", plan)),
        public_targets=targets,
        hm_config=PublicStrikeConfig(),
    )
    observation = {"entities": {
        1: {"type": 21000, "position": {"alt": 1000}},
        2: {"type": 21000, "position": {"alt": 1000}},
        3: {"type": 21001, "position": {"alt": 1000}},
        4: {"type": 21002, "position": {"alt": 1000}},
        5: {"type": 21002, "position": {"alt": 1000}},
    }}
    agent.set_observation(observation)

    agent.get_action()

    assert [event for event, _ in events] == ["mission", "deployment"]
    mission = events[0][1]
    assert isinstance(mission, PreMissionPlan)
    assert mission.entity_types == {
        1: 21000, 2: 21000, 3: 21001, 4: 21002, 5: 21002,
    }
    assert mission.l_roster == (4, 5)
    assert mission.l_roles == {4: "core", 5: "core"}
    assert set(mission.hm_assignments) <= {1, 2, 3}
    with pytest.raises(TypeError):
        mission.hm_assignments[1] = 901
    with pytest.raises(TypeError):
        mission.entity_core[1]["type"] = 21002

    agent.reset()
    agent.set_observation({"entities": {
        entity_id: {
            **entity,
            "position": {**entity["position"], "lon": float(entity_id)},
        }
        for entity_id, entity in observation["entities"].items()
    }})
    agent.get_action()

    assert [event for event, _ in events] == [
        "mission", "deployment", "mission", "deployment",
    ]
    assert events[2][1] is mission


def test_easy_style_h_reserve_two_is_staged_by_latitude_region():
    missions = []
    plans = []
    targets = tuple(
        PublicTargetTrack(
            900 + index,
            9400 if index < 4 else 9600,
            0.8,
            0.05 + index * 0.08,
            0,
            24 if index < 4 else 18,
            5 if index < 4 else 2,
        )
        for index in range(11)
    )
    agent = StaggeredDeployAgent(
        RECTANGLE,
        RECTANGLE,
        config=StaggeredDeploymentConfig(
            deployment_gap_km=1,
            l_depth_km=1,
            hm_depth_km=1,
        ),
        batch_size=20,
        launch_altitude=1000,
        on_mission_plan=missions.append,
        on_plan=plans.append,
        public_targets=targets,
        hm_config=PublicStrikeConfig(reserve_h=2),
    )
    agent.set_observation({"entities": {
        entity_id: {"type": 21000, "position": {"alt": 1000}}
        for entity_id in range(1, 7)
    }})

    agent.get_action()

    mission = missions[0]
    assert len(mission.reserve_packages) == 2
    assert {package.responsibility_region for package in mission.reserve_packages} == {
        "north", "south",
    }
    assert all(package.state == "STAGING" for package in mission.reserve_packages)
    reserve_ids = {
        member_id
        for package in mission.reserve_packages
        for member_id in package.member_ids
    }
    assert reserve_ids == {5, 6}
    assert reserve_ids.isdisjoint(mission.hm_assignments)
    points = {point.entity_id: point for point in plans[0].points}
    assert all(points[member_id].target_id is None for member_id in reserve_ids)
    assert {
        points[member_id].deployment_role for member_id in reserve_ids
    } == {"reserve_h"}
    assert all(
        points[member_id].deployment_role == "core"
        for member_id in points.keys() - reserve_ids
    )
    for package in mission.reserve_packages:
        provisional = next(
            target
            for target in targets
            if target.target_id == package.provisional_target_id
        )
        for member_id in package.member_ids:
            assert points[member_id].latitude == pytest.approx(
                provisional.lat, abs=0.01
            )


def test_deploy_freezes_spatial_core_and_reserve_l_roster():
    missions = []
    plans = []
    agent = StaggeredDeployAgent(
        RECTANGLE,
        RECTANGLE,
        config=StaggeredDeploymentConfig(),
        batch_size=20,
        launch_altitude=1000,
        on_mission_plan=missions.append,
        on_plan=plans.append,
        l_first_wave_per_target=2,
        expected_l_target_count=2,
    )
    agent.set_observation({"entities": {
        entity_id: {"type": 21002, "position": {"alt": 1000}}
        for entity_id in range(100, 110)
    }})

    agent.get_action()

    mission = missions[0]
    plan = plans[0]
    assert tuple(mission.l_roles[owner] for owner in mission.l_roster) == (
        "core", "core", "core", "core",
        "reserve_l", "reserve_l", "reserve_l", "reserve_l", "reserve_l", "reserve_l",
    )
    points = {point.entity_id: point for point in plan.points}
    assert all(
        points[owner].deployment_role == mission.l_roles[owner]
        for owner in mission.l_roster
    )
    assert {
        points[owner].wave_role
        for owner in mission.l_roster
        if mission.l_roles[owner] == "reserve_l"
    } == {"reserve_l"}
    assert tuple(
        point.entity_id
        for point in sorted(plan.points, key=lambda point: (point.lateral_km, point.entity_id))
    ) == mission.l_roster
