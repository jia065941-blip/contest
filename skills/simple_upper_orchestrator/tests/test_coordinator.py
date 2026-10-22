from pathlib import Path

import numpy as np

from hm_public_strike_packages_skill import PublicStrikeConfig
from l_satellite_strike_packages_skill import SatelliteStrikeConfig
from staggered_deployment_skill import StaggeredDeploymentConfig

from simple_upper_orchestrator_skill.config import OrchestratorConfig
from simple_upper_orchestrator_skill.coordinator import TeamCoordinator


class Policy:
    def __init__(self):
        self.encode_calls = []
        self.reset_calls = 0

    def encode_motion_observation(self, observation, **kwargs):
        self.encode_calls.append((observation["step"], kwargs["entity_id"], kwargs["goal_position"]))
        return np.zeros(3)

    def prepare_action_mask(self, observation, mask):
        return mask

    def select_action(self, observation, mask):
        return 1

    def reset_episode(self):
        self.reset_calls += 1


def config(**l_values):
    return OrchestratorConfig(
        Path("."), Path("checkpoint"), 100,
        deployment=StaggeredDeploymentConfig(deployment_gap_km=1),
        l_strike=SatelliteStrikeConfig(**l_values),
        hm_strike=PublicStrikeConfig(),
    )


def observation(step, owner, kind, *, peers=(), tracks=None):
    return {
        "step": step,
        "self": {
            "entity_id": owner, "type": kind, "health": 100,
            "position": {"lon": float(owner), "lat": 0.0},
            "detectInfo": tracks or {},
            "commRangeInfo": {peer: {} for peer in peers},
        },
        "_orchestrator_launch_step": 0,
        "_orchestrator_maneuver_state": 0,
    }


def hidden(target_id, lon=9.0):
    return {"entity_id": target_id, "entity_type": 9500, "lla": {"x": lon, "y": 2.0}, "time": 1}


def test_once_per_step_owner_routing_and_satellite_singleton():
    policy = Policy()
    members = ((1, 11), (2, 12))
    coordinator = TeamCoordinator(members, {"entities": {}}, policy, config(satellite_trigger_step=0))
    coordinator.submit(11, observation(1, 11, 21002))
    coordinator.submit(12, observation(1, 12, 21002))
    first = coordinator.actions_for(11, 0, 0)
    second = coordinator.actions_for(12, 0, 0)
    assert len(policy.encode_calls) == 2
    assert all(int(row[1]) == 11 for row in first)
    assert all(int(row[1]) == 12 for row in second)
    assert sum(row[0] == 3 for row in (*first, *second)) == 1
    coordinator.actions_for(11, 0, 0)
    assert len(policy.encode_calls) == 2


def test_owner_local_hidden_track_is_not_promoted_to_fleet_information():
    policy = Policy()
    coordinator = TeamCoordinator(
        ((1, 11), (2, 12)), {"entities": {}}, policy,
        config(satellite_trigger_step=99),
    )
    coordinator.submit(11, observation(1, 11, 21002, peers=(12,), tracks={9501: hidden(9501)}))
    coordinator.submit(12, observation(1, 12, 21002, peers=(11,)))
    rows11 = coordinator.actions_for(11, 0, 0)
    rows12 = coordinator.actions_for(12, 0, 0)
    assert not any(row[0] == 4 for row in rows11)
    assert not any(row[0] == 4 for row in rows12)


def test_hidden_attack_requires_track_common_to_all_active_l_members():
    policy = Policy()
    coordinator = TeamCoordinator(
        ((1, 11), (2, 12)), {"entities": {}}, policy,
        config(satellite_trigger_step=99),
    )
    tracks = {9501: hidden(9501)}
    coordinator.submit(11, observation(1, 11, 21002, peers=(12,), tracks=tracks))
    coordinator.submit(12, observation(1, 12, 21002, peers=(11,), tracks=tracks))

    rows11 = coordinator.actions_for(11, 0, 0)
    rows12 = coordinator.actions_for(12, 0, 0)

    assert any(row[0] == 4 and row[2] == 9501 for row in (*rows11, *rows12))


def test_hm_goal_refresh_is_owner_local_and_routes_action4_to_owner():
    public = {"entities": {9401: {"type": 9400, "position": {"lon": 8, "lat": 1}}}}
    policy = Policy()
    coordinator = TeamCoordinator(((1, 11), (2, 12)), public, policy, config())
    refreshed = {9401: {"entity_id": 9401, "entity_type": 9400, "lla": {"x": 7, "y": 3}, "time": 1}}
    coordinator.submit(11, observation(1, 11, 21000, tracks=refreshed))
    coordinator.submit(12, observation(1, 12, 21000))
    rows11 = coordinator.actions_for(11, 0, 0)
    rows12 = coordinator.actions_for(12, 0, 0)
    assert all(row[1] == 11 for row in rows11)
    assert all(row[1] == 12 for row in rows12)
    goal_by_owner = {owner: goal for _, owner, goal in policy.encode_calls}
    assert goal_by_owner[11] == {"lon": 7.0, "lat": 3.0}
    assert goal_by_owner[12] != {"lon": 7.0, "lat": 3.0}


def test_shared_reset_is_idempotent_across_members():
    policy = Policy()
    coordinator = TeamCoordinator(((1, 11), (2, 12)), {"entities": {}}, policy, config())
    coordinator.reset_member(11)
    coordinator.reset_member(11)
    coordinator.reset_member(12)
    assert policy.reset_calls == 1


def test_missing_member_on_later_step_is_treated_as_inactive():
    policy = Policy()
    coordinator = TeamCoordinator(
        ((1, 11), (2, 12)), {"entities": {}}, policy,
        config(satellite_trigger_step=99),
    )
    coordinator.submit(11, observation(1, 11, 21002))
    coordinator.submit(12, observation(1, 12, 21002))
    coordinator.actions_for(11, 0, 0)

    coordinator.submit(11, observation(2, 11, 21002))
    rows = coordinator.actions_for(11, 0, 0)

    assert any(row[0] == 0 for row in rows)
    assert coordinator._actions[12] == ()


def test_locked_l_allocation_survives_losses_below_target_count():
    policy = Policy()
    members = tuple((index, 100 + index) for index in range(10))
    coordinator = TeamCoordinator(
        members, {"entities": {}}, policy,
        config(satellite_trigger_step=99, expected_target_count=10, package_count=10),
    )
    tracks = {9500 + index: hidden(9500 + index) for index in range(10)}
    for _, owner in members:
        coordinator.submit(owner, observation(1, owner, 21002, tracks=tracks))
    coordinator.actions_for(100, 0, 0)

    for owner in range(100, 104):
        coordinator.submit(owner, observation(2, owner, 21002, tracks=tracks))
    rows = coordinator.actions_for(100, 0, 0)

    assert any(row[0] == 0 for row in rows)
    assert all(coordinator._actions[owner] == () for owner in range(104, 110))
