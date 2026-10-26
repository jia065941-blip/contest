import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from dynamic_retarget_skill import DynamicRetargetConfig
from hm_public_strike_packages_skill import PublicStrikeConfig
from l_satellite_strike_packages_skill import SatelliteStrikeConfig
from staggered_deployment_skill import StaggeredDeploymentConfig
from terminal_guidance_override_skill import TerminalGuidanceConfig

from simple_upper_orchestrator_skill.config import OrchestratorConfig
from simple_upper_orchestrator_skill.coordinator import TeamCoordinator
from simple_upper_orchestrator_skill.deploy_agent import PreMissionPlan


class Point:
    def __init__(self, entity_id, entity_type, band):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.band = band


class Plan:
    def __init__(self, points):
        self.points = points


class Policy:
    def __init__(self, action=1):
        self.encode_calls = []
        self.reset_calls = 0
        self.action = action

    def encode_motion_observation(self, observation, **kwargs):
        self.encode_calls.append((observation["step"], kwargs["entity_id"], kwargs["goal_position"]))
        return np.zeros(3)

    def prepare_action_mask(self, observation, mask):
        return mask

    def select_action(self, observation, mask):
        return self.action

    def reset_episode(self):
        self.reset_calls += 1


def config(
    *, dynamic=None, hm=None, dynamic_enabled=True, audit_path=None,
    track_max_age=5, formation_audit_interval=25, trace_interval=0,
    terminal_history=20, terminal=None, terminal_enabled=True,
    seconds_per_step=1.0, velocity_window_steps=5, **l_values
):
    return OrchestratorConfig(
        Path("."), Path("checkpoint"), 100,
        deployment=StaggeredDeploymentConfig(deployment_gap_km=1),
        l_strike=SatelliteStrikeConfig(**l_values),
        hm_strike=hm or PublicStrikeConfig(
            allocation_mode="kill_probability",
            min_h_per_mother=2,
            min_m_per_mother=1,
            min_h_per_sentry=0,
            max_h_per_mother=3,
        ),
        dynamic_retarget=dynamic or DynamicRetargetConfig(),
        terminal_guidance=terminal or TerminalGuidanceConfig(),
        retarget_track_max_age_steps=track_max_age,
        dynamic_retarget_enabled=dynamic_enabled,
        terminal_guidance_enabled=terminal_enabled,
        audit_path=audit_path,
        formation_audit_interval_steps=formation_audit_interval,
        missile_trace_interval_steps=trace_interval,
        missile_terminal_history=terminal_history,
        seconds_per_step=seconds_per_step,
        velocity_window_steps=velocity_window_steps,
    )


def observation(
    step, owner, kind, *, peers=(), tracks=None, position=None, stage=2,
    comm_as_list=False, velocity=None, vel_lla=None, health=100,
):
    own = {
        "entity_id": owner, "type": kind, "health": health, "stage": stage,
        "position": position or {"lon": float(owner), "lat": 0.0},
        "detectInfo": tracks or {},
        "commRangeInfo": list(peers) if comm_as_list else {peer: {} for peer in peers},
    }
    if velocity is not None:
        own["velocity"] = velocity
    if vel_lla is not None:
        own["vel_lla"] = vel_lla
    return {
        "step": step,
        "self": own,
        "_orchestrator_launch_step": 0,
        "_orchestrator_maneuver_state": 0,
    }


def hidden(target_id, lon=9.0):
    return {"entity_id": target_id, "entity_type": 9500, "lla": {"x": lon, "y": 2.0}, "time": 1}


def dynamic_public_targets():
    return {"entities": {
        9401: {"type": 9400, "position": {"lon": 0.0, "lat": 0.0}},
        9601: {"type": 9600, "position": {"lon": 0.1, "lat": 0.0}},
    }}


def public_track(target_id, target_type, lon, step, lat=0.0):
    return {
        "entity_id": target_id,
        "entity_type": target_type,
        "lla": {"x": lon, "y": lat},
        "time": step,
    }


def interceptor(entity_id, lon, lat=0.0):
    return {
        "entity_id": entity_id,
        "entity_type": 24000,
        "lla": {"x": lon, "y": lat},
    }


def register_plan(coordinator, entity_types, assignments=None):
    if assignments is None:
        public_ids = tuple(sorted(coordinator._public_by_id))
        assignments = {
            owner: public_ids[0]
            for owner, kind in entity_types.items()
            if kind in (21000, 21001) and public_ids
        }
    coordinator.set_mission_plan(PreMissionPlan(
        assignments,
        entity_types,
        {
            owner: {"entity_id": owner, "type": kind}
            for owner, kind in entity_types.items()
        },
        tuple(sorted(
            owner for owner, kind in entity_types.items() if kind == 21002
        )),
    ))
    return coordinator


def terminal_h_scenario(
    *, same_group=True, m_stage=2, m_lon=-0.2, audit_path=None, dynamic=None,
    candidate_lon=0.1, include_fresh_candidate=True, velocity=(1.0, 0.0),
):
    policy = Policy()
    coordinator = TeamCoordinator(
        ((1, 11), (2, 12), (3, 13)), dynamic_public_targets(), policy,
        config(audit_path=audit_path, dynamic=dynamic),
    )
    register_plan(coordinator, {11: 21000, 12: 21000, 13: 21001})
    old_track = {9401: public_track(9401, 9400, 0.0, 1)}
    h_peers = (12, 13) if same_group else (12,)
    coordinator.submit(11, observation(
        1, 11, 21000, peers=h_peers, tracks=old_track,
        position={"lon": 0.0, "lat": 0.0}, stage=3,
    ))
    h_peers = (11, 13) if same_group else (11,)
    coordinator.submit(12, observation(
        1, 12, 21000, peers=h_peers, tracks=old_track,
        position={"lon": 0.0, "lat": 0.0}, stage=3,
    ))
    m_tracks = {
        **old_track,
        9601: public_track(9601, 9600, candidate_lon, 1),
    }
    coordinator.submit(13, observation(
        1, 13, 21001, peers=(11, 12) if same_group else (),
        tracks=m_tracks, position={"lon": m_lon, "lat": 0.0}, stage=m_stage,
        velocity=(
            None if velocity is None else {"x": velocity[0], "y": velocity[1]}
        ),
    ))
    coordinator.actions_for(11, 0, 0)
    coordinator.actions_for(12, 0, 0)
    coordinator.actions_for(13, 0, 0)
    current_tracks = {9401: public_track(9401, 9400, 0.0, 2)}
    if include_fresh_candidate:
        current_tracks[9601] = public_track(9601, 9600, candidate_lon, 2)
    coordinator.submit(11, observation(
        2, 11, 21000, peers=h_peers, tracks=old_track,
        position={"lon": 0.0, "lat": 0.0}, stage=3, health=0,
    ))
    coordinator.submit(12, observation(
        2, 12, 21000,
        peers=(11, 13) if same_group else (11,), tracks=old_track,
        position={"lon": 0.0, "lat": 0.0}, stage=3, health=0,
    ))
    coordinator.submit(13, observation(
        2, 13, 21001, tracks=current_tracks,
        position={"lon": m_lon, "lat": 0.0}, stage=m_stage,
        velocity=(
            None if velocity is None else {"x": velocity[0], "y": velocity[1]}
        ),
    ))
    rows = coordinator.actions_for(13, 0, 0)
    return coordinator, policy, rows


def audit_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_once_per_step_owner_routing_and_satellite_singleton():
    policy = Policy()
    members = ((1, 11), (2, 12))
    coordinator = TeamCoordinator(members, {"entities": {}}, policy, config(satellite_trigger_step=0))
    register_plan(coordinator, {11: 21002, 12: 21002})
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


def test_terminal_guidance_overrides_fresh_safe_stage4_m_to_straight():
    policy = Policy(action=2)
    coordinator = TeamCoordinator(
        ((1, 13),), dynamic_public_targets(), policy,
        config(dynamic_enabled=False),
    )
    register_plan(coordinator, {13: 21001})
    coordinator.submit(13, observation(
        1,
        13,
        21001,
        tracks={9401: public_track(9401, 9400, 0.0, 1)},
        position={"lon": 0.005, "lat": 0.0},
        stage=4,
        velocity={"speed": 120.0, "up": 0.0, "heading": 270.0},
    ))

    rows = coordinator.actions_for(13, 0, 0)

    assert len(policy.encode_calls) == 1
    assert next(row for row in rows if row[0] == 0) == (0, 13, 0.0, 0)
    assert coordinator._motion_guidance[13] == {
        "step": 1,
        "baseline_action": 2,
        "final_action": 1,
        "override_reason": "predicted_miss_straight",
        "predicted_miss_distance": pytest.approx(0.0, abs=1e-6),
        "control_state": "STRAIGHT",
        "no_improvement_count": 0,
        "heading_error": pytest.approx(0.0, abs=1e-6),
        "velocity_source": "observed",
    }


def test_terminal_guidance_threat_and_stale_track_pass_baseline_through():
    policy = Policy(action=2)
    coordinator = TeamCoordinator(
        ((1, 13),), dynamic_public_targets(), policy,
        config(dynamic_enabled=False),
    )
    register_plan(coordinator, {13: 21001})
    coordinator.submit(13, observation(
        1,
        13,
        21001,
        tracks={
            9401: public_track(9401, 9400, 0.0, 1),
            24001: interceptor(24001, 0.011),
        },
        position={"lon": 0.005, "lat": 0.0},
        stage=4,
        velocity={"speed": 120.0, "up": 0.0, "heading": 270.0},
    ))

    threatened = coordinator.actions_for(13, 0, 0)

    assert next(row for row in threatened if row[0] == 0) == (0, 13, 0.0, 0)
    assert coordinator._motion_guidance[13]["override_reason"] == (
        "predicted_miss_straight"
    )

    coordinator.submit(13, observation(
        7,
        13,
        21001,
        tracks={9401: public_track(9401, 9400, 0.0, 1)},
        position={"lon": 0.01, "lat": 0.0},
        stage=4,
        velocity={"speed": 120.0, "up": 0.0, "heading": 270.0},
    ))
    stale = coordinator.actions_for(13, 0, 0)

    assert next(row for row in stale if row[0] == 0) == (0, 13, 0.0, 0)
    assert coordinator._motion_guidance[13]["override_reason"] == "stale_grace"
    assert coordinator._motion_guidance[13]["control_state"] == "STALE_GRACE"


@pytest.mark.parametrize("kind", (21000, 21002))
def test_terminal_guidance_leaves_h_and_l_policy_categories_unchanged(kind):
    policy = Policy(action=0)
    public = dynamic_public_targets() if kind == 21000 else {"entities": {}}
    coordinator = TeamCoordinator(
        ((1, 11),), public, policy,
        config(
            dynamic_enabled=False,
            satellite_trigger_step=99,
            expected_target_count=1 if kind == 21002 else None,
        ),
    )
    register_plan(coordinator, {11: kind})
    tracks = (
        {9401: public_track(9401, 9400, 0.0, 1)}
        if kind == 21000
        else {9501: hidden(9501, lon=0.0)}
    )
    coordinator.submit(11, observation(
        1,
        11,
        kind,
        tracks=tracks,
        position={"lon": 0.01, "lat": 0.0},
        stage=4,
    ))

    rows = coordinator.actions_for(11, 0, 0)

    assert next(row for row in rows if row[0] == 0) == (0, 11, -1.0, 0)
    assert coordinator._motion_guidance[11]["override_reason"] == (
        "missile_type_passthrough"
    )


def test_terminal_guidance_state_and_actions_are_owner_isolated():
    policy = Policy(action=2)
    coordinator = TeamCoordinator(
        ((1, 13), (2, 14)), dynamic_public_targets(), policy,
        config(dynamic_enabled=False),
    )
    register_plan(coordinator, {13: 21001, 14: 21001})
    track = {9401: public_track(9401, 9400, 0.0, 1)}
    coordinator.submit(13, observation(
        1, 13, 21001, tracks=track,
        position={"lon": 0.005, "lat": 0.0}, stage=4,
        velocity={"speed": 120.0, "up": 0.0, "heading": 270.0},
    ))
    coordinator.submit(14, observation(
        1, 14, 21001, tracks=track,
        position={"lon": 0.03, "lat": 0.0}, stage=4,
        velocity={"speed": 120.0, "up": 0.0, "heading": 270.0},
    ))

    rows13 = coordinator.actions_for(13, 0, 0)
    rows14 = coordinator.actions_for(14, 0, 0)

    assert next(row for row in rows13 if row[0] == 0) == (0, 13, 0.0, 0)
    assert next(row for row in rows14 if row[0] == 0) == (0, 14, 1.0, 0)
    assert all(int(row[1]) == 13 for row in rows13)
    assert all(int(row[1]) == 14 for row in rows14)

    track = {9401: public_track(9401, 9400, 0.0, 2)}
    for owner in (13, 14):
        coordinator.submit(owner, observation(
            2, owner, 21001, tracks=track,
            position={"lon": 0.016 if owner == 13 else 0.04, "lat": 0.0},
            stage=4,
            velocity={"speed": 120.0, "up": 0.0, "heading": 270.0},
        ))
    latched13 = coordinator.actions_for(13, 0, 0)
    unlatched14 = coordinator.actions_for(14, 0, 0)

    assert next(row for row in latched13 if row[0] == 0) == (0, 13, 0.0, 0)
    assert next(row for row in unlatched14 if row[0] == 0) == (0, 14, 1.0, 0)


def test_terminal_guidance_fail_closed_without_assigned_goal_or_legal_position():
    policy = Policy(action=2)
    without_goal = TeamCoordinator(
        ((1, 13),), dynamic_public_targets(), policy,
        config(dynamic_enabled=False),
    )
    without_goal.submit(13, observation(
        1, 13, 21001,
        tracks={9401: public_track(9401, 9400, 0.0, 1)},
        position={"lon": 0.01, "lat": 0.0}, stage=4,
    ))

    no_goal_rows = without_goal.actions_for(13, 0, 0)

    assert next(row for row in no_goal_rows if row[0] == 0) == (0, 13, 1.0, 0)
    assert without_goal._motion_guidance[13]["override_reason"] == (
        "missing_assigned_goal"
    )

    invalid_position = TeamCoordinator(
        ((1, 13),), dynamic_public_targets(), Policy(action=2),
        config(dynamic_enabled=False),
    )
    register_plan(invalid_position, {13: 21001})
    invalid_position.submit(13, observation(
        1, 13, 21001,
        tracks={9401: public_track(9401, 9400, 0.0, 1)},
        position={"lon": 999.0, "lat": 999.0}, stage=4,
    ))

    invalid_rows = invalid_position.actions_for(13, 0, 0)

    assert next(row for row in invalid_rows if row[0] == 0) == (0, 13, 1.0, 0)
    assert invalid_position._motion_guidance[13]["override_reason"] == (
        "missing_owner_position"
    )


def test_terminal_guidance_audit_trace_and_reset_latch(tmp_path):
    path = tmp_path / "retarget_audit.jsonl"
    coordinator = TeamCoordinator(
        ((1, 13),), dynamic_public_targets(), Policy(action=2),
        config(
            audit_path=path,
            dynamic_enabled=False,
            trace_interval=1,
        ),
    )
    register_plan(coordinator, {13: 21001})
    for step in (1, 2):
        coordinator.submit(13, observation(
            step,
            13,
            21001,
            tracks={9401: public_track(9401, 9400, 0.0, step)},
            position={"lon": 0.005, "lat": 0.0},
            stage=4,
            velocity={"speed": 120.0, "up": 0.0, "heading": 270.0},
        ))
        coordinator.actions_for(13, 0, 0)

    records = audit_rows(path)
    guidance = [row for row in records if row["event"] == "terminal_guidance"]
    traces = [row for row in records if row["event"] == "missile_trace"]
    assert len(guidance) == 2
    assert guidance[0] == {
        "event": "terminal_guidance",
        "step": 1,
        "owner": 13,
        "type": 21001,
        "baseline": 2,
        "final": 1,
        "stage": 4,
        "distance_m": pytest.approx(555.975, rel=1e-4),
        "track_source": "detect_fresh",
        "track_age": 0.0,
        "nearest_interceptor_distance_m": None,
        "threat": False,
        "latched": True,
        "overridden": True,
        "reason": "predicted_miss_straight",
        "velocity_source": "observed",
        "velocity_east": pytest.approx(-120.0),
        "velocity_north": pytest.approx(0.0, abs=1e-10),
        "mode": "straight",
        "heading_error_degrees": pytest.approx(0.0, abs=1e-6),
        "heading_error": pytest.approx(0.0, abs=1e-6),
        "predicted_miss_distance": pytest.approx(0.0, abs=1e-6),
        "control_state": "STRAIGHT",
        "no_improvement_count": 0,
    }
    assert traces[-1]["baseline_action"] == 2
    assert traces[-1]["final_action"] == 1
    assert traces[-1]["override_reason"] == "predicted_miss_straight"
    assert traces[-1]["velocity_source"] == "observed"
    assert traces[-1]["velocity_east"] == pytest.approx(-120.0)
    assert traces[-1]["velocity_north"] == pytest.approx(0.0, abs=1e-10)
    assert traces[-1]["predicted_miss_distance"] == pytest.approx(
        0.0, abs=1e-6
    )
    assert traces[-1]["control_state"] == "STRAIGHT"
    assert traces[-1]["no_improvement_count"] == 0
    assert traces[-1]["heading_error"] == pytest.approx(0.0, abs=1e-6)

    coordinator.reset_member(13)
    coordinator.submit(13, observation(
        1,
        13,
        21001,
        tracks={9401: public_track(9401, 9400, 0.0, 1)},
        position={"lon": 0.04, "lat": 0.0},
        stage=4,
        velocity={"speed": 120.0, "up": 0.0, "heading": 270.0},
    ))

    after_reset = coordinator.actions_for(13, 0, 0)

    assert next(row for row in after_reset if row[0] == 0) == (0, 13, 1.0, 0)
    assert coordinator._motion_guidance[13]["override_reason"] == (
        "outside_capture_distance"
    )


def test_terminal_guidance_state_transitions_are_audited_in_guidance_and_trace(
    tmp_path,
):
    path = tmp_path / "retarget_audit.jsonl"
    coordinator = TeamCoordinator(
        ((1, 13),), dynamic_public_targets(), Policy(action=1),
        config(audit_path=path, dynamic_enabled=False, trace_interval=1),
    )
    register_plan(coordinator, {13: 21001})

    for step, heading in enumerate((240.0, 245.0, 250.0, 255.0), start=1):
        coordinator.submit(13, observation(
            step,
            13,
            21001,
            tracks={9401: public_track(9401, 9400, 0.0, step)},
            position={"lon": 0.01, "lat": 0.0},
            stage=4,
            velocity={"speed": 120.0, "heading": heading},
        ))
        coordinator.actions_for(13, 0, 0)

    records = audit_rows(path)
    guidance = [
        row for row in records if row["event"] == "terminal_guidance"
    ]
    traces = [row for row in records if row["event"] == "missile_trace"]
    assert [row["control_state"] for row in guidance] == [
        "TURN_PULSE",
        "STRAIGHT_RECOVERY",
        "TURN_PULSE",
    ]
    assert [row["step"] for row in guidance] == [1, 2, 4]
    assert [row["control_state"] for row in traces] == [
        "TURN_PULSE",
        "STRAIGHT_RECOVERY",
        "STRAIGHT_RECOVERY",
        "TURN_PULSE",
    ]
    assert all(row["predicted_miss_distance"] > 80.0 for row in traces)
    assert all(abs(row["heading_error"]) > 10.0 for row in traces)
    assert all(row["velocity_source"] == "observed" for row in traces)
    assert all(row["no_improvement_count"] == 0 for row in traces)


def test_terminal_guidance_estimates_velocity_after_first_fail_closed_step(tmp_path):
    path = tmp_path / "retarget_audit.jsonl"
    coordinator = TeamCoordinator(
        ((1, 13),), dynamic_public_targets(), Policy(action=2),
        config(audit_path=path, dynamic_enabled=False, trace_interval=1),
    )
    register_plan(coordinator, {13: 21001})
    coordinator.submit(13, observation(
        1, 13, 21001,
        tracks={9401: public_track(9401, 9400, 0.0, 1)},
        position={"lon": 0.01, "lat": 0.0}, stage=4,
    ))
    first_rows = coordinator.actions_for(13, 0, 0)
    coordinator.submit(13, observation(
        2, 13, 21001,
        tracks={9401: public_track(9401, 9400, 0.0, 2)},
        position={"lon": 0.005, "lat": 0.0}, stage=4,
    ))
    second_rows = coordinator.actions_for(13, 0, 0)

    assert next(row for row in first_rows if row[0] == 0) == (0, 13, 1.0, 0)
    assert next(row for row in second_rows if row[0] == 0) == (0, 13, 0.0, 0)
    guidance = [
        row for row in audit_rows(path) if row["event"] == "terminal_guidance"
    ]
    assert guidance[0]["reason"] == "missing_owner_velocity"
    assert guidance[0]["velocity_source"] == "none"
    assert guidance[0]["velocity_east"] is None
    assert guidance[0]["velocity_north"] is None
    assert guidance[-1]["velocity_source"] == "estimated_window"
    assert guidance[-1]["velocity_east"] == pytest.approx(-556.597, rel=1e-4)
    assert guidance[-1]["velocity_north"] == pytest.approx(0.0)
    traces = [row for row in audit_rows(path) if row["event"] == "missile_trace"]
    assert traces[0]["velocity_source"] == "none"
    assert traces[-1]["velocity_source"] == "estimated_window"
    assert traces[-1]["velocity_east"] == pytest.approx(-556.597, rel=1e-4)


def test_terminal_guidance_prefers_observed_ecf_over_heading_and_estimate(tmp_path):
    path = tmp_path / "retarget_audit.jsonl"
    coordinator = TeamCoordinator(
        ((1, 13),), dynamic_public_targets(), Policy(action=2),
        config(audit_path=path, dynamic_enabled=False),
    )
    register_plan(coordinator, {13: 21001})
    coordinator.submit(13, observation(
        1, 13, 21001,
        tracks={9401: public_track(9401, 9400, 0.0, 1)},
        position={"lon": 0.006, "lat": 0.0}, stage=4,
    ))
    coordinator.actions_for(13, 0, 0)
    coordinator.submit(13, observation(
        2, 13, 21001,
        tracks={9401: public_track(9401, 9400, 0.0, 2)},
        position={"lon": 0.005, "lat": 0.0}, stage=4,
        velocity={
            "x": 0.0, "y": -120.0, "z": 0.0,
            "speed": 120.0, "heading": 90.0,
        },
    ))
    coordinator.actions_for(13, 0, 0)

    guidance = [
        row for row in audit_rows(path) if row["event"] == "terminal_guidance"
    ][-1]
    assert guidance["velocity_source"] == "observed"
    assert guidance["velocity_east"] == pytest.approx(-120.0, rel=1e-6)
    assert guidance["velocity_north"] == pytest.approx(0.0, abs=1e-10)


def test_position_velocity_history_is_owner_local():
    coordinator = TeamCoordinator(
        ((1, 13), (2, 14)), dynamic_public_targets(), Policy(),
        config(dynamic_enabled=False),
    )
    for owner in (13, 14):
        coordinator.submit(owner, observation(
            1, owner, 21001, position={"lon": 0.0, "lat": 0.0},
        ))
    coordinator.submit(13, observation(
        2, 13, 21001, position={"lon": 0.001, "lat": 0.0},
    ))
    state = coordinator._owner_velocity_history[13]
    expected = (state.last_valid_east, state.last_valid_north)
    coordinator.submit(14, observation(
        2, 14, 21001, position={"lon": 90.0, "lat": 45.0},
    ))

    unchanged = coordinator._owner_velocity_history[13]
    assert (unchanged.last_valid_east, unchanged.last_valid_north) == expected
    assert unchanged.last_valid_east == pytest.approx(111.31949, rel=1e-5)
    assert unchanged.last_valid_north == pytest.approx(0.0)
    assert unchanged.source == "estimated_window"
    assert unchanged.positions.maxlen == 5


def test_position_velocity_uses_oldest_sample_in_bounded_five_step_window():
    coordinator = TeamCoordinator(
        ((1, 13),), dynamic_public_targets(), Policy(),
        config(dynamic_enabled=False, velocity_window_steps=5),
    )
    longitudes = (0.0, 0.002, 0.001, 0.004, 0.004)
    for step, longitude in enumerate(longitudes, start=1):
        coordinator.submit(13, observation(
            step, 13, 21001, position={"lon": longitude, "lat": 0.0},
        ))
        state = coordinator._owner_velocity_history[13]
        if step == 1:
            assert state.source == "none"
            assert state.last_valid_east is None

    state = coordinator._owner_velocity_history[13]
    assert len(state.positions) == state.positions.maxlen == 5
    assert state.positions[0].step == pytest.approx(1.0)
    assert state.last_valid_east == pytest.approx(111.31949, rel=1e-5)
    assert state.last_valid_north == pytest.approx(0.0)
    assert state.source == "estimated_window"

    coordinator.submit(13, observation(
        6, 13, 21001, position={"lon": 0.005, "lat": 0.0},
    ))
    shifted = coordinator._owner_velocity_history[13]
    assert len(shifted.positions) == shifted.positions.maxlen == 5
    assert shifted.positions[0].step == pytest.approx(2.0)
    assert shifted.last_valid_east == pytest.approx(83.48962, rel=1e-5)


def test_position_velocity_wraps_longitude_and_uses_wgs84_latitude():
    coordinator = TeamCoordinator(
        ((1, 13), (2, 14)), dynamic_public_targets(), Policy(),
        config(dynamic_enabled=False, seconds_per_step=2.0),
    )
    coordinator.submit(13, observation(
        1, 13, 21001, position={"lon": 179.999, "lat": 0.0},
    ))
    coordinator.submit(13, observation(
        2, 13, 21001, position={"lon": -179.999, "lat": 0.0},
    ))
    coordinator.submit(14, observation(
        1, 14, 21001, position={"lon": 10.0, "lat": 60.0},
    ))
    coordinator.submit(14, observation(
        2, 14, 21001, position={"lon": 10.0, "lat": 60.001},
    ))

    wrapped = coordinator._owner_velocity_history[13]
    latitude = coordinator._owner_velocity_history[14]
    assert wrapped.last_valid_east == pytest.approx(111.31949, rel=1e-5)
    assert wrapped.last_valid_north == pytest.approx(0.0)
    assert latitude.last_valid_east == pytest.approx(0.0)
    assert latitude.last_valid_north == pytest.approx(55.706, rel=1e-4)


def test_position_velocity_filters_low_speed_and_retains_on_outlier():
    coordinator = TeamCoordinator(
        ((1, 13), (2, 14)), dynamic_public_targets(), Policy(),
        config(dynamic_enabled=False),
    )
    coordinator.submit(13, observation(
        1, 13, 21001, position={"lon": 0.0, "lat": 0.0},
    ))
    coordinator.submit(13, observation(
        2, 13, 21001, position={"lon": 0.001, "lat": 0.0},
    ))
    before = coordinator._owner_velocity_history[13].last_valid_east
    coordinator.submit(13, observation(
        3, 13, 21001, position={"lon": 1.0, "lat": 0.0},
    ))
    retained = coordinator._owner_velocity_history[13]
    assert retained.last_valid_east == before
    assert retained.source == "estimated_window"
    assert retained.previous_lon == pytest.approx(1.0)
    assert len(retained.positions) == 1

    coordinator.submit(14, observation(
        1, 14, 21001, position={"lon": 0.0, "lat": 0.0},
    ))
    coordinator.submit(14, observation(
        2, 14, 21001, position={"lon": 0.000001, "lat": 0.0},
    ))
    low = coordinator._owner_velocity_history[14]
    assert low.source == "none"
    assert low.last_valid_east is None
    assert low.last_valid_north is None


def test_position_velocity_rejects_nonfinite_and_nonforward_samples():
    coordinator = TeamCoordinator(
        ((1, 13),), dynamic_public_targets(), Policy(),
        config(dynamic_enabled=False),
    )
    coordinator.submit(13, observation(
        1, 13, 21001, position={"lon": 0.0, "lat": 0.0},
    ))
    coordinator.submit(13, observation(
        2, 13, 21001, position={"lon": 0.001, "lat": 0.0},
    ))
    before = coordinator._owner_velocity_history[13]
    expected = (before.previous_step, before.last_valid_east)
    coordinator.submit(13, observation(
        2, 13, 21001, position={"lon": 0.5, "lat": 0.0},
    ))
    duplicate = coordinator._owner_velocity_history[13]
    assert (duplicate.previous_step, duplicate.last_valid_east) == expected

    coordinator.submit(13, observation(
        3, 13, 21001, position={"lon": float("nan"), "lat": 0.0},
    ))
    invalid = coordinator._owner_velocity_history[13]
    assert invalid.previous_step == pytest.approx(2.0)
    assert invalid.last_valid_east == pytest.approx(expected[1])
    assert invalid.source == "estimated_window"


def test_position_velocity_history_clears_on_terminal_and_reset():
    coordinator = TeamCoordinator(
        ((1, 13),), dynamic_public_targets(), Policy(),
        config(dynamic_enabled=False),
    )
    register_plan(coordinator, {13: 21001})
    coordinator.submit(13, observation(
        1, 13, 21001, position={"lon": 0.0, "lat": 0.0},
    ))
    coordinator.actions_for(13, 0, 0)
    coordinator.submit(13, observation(
        2, 13, 21001, position={"lon": 0.001, "lat": 0.0}, health=0,
    ))
    coordinator.actions_for(13, 0, 0)
    terminal = coordinator._owner_velocity_history[13]
    assert terminal.previous_step is None
    assert list(terminal.positions) == []
    assert terminal.last_valid_east is None
    assert terminal.source == "none"

    coordinator.submit(13, observation(
        3, 13, 21001, position={"lon": 0.002, "lat": 0.0},
    ))
    assert coordinator._owner_velocity_history[13].previous_step == 3.0
    coordinator.reset_member(13)
    reset = coordinator._owner_velocity_history[13]
    assert reset.previous_step is None
    assert list(reset.positions) == []
    assert reset.positions.maxlen == 5
    assert reset.previous_lon is None
    assert reset.previous_lat is None
    assert reset.last_valid_east is None
    assert reset.last_valid_north is None
    assert reset.source == "none"


def test_official_preflight_stage_minus_one_is_normalized() -> None:
    coordinator = TeamCoordinator(
        ((1, 11),), dynamic_public_targets(), Policy(), config(),
    )
    register_plan(coordinator, {11: 21000})
    coordinator.submit(11, observation(1, 11, 21000, stage=-1))

    coordinator.actions_for(11, 0, 0)

    assert coordinator._hm_progress[11].stage == 0


def test_hm_without_deployment_preplan_fails_closed() -> None:
    coordinator = TeamCoordinator(
        ((1, 11),), dynamic_public_targets(), Policy(),
        config(dynamic_enabled=False),
    )
    coordinator.submit(11, observation(
        1, 11, 21000,
        tracks={9401: public_track(9401, 9400, 9.0, 1)},
    ))

    rows = coordinator.actions_for(11, 0, 0)

    assert not any(row[0] == 4 for row in rows)
    assert coordinator._assignments == {}
    assert coordinator._hm_progress == {}


def test_hm_first_action_uses_preplan_and_ignores_out_group_observation() -> None:
    coordinator = TeamCoordinator(
        ((1, 11), (2, 12)), dynamic_public_targets(), Policy(),
        config(dynamic_enabled=False),
    )
    register_plan(
        coordinator,
        {11: 21000, 12: 21001},
        {11: 9601, 12: 9401},
    )
    coordinator.submit(11, observation(1, 11, 21000))
    coordinator.submit(12, observation(
        1, 12, 21001,
        tracks={9601: public_track(9601, 9600, 999.0, 1)},
        position={"lon": 999.0, "lat": 999.0}, health=0,
    ))

    rows = coordinator.actions_for(11, 0, 0)

    assert [row for row in rows if row[0] == 4] == [(4, 11, 9601, 0)]
    assert coordinator._assignments[11] == 9601
    assert coordinator._goals[11] == {"lon": 0.1, "lat": 0.0}


@pytest.mark.parametrize("value", (-1, float("inf"), float("nan"), True))
def test_retarget_track_max_age_must_be_nonnegative_and_finite(value):
    with pytest.raises(ValueError, match="nonnegative and finite"):
        config(track_max_age=value)


@pytest.mark.parametrize("value", (0, -1, 1.5, True))
def test_formation_audit_interval_must_be_a_positive_integer(value):
    with pytest.raises(ValueError, match="positive integer"):
        config(formation_audit_interval=value)


@pytest.mark.parametrize("value", (-1, 1.5, True))
def test_missile_trace_interval_must_be_a_nonnegative_integer(value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        config(trace_interval=value)


@pytest.mark.parametrize("value", (0, -1, 1.5, True))
def test_missile_terminal_history_must_be_a_positive_integer(value):
    with pytest.raises(ValueError, match="positive integer"):
        config(terminal_history=value)


@pytest.mark.parametrize("value", (0, -1, float("inf"), float("nan"), True))
def test_seconds_per_step_must_be_positive_and_finite(value):
    with pytest.raises(ValueError, match="positive and finite"):
        config(seconds_per_step=value)


@pytest.mark.parametrize("value", (1, 0, -1, 2.5, True))
def test_velocity_window_steps_must_be_an_integer_at_least_two(value):
    with pytest.raises(ValueError, match="integer at least 2"):
        config(velocity_window_steps=value)


def test_deployment_plan_sets_l_wave_for_distance_allocator():
    coordinator = TeamCoordinator(
        ((1, 11), (2, 12)), {"entities": {}}, Policy(),
        config(satellite_trigger_step=99),
    )
    register_plan(coordinator, {11: 21002, 12: 21002})
    coordinator.set_deployment_plan(Plan((
        Point(11, 21002, "front"), Point(12, 21002, "second"),
    )))

    assert coordinator._deployment_waves == {11: 1, 12: 2}


def test_owner_local_hidden_track_is_not_promoted_to_fleet_information():
    policy = Policy()
    coordinator = TeamCoordinator(
        ((1, 11), (2, 12)), {"entities": {}}, policy,
        config(satellite_trigger_step=99),
    )
    register_plan(coordinator, {11: 21002, 12: 21002})
    coordinator.submit(11, observation(1, 11, 21002, peers=(12,), tracks={9501: hidden(9501)}))
    coordinator.submit(12, observation(1, 12, 21002, peers=(11,)))
    rows11 = coordinator.actions_for(11, 0, 0)
    rows12 = coordinator.actions_for(12, 0, 0)
    assert any(row[0] == 4 and row[2] == 9501 for row in rows11)
    assert not any(row[0] == 4 for row in rows12)


def test_hidden_attack_requires_only_the_owners_current_track():
    policy = Policy()
    coordinator = TeamCoordinator(
        ((1, 11), (2, 12)), {"entities": {}}, policy,
        config(satellite_trigger_step=99),
    )
    register_plan(coordinator, {11: 21002, 12: 21002})
    tracks = {9501: hidden(9501)}
    coordinator.submit(11, observation(1, 11, 21002, peers=(12,), tracks=tracks))
    coordinator.submit(12, observation(1, 12, 21002, peers=(11,), tracks=tracks))

    rows11 = coordinator.actions_for(11, 0, 0)
    rows12 = coordinator.actions_for(12, 0, 0)

    assert all(any(row[0] == 4 and row[2] == 9501 for row in rows) for rows in (rows11, rows12))


def test_l_mapping_ignores_another_l_position_and_tracks() -> None:
    def owner_target(other_tracks, other_position):
        coordinator = TeamCoordinator(
            ((1, 11), (2, 12)), {"entities": {}}, Policy(),
            config(satellite_trigger_step=99, expected_target_count=2),
        )
        register_plan(coordinator, {11: 21002, 12: 21002})
        coordinator.submit(11, observation(
            1, 11, 21002,
            tracks={9501: hidden(9501), 9502: hidden(9502)},
            position={"lon": 1.0, "lat": 1.0},
        ))
        coordinator.submit(12, observation(
            1, 12, 21002, tracks=other_tracks, position=other_position,
        ))
        rows = coordinator.actions_for(11, 0, 0)
        return next(int(row[2]) for row in rows if row[0] == 4)

    assert owner_target(
        {9501: hidden(9501), 9503: hidden(9503)},
        {"lon": -100.0, "lat": -50.0},
    ) == owner_target(
        {9510: hidden(9510), 9520: hidden(9520)},
        {"lon": 100.0, "lat": 50.0},
    ) == 9501


def test_l_roster_rank_mapping_is_balanced() -> None:
    members = tuple((index, 100 + index) for index in range(7))
    coordinator = TeamCoordinator(
        members, {"entities": {}}, Policy(),
        config(satellite_trigger_step=99, expected_target_count=3),
    )
    register_plan(coordinator, {owner: 21002 for _, owner in members})
    tracks = {
        target_id: hidden(target_id)
        for target_id in (9501, 9502, 9503)
    }
    for _, owner in members:
        coordinator.submit(owner, observation(1, owner, 21002, tracks=tracks))

    coordinator.actions_for(100, 0, 0)

    assert Counter(coordinator._assignments.values()) == Counter({
        9501: 3,
        9502: 2,
        9503: 2,
    })


def test_hm_goal_refresh_is_owner_local_and_routes_action4_to_owner():
    public = {"entities": {9401: {"type": 9400, "position": {"lon": 8, "lat": 1}}}}
    policy = Policy()
    coordinator = TeamCoordinator(((1, 11), (2, 12)), public, policy, config())
    register_plan(coordinator, {11: 21000, 12: 21000})
    refreshed = {9401: {"entity_id": 9401, "entity_type": 9400, "lla": {"x": 7, "y": 3}, "time": 1}}
    coordinator.submit(11, observation(1, 11, 21000, tracks=refreshed))
    coordinator.submit(12, observation(1, 12, 21000))
    rows11 = coordinator.actions_for(11, 0, 0)
    rows12 = coordinator.actions_for(12, 0, 0)
    assert all(row[1] == 11 for row in rows11)
    assert all(row[1] == 12 for row in rows12)
    goal_by_owner = {owner: goal for _, owner, goal in policy.encode_calls}
    assert goal_by_owner == {
        11: {"lon": 8.0, "lat": 1.0},
        12: {"lon": 8.0, "lat": 1.0},
    }
    coordinator.submit(11, observation(2, 11, 21000, tracks={
        9401: {**refreshed[9401], "time": 2},
    }))
    coordinator.submit(12, observation(2, 12, 21000, position={"lon": 999, "lat": 999}))
    rows11 = coordinator.actions_for(11, 0, 0)
    rows12 = coordinator.actions_for(12, 0, 0)
    assert any(row[0] == 4 and row[2] == 9401 for row in rows11)
    assert not any(row[0] == 4 for row in rows12)
    assert coordinator._goals[11] == {"lon": 7.0, "lat": 3.0}


def test_shared_reset_is_idempotent_across_members():
    policy = Policy()
    coordinator = TeamCoordinator(((1, 11), (2, 12)), {"entities": {}}, policy, config())
    register_plan(coordinator, {11: 21002, 12: 21002})
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
    register_plan(coordinator, {11: 21002, 12: 21002})
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
    register_plan(coordinator, {owner: 21002 for _, owner in members})
    tracks = {9500 + index: hidden(9500 + index) for index in range(10)}
    for _, owner in members:
        coordinator.submit(owner, observation(1, owner, 21002, tracks=tracks))
    coordinator.actions_for(100, 0, 0)

    for owner in range(100, 104):
        coordinator.submit(owner, observation(2, owner, 21002, tracks=tracks))
    rows = coordinator.actions_for(100, 0, 0)

    assert any(row[0] == 0 for row in rows)
    assert all(coordinator._actions[owner] == () for owner in range(104, 110))


def test_terminal_h_evidence_retargets_same_group_m_and_transfers_commitment():
    coordinator, _, rows = terminal_h_scenario()

    belief = coordinator._damage_by_group[(13,)].estimate(9401)
    assert belief.destruction_probability == pytest.approx(0.9604)
    assert coordinator._assignments[13] == 9601
    assert coordinator._damage_by_group[(13,)].estimate(9401).flexible_committed_missiles == ()
    assert coordinator._damage_by_group[(13,)].estimate(9601).flexible_committed_missiles == (13,)
    target_actions = [row for row in rows if row[0] == 4]
    assert target_actions == [(4, 13, 9601, 0)]
    assert all(int(row[1]) == 13 for row in rows)


def test_terminal_public_fallback_near_target_is_forced_to_point_15(tmp_path):
    path = tmp_path / "retarget_audit.jsonl"
    coordinator = TeamCoordinator(
        ((1, 11), (2, 12)),
        {"entities": {9401: {
            "type": 9400, "position": {"lon": 0.0, "lat": 0.0},
        }}},
        Policy(),
        config(audit_path=path, satellite_trigger_step=99),
    )
    register_plan(coordinator, {11: 21000, 12: 21002})
    coordinator.submit(11, observation(
        1, 11, 21000, position={"lon": 0.0, "lat": 0.0}, stage=4,
    ))
    coordinator.submit(12, observation(1, 12, 21002))
    coordinator.actions_for(11, 0, 0)
    coordinator.submit(11, observation(
        2, 11, 21000, position={"lon": 0.0, "lat": 0.0}, stage=4,
        health=0,
    ))
    coordinator.submit(12, observation(2, 12, 21002))
    coordinator.actions_for(12, 0, 0)

    terminal = next(
        record for record in audit_rows(path)
        if record["event"] == "terminal_evidence"
    )
    assert terminal["min_distance_m"] == pytest.approx(0.0)
    assert terminal["source"] == "public"
    assert terminal["age"] is None
    assert terminal["forced_low_confidence"] is True
    assert terminal["delivery_probability"] == pytest.approx(0.15)


def test_terminal_fresh_track_within_500m_keeps_point_98_heuristic(tmp_path):
    path = tmp_path / "retarget_audit.jsonl"
    coordinator = TeamCoordinator(
        ((1, 11), (2, 12)),
        {"entities": {9401: {
            "type": 9400, "position": {"lon": 1.0, "lat": 0.0},
        }}},
        Policy(),
        config(audit_path=path, satellite_trigger_step=99),
    )
    register_plan(coordinator, {11: 21000, 12: 21002})
    coordinator.submit(11, observation(
        1, 11, 21000,
        tracks={9401: public_track(9401, 9400, 0.0, 1)},
        position={"lon": 0.004, "lat": 0.0}, stage=2,
    ))
    coordinator.submit(12, observation(1, 12, 21002))
    coordinator.actions_for(11, 0, 0)
    coordinator.submit(11, observation(
        2, 11, 21000,
        tracks={9401: public_track(9401, 9400, 0.0, 2)},
        position={"lon": 0.004, "lat": 0.0}, stage=2, health=0,
    ))
    coordinator.submit(12, observation(2, 12, 21002))
    coordinator.actions_for(12, 0, 0)

    terminal = next(
        record for record in audit_rows(path)
        if record["event"] == "terminal_evidence"
    )
    assert terminal["min_distance_m"] < 500.0
    assert terminal["source"] == "detect_fresh"
    assert terminal["age"] == pytest.approx(0.0)
    assert terminal["forced_low_confidence"] is False
    assert terminal["delivery_probability"] == pytest.approx(0.98)


def test_terminal_stale_track_near_target_is_forced_to_point_15(tmp_path):
    path = tmp_path / "retarget_audit.jsonl"
    coordinator = TeamCoordinator(
        ((1, 11), (2, 12)),
        {"entities": {9401: {
            "type": 9400, "position": {"lon": 1.0, "lat": 0.0},
        }}},
        Policy(),
        config(audit_path=path, track_max_age=5, satellite_trigger_step=99),
    )
    register_plan(coordinator, {11: 21000, 12: 21002})
    coordinator.submit(11, observation(
        1, 11, 21000,
        tracks={9401: public_track(9401, 9400, 0.0, 1)},
        position={"lon": 0.0, "lat": 0.0}, stage=3,
    ))
    coordinator.submit(12, observation(1, 12, 21002))
    coordinator.actions_for(11, 0, 0)
    coordinator.submit(11, observation(
        10, 11, 21000,
        tracks={9401: public_track(9401, 9400, 0.0, 1)},
        position={"lon": 0.0, "lat": 0.0}, stage=4,
    ))
    coordinator.submit(12, observation(10, 12, 21002))
    coordinator.actions_for(11, 0, 0)
    coordinator.submit(11, observation(
        11, 11, 21000,
        tracks={9401: public_track(9401, 9400, 0.0, 1)},
        position={"lon": 0.0, "lat": 0.0}, stage=4, health=0,
    ))
    coordinator.submit(12, observation(11, 12, 21002))
    coordinator.actions_for(12, 0, 0)

    terminal = next(
        record for record in audit_rows(path)
        if record["event"] == "terminal_evidence"
    )
    assert terminal["source"] == "detect_stale"
    assert terminal["age"] == pytest.approx(9.0)
    assert terminal["forced_low_confidence"] is True
    assert terminal["delivery_probability"] == pytest.approx(0.15)


def test_no_owner_fresh_candidate_is_rejected_as_track_stale(tmp_path):
    path = tmp_path / "retarget_audit.jsonl"
    coordinator, _, rows = terminal_h_scenario(
        audit_path=path, include_fresh_candidate=False,
    )

    statuses = [
        record for record in audit_rows(path)
        if record["event"] == "retarget_status" and record["owner"] == 13
    ]
    assert statuses[-1]["reason"] == "candidate_track_stale"
    assert statuses[-1]["fresh_target_ids"] == [9401]
    assert statuses[-1]["velocity_x"] == pytest.approx(1.0)
    assert statuses[-1]["velocity_y"] == pytest.approx(0.0)
    assert statuses[-1]["max_target_separation_km"] == pytest.approx(15.0)
    assert statuses[-1]["max_turn_angle_degrees"] == pytest.approx(60.0)
    assert statuses[-1]["max_candidate_distance_km"] == pytest.approx(60.0)
    assert statuses[-1]["max_per_old_target_per_step"] == 1
    assert coordinator._assignments[13] == 9401
    assert not any(row[0] == 4 for row in rows)


@pytest.mark.parametrize(
    ("scenario_values", "expected"),
    (
        ({"candidate_lon": 0.2}, "target_separation"),
        ({"velocity": (-1.0, 0.0)}, "turn_angle"),
        ({"velocity": None}, "turn_angle"),
    ),
)
def test_retarget_geometry_and_missing_velocity_rejections(
    tmp_path, scenario_values, expected
):
    path = tmp_path / f"{expected}-{len(scenario_values)}.jsonl"
    coordinator, _, rows = terminal_h_scenario(
        audit_path=path, **scenario_values,
    )

    status = [
        record for record in audit_rows(path)
        if record["event"] == "retarget_status" and record["owner"] == 13
    ][-1]
    assert status["reason"] == expected
    assert coordinator._assignments[13] == 9401
    assert not any(row[0] == 4 for row in rows)


def test_skill_limits_same_old_target_to_one_request_and_uses_latest_group_track(
    tmp_path,
):
    path = tmp_path / "retarget_audit.jsonl"
    dynamic = DynamicRetargetConfig(max_fraction_per_step=1.0)
    coordinator = TeamCoordinator(
        ((1, 11), (2, 12), (3, 13), (4, 14)),
        dynamic_public_targets(),
        Policy(),
        config(audit_path=path, dynamic=dynamic),
    )
    register_plan(
        coordinator,
        {11: 21000, 12: 21000, 13: 21001, 14: 21001},
    )
    all_owners = (11, 12, 13, 14)
    for owner, kind in ((11, 21000), (12, 21000), (13, 21001), (14, 21001)):
        tracks = {9401: public_track(9401, 9400, 0.0, 1)}
        if kind == 21001:
            tracks[9601] = public_track(9601, 9600, 0.1, 1)
        coordinator.submit(owner, observation(
            1, owner, kind,
            peers=tuple(item for item in all_owners if item != owner),
            tracks=tracks,
            position={"lon": -0.2 if kind == 21001 else 0.0, "lat": 0.0},
            stage=2 if kind == 21001 else 3,
            velocity={"x": 1.0, "y": 0.0} if kind == 21001 else None,
        ))
    coordinator.actions_for(11, 0, 0)
    coordinator.actions_for(13, 0, 0)
    coordinator.actions_for(14, 0, 0)

    coordinator.submit(11, observation(
        2, 11, 21000, peers=(12, 13, 14),
        tracks={9401: public_track(9401, 9400, 0.0, 2)},
        position={"lon": 0.0, "lat": 0.0}, stage=3, health=0,
    ))
    coordinator.submit(12, observation(
        2, 12, 21000, peers=(11, 13, 14),
        tracks={9401: public_track(9401, 9400, 0.0, 2)},
        position={"lon": 0.0, "lat": 0.0}, stage=3, health=0,
    ))
    coordinator.submit(13, observation(
        2, 13, 21001, peers=(14,),
        tracks={
            9401: public_track(9401, 9400, 0.0, 2),
            9601: public_track(9601, 9600, 0.09, 1),
        },
        position={"lon": -0.2, "lat": 0.0},
        velocity={"x": 1.0, "y": 0.0},
    ))
    coordinator.submit(14, observation(
        2, 14, 21001, peers=(13,),
        tracks={
            9401: public_track(9401, 9400, 0.0, 2),
            9601: public_track(9601, 9600, 0.12, 2),
        },
        position={"lon": -0.2, "lat": 0.0},
        velocity={"x": 1.0, "y": 0.0},
    ))
    rows = coordinator.actions_for(13, 0, 0)

    assert sum(row[0] == 4 and row[2] == 9601 for row in rows) == 1
    assert sorted(coordinator._assignments[owner] for owner in (13, 14)) == [9401, 9601]
    step_two_statuses = {
        record["owner"]: record["reason"]
        for record in audit_rows(path)
        if record["event"] == "retarget_status"
        and record["step"] == 2
        and record["owner"] in (13, 14)
    }
    assert sorted(step_two_statuses.values()) == ["eligible", "old_target_step_limit"]
    request = next(
        record for record in audit_rows(path)
        if record["event"] == "retarget_request"
    )
    assert request["candidate_position"] == {"lat": 0.0, "lon": 0.12}
    assert request["candidate_position_source"] == "detect_fresh"


def test_projected_probability_from_initial_commitments_never_triggers_retarget():
    coordinator = TeamCoordinator(
        ((1, 11), (2, 12), (3, 13)), dynamic_public_targets(), Policy(), config(),
    )
    register_plan(coordinator, {11: 21000, 12: 21000, 13: 21001})
    for owner, kind in ((11, 21000), (12, 21000), (13, 21001)):
        peers = tuple(item for item in (11, 12, 13) if item != owner)
        coordinator.submit(owner, observation(
            1, owner, kind, peers=peers,
            position={"lon": 0.5 if owner == 13 else 0.0, "lat": 0.0},
        ))

    rows = coordinator.actions_for(13, 0, 0)
    belief = coordinator._damage_by_group[(11, 12, 13)].estimate(9401)

    assert belief.destruction_probability == 0.0
    assert belief.projected_destruction_probability == 1.0
    assert coordinator._assignments[13] == 9401
    assert [row for row in rows if row[0] == 4] == [(4, 13, 9401, 0)]


def test_terminal_evidence_is_not_visible_to_a_different_group():
    coordinator, _, rows = terminal_h_scenario(same_group=False)

    assert coordinator._damage_by_group[(13,)].estimate(9401).destruction_probability == 0.0
    assert coordinator._assignments[13] == 9401
    assert not any(row[0] == 4 for row in rows)


@pytest.mark.parametrize(
    ("m_stage", "m_lon"),
    ((3, -0.2), (2, 0.01)),
)
def test_stage_and_current_distance_prevent_retarget(m_stage, m_lon):
    coordinator, _, rows = terminal_h_scenario(m_stage=m_stage, m_lon=m_lon)

    assert coordinator._damage_by_group[(13,)].estimate(9401).destruction_probability == pytest.approx(0.9604)
    assert coordinator._assignments[13] == 9401
    assert not any(row[0] == 4 for row in rows)


def test_h_commitment_is_core_and_cannot_retarget():
    hm = PublicStrikeConfig(
        allocation_mode="kill_probability",
        max_h_per_mother=3,
        min_h_per_sentry=0,
    )
    coordinator = TeamCoordinator(
        ((1, 11), (2, 12), (3, 14)), dynamic_public_targets(), Policy(),
        config(hm=hm),
    )
    register_plan(coordinator, {11: 21000, 12: 21000, 14: 21000})
    for owner in (11, 12, 14):
        peers = tuple(item for item in (11, 12, 14) if item != owner)
        lon = 0.0 if owner != 14 else -0.2
        coordinator.submit(owner, observation(
            1, owner, 21000, peers=peers,
            tracks={9401: public_track(9401, 9400, 0.0, 1)},
            position={"lon": lon, "lat": 0.0},
        ))
    coordinator.actions_for(11, 0, 0)
    coordinator.actions_for(14, 0, 0)
    assert coordinator._damage_by_group[(11, 12, 14)].estimate(9401).core_committed_missiles == (11, 12, 14)

    coordinator.submit(14, observation(
        2, 14, 21000,
        tracks={9401: public_track(9401, 9400, 0.0, 2)},
        position={"lon": -0.2, "lat": 0.0},
    ))
    for owner in (11, 12):
        coordinator.submit(owner, observation(
            2, owner, 21000, peers=(12 if owner == 11 else 11, 14),
            tracks={9401: public_track(9401, 9400, 0.0, 2)},
            position={"lon": 0.0, "lat": 0.0},
        ))
    rows = coordinator.actions_for(14, 0, 0)

    assert coordinator._damage_by_group[(14,)].estimate(9401).destruction_probability == 0.0
    assert coordinator._assignments[14] == 9401
    assert not any(row[0] == 4 for row in rows)


def test_terminal_event_ids_are_idempotent():
    coordinator, _, _ = terminal_h_scenario()
    terminal_events = tuple(
        event
        for event in coordinator._owner_ledgers[13].values()
        if event.operation == "delivery"
    )
    assert len(terminal_events) == 2
    before = dict(coordinator._owner_ledgers[13])

    for event in terminal_events:
        coordinator._add_ledger_event((13,), event)

    assert coordinator._owner_ledgers[13] == before
    assert len({event.event_id for event in terminal_events}) == 2


def test_reset_clears_evidence_and_rebuilds_initial_commitments():
    coordinator, policy, _ = terminal_h_scenario()
    mission_plan = coordinator._mission_plan
    for owner in (11, 11, 12, 13):
        coordinator.reset_member(owner)

    assert all(
        tuple(event.operation for event in ledger.values()) == ("commit",)
        for ledger in coordinator._owner_ledgers.values()
    )
    assert coordinator._hm_progress == {}
    assert coordinator._retarget_counts == {}
    assert coordinator._last_retarget_steps == {}
    assert coordinator._mission_plan is mission_plan
    assert coordinator._assignments == {11: 9401, 12: 9401, 13: 9401}
    assert coordinator._initial_attack_sent == set()
    assert policy.reset_calls == 1

    for owner, kind in ((11, 21000), (12, 21000), (13, 21001)):
        peers = tuple(item for item in (11, 12, 13) if item != owner)
        coordinator.submit(owner, observation(
            1, owner, kind, peers=peers,
            position={"lon": 0.5 if owner == 13 else 0.0, "lat": 0.0},
        ))
    coordinator.actions_for(11, 0, 0)
    belief = coordinator._damage_by_group[(11, 12, 13)].estimate(9401)
    assert belief.core_committed_missiles == (11, 12)
    assert belief.flexible_committed_missiles == (13,)


def test_communication_group_accepts_official_dict_and_list_forms():
    assert TeamCoordinator._communication_group(
        11, {"commRangeInfo": {12: {}, "13": {}}}
    ) == (11, 12, 13)
    assert TeamCoordinator._communication_group(
        11, {"commRangeInfo": [13, 12]}
    ) == (11, 12, 13)


def test_velocity_xy_falls_back_from_velocity_to_vel_lla():
    assert TeamCoordinator._velocity_xy({
        "velocity": {"speed": 100.0, "heading": 90.0},
        "vel_lla": {"x": 3.0, "y": 4.0},
    }) == (3.0, 4.0)


def test_formation_audit_records_fields_interval_and_same_group(tmp_path):
    path = tmp_path / "formation" / "retarget_audit.jsonl"
    coordinator = TeamCoordinator(
        ((1, 11), (2, 12), (3, 13)), dynamic_public_targets(), Policy(),
        config(audit_path=path, formation_audit_interval=2),
    )
    register_plan(coordinator, {11: 21000, 12: 21000, 13: 21001})
    for step in range(1, 5):
        tracks = {9401: public_track(9401, 9400, 0.0, step)}
        coordinator.submit(11, observation(
            step, 11, 21000, peers=(12, 13), tracks=tracks,
            position={"lon": -0.02, "lat": 0.001},
        ))
        coordinator.submit(12, observation(
            step, 12, 21000, peers=(11, 13), tracks=tracks,
            position={"lon": -0.04, "lat": -0.001},
        ))
        coordinator.submit(13, observation(
            step, 13, 21001, peers=(11, 12), tracks=tracks,
            position={"lon": -0.1, "lat": 0.0}, stage=2,
            velocity={"speed": 400.0, "up": 1.0, "heading": 90.0},
        ))
        coordinator.actions_for(13, 0, 0)

    records = [
        record for record in audit_rows(path)
        if record["event"] == "formation_status"
    ]
    assert [record["step"] for record in records] == [2, 4]
    assert all(record["m_id"] == 13 for record in records)
    sample = records[0]
    assert sample["target_id"] == 9401
    assert sample["m_stage"] == 2
    assert sample["m_lon"] == pytest.approx(-0.1)
    assert sample["m_lat"] == pytest.approx(0.0)
    assert sample["m_track_source"] == "detect_fresh"
    assert sample["m_track_age"] == pytest.approx(0.0)
    assert sample["h_ids"] == [11, 12]
    assert sample["nearest_h_id"] == 12
    assert sample["separation_km"] > 0.0
    assert sample["lateral_km"] is not None
    assert sample["forward_km"] > 0.0
    assert sample["same_communication_group"] is True
    assert sample["group"] == [11, 12, 13]
    assert sample["movement"] == 0
    assert sample["velocity"] == {
        "heading": 90.0, "speed": 400.0, "up": 1.0,
    }


def test_formation_audit_records_no_h_and_resets_deduplication(tmp_path):
    path = tmp_path / "retarget_audit.jsonl"
    coordinator = TeamCoordinator(
        ((1, 13),), dynamic_public_targets(), Policy(),
        config(audit_path=path, formation_audit_interval=2),
    )
    register_plan(coordinator, {13: 21001})

    coordinator.submit(13, observation(2, 13, 21001))
    coordinator.actions_for(13, 0, 0)
    coordinator.actions_for(13, 0, 0)
    first = [
        record for record in audit_rows(path)
        if record["event"] == "formation_status"
    ]
    assert len(first) == 1
    assert first[0]["h_ids"] == []
    assert first[0]["nearest_h_id"] is None
    assert first[0]["separation_km"] is None
    assert first[0]["same_communication_group"] is None
    assert first[0]["reason"] == "no_active_h_lead"

    coordinator.reset_member(13)
    coordinator.submit(13, observation(2, 13, 21001))
    coordinator.actions_for(13, 0, 0)
    after_reset = [
        record for record in audit_rows(path)
        if record["event"] == "formation_status"
    ]
    assert len(after_reset) == 2


def test_formation_audit_uses_only_same_target_h_in_current_group(tmp_path):
    path = tmp_path / "retarget_audit.jsonl"
    coordinator = TeamCoordinator(
        ((1, 11), (2, 12), (3, 13)), dynamic_public_targets(), Policy(),
        config(
            audit_path=path,
            formation_audit_interval=1,
            dynamic_enabled=False,
        ),
    )
    register_plan(coordinator, {11: 21000, 12: 21000, 13: 21001})
    coordinator.submit(11, observation(
        1, 11, 21000, peers=(13,), position={"lon": -0.1, "lat": 0.0},
    ))
    coordinator.submit(12, observation(
        1, 12, 21000, position={"lon": -0.001, "lat": 0.0},
    ))
    coordinator.submit(13, observation(
        1, 13, 21001, peers=(11,), position={"lon": 0.0, "lat": 0.0},
    ))

    coordinator.actions_for(13, 0, 0)

    record = next(
        row for row in audit_rows(path)
        if row["event"] == "formation_status"
    )
    assert record["h_ids"] == [11]
    assert record["nearest_h_id"] == 11
    assert record["group"] == [11, 13]


def test_missile_trace_is_owner_local_bounded_and_terminalizes_missing_m(tmp_path):
    path = tmp_path / "retarget_audit.jsonl"
    coordinator = TeamCoordinator(
        ((1, 11), (2, 13)), dynamic_public_targets(), Policy(),
        config(
            audit_path=path,
            dynamic_enabled=False,
            trace_interval=2,
            terminal_history=2,
        ),
    )
    register_plan(coordinator, {11: 21000, 13: 21001})
    for step in (1, 2, 3, 4, 5):
        tracks = {9401: public_track(9401, 9400, 0.0, step)}
        coordinator.submit(11, observation(
            step, 11, 21000, tracks=tracks,
            position={"lon": -0.02, "lat": 0.0},
        ))
        coordinator.submit(13, observation(
            step, 13, 21001, tracks=tracks,
            position={"lon": -0.1 + step * 0.01, "lat": 0.0},
            velocity={"x": 120.0, "y": 0.0},
        ))
        coordinator.actions_for(13, 0, 0)
        coordinator.actions_for(13, 0, 0)

    coordinator.submit(11, observation(
        6, 11, 21000,
        tracks={9401: public_track(9401, 9400, 0.0, 6)},
        position={"lon": -0.01, "lat": 0.0},
    ))
    coordinator.actions_for(11, 0, 0)

    records = audit_rows(path)
    traces = [row for row in records if row["event"] == "missile_trace"]
    assert [row["step"] for row in traces] == [1, 3, 5]
    assert len(coordinator._missile_trace_history[13]) == 2
    assert coordinator._missile_trace_history[13].maxlen == 2
    sample = traces[-1]
    assert sample["id"] == 13
    assert sample["type"] == 21001
    assert sample["position"] == {"lat": 0.0, "lon": pytest.approx(-0.05)}
    assert sample["velocity"] == {"x": 120.0, "y": 0.0}
    assert sample["current_target_id"] == 9401
    assert sample["aimpoint"] == {"lat": 0.0, "lon": 0.0}
    assert sample["track_source"] == "detect_fresh"
    assert sample["track_fingerprint"] == [5.0, 0.0, 0.0]
    assert sample["track_age"] == pytest.approx(0.0)
    assert sample["aimpoint_distance_km"] > 0.0
    assert sample["comm_group"] == [13]
    assert sample["same_target_active_h_ids"] == [11]
    assert sample["nearest_h_id"] == 11
    assert sample["nearest_h_separation_km"] > 0.0
    assert sample["same_group"] is False
    assert sample["launch_step"] == 0
    assert sample["steps_since_launch"] == 5
    assert sample["progress_step"] == 5

    terminal_evidence = next(
        row for row in records
        if row["event"] == "terminal_evidence" and row["owner"] == 13
    )
    assert terminal_evidence["step"] == 6
    assert terminal_evidence["terminal_reason"] == "owner_missing"
    assert terminal_evidence["evidence_recorded"] is True
    assert terminal_evidence["delivery_probability"] == pytest.approx(0.15)
    terminal = next(row for row in records if row["event"] == "missile_terminal")
    assert terminal["id"] == 13
    assert terminal["terminal_reason"] == "owner_missing"
    assert terminal["history_length"] == 2
    assert [row["step"] for row in terminal["history"]] == [3, 5]
    assert terminal["last_sample"] == terminal["history"][-1]


def test_disabled_missile_trace_adds_no_trace_or_terminal_events(tmp_path):
    path = tmp_path / "retarget_audit.jsonl"
    coordinator = TeamCoordinator(
        ((1, 11), (2, 13)), dynamic_public_targets(), Policy(),
        config(audit_path=path, dynamic_enabled=False, trace_interval=0),
    )
    register_plan(coordinator, {11: 21000, 13: 21001})
    for owner, kind in ((11, 21000), (13, 21001)):
        coordinator.submit(owner, observation(1, owner, kind))
    coordinator.actions_for(13, 0, 0)
    coordinator.submit(11, observation(2, 11, 21000))
    coordinator.actions_for(11, 0, 0)

    events = [row["event"] for row in audit_rows(path)]
    assert "terminal_evidence" in events
    assert "missile_trace" not in events
    assert "missile_terminal" not in events
    assert coordinator._missile_trace_history == {}


def test_disabled_dynamic_retarget_status_only_records_reason_changes(tmp_path):
    path = tmp_path / "retarget_audit.jsonl"
    coordinator = TeamCoordinator(
        ((1, 13),), dynamic_public_targets(), Policy(),
        config(audit_path=path, dynamic_enabled=False),
    )
    register_plan(coordinator, {13: 21001})
    for step in (1, 2, 3):
        coordinator.submit(13, observation(
            step, 13, 21001,
            position={"lon": 0.5 - step * 0.1, "lat": 0.0},
        ))
        coordinator.actions_for(13, 0, 0)

    statuses = [
        row for row in audit_rows(path) if row["event"] == "retarget_status"
    ]
    assert len(statuses) == 1
    assert statuses[0]["reason"] == "dynamic_retarget_disabled"


def test_retarget_audit_records_structured_decision_chain_and_reset(tmp_path):
    path = tmp_path / "audit" / "retarget_audit.jsonl"
    path.parent.mkdir()
    path.write_text("stale data must be truncated\n", encoding="utf-8")
    coordinator, _, _ = terminal_h_scenario(audit_path=path)
    coordinator.submit(13, observation(
        3, 13, 21001, position={"lon": 0.5, "lat": 0.0}, stage=2,
    ))
    coordinator.actions_for(13, 0, 0)

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in raw_lines]
    assert records
    assert all(list(record) == sorted(record) for record in records)
    assert all("event" in record and "step" in record for record in records)

    commitments = [
        record for record in records if record["event"] == "initial_commitment"
    ]
    assert {(record["owner"], record["type"], record["commitment"]) for record in commitments} == {
        (11, 21000, "core"),
        (12, 21000, "core"),
        (13, 21001, "flexible"),
    }
    assert all(record["target"] == 9401 for record in commitments)
    assert {tuple(record["group"]) for record in commitments} == {
        (11,), (12,), (13,),
    }

    terminal = [
        record for record in records if record["event"] == "terminal_evidence"
    ]
    assert {record["owner"] for record in terminal} == {11, 12}
    assert all(record["type"] == 21000 for record in terminal)
    assert all(record["target"] == 9401 for record in terminal)
    assert all(record["last_stage"] == 3 for record in terminal)
    assert all(record["min_distance_m"] == pytest.approx(0.0) for record in terminal)
    assert all(record["last_group"] == [11, 12, 13] for record in terminal)
    assert all(record["delivery_probability"] == pytest.approx(0.98) for record in terminal)
    assert all(record["event_id"].startswith("terminal:") for record in terminal)

    beliefs = [
        record for record in records
        if record["event"] == "belief_change"
        and record["group"] == [13]
        and record["target"] == 9401
    ]
    assert beliefs[-1]["type"] == 9400
    assert beliefs[-1]["delivered_probability"] == pytest.approx(0.9604)
    assert beliefs[-1]["committed_ids"] == []
    assert beliefs[-1]["delivery_ids"] == [11, 12]

    statuses = [
        record for record in records
        if record["event"] == "retarget_status" and record["owner"] == 13
    ]
    assert [record["reason"] for record in statuses] == [
        "target_probability_below_threshold",
        "eligible",
        "max_retargets",
    ]
    assert all(record["threshold"] == pytest.approx(0.90) for record in statuses)
    assert all(record["type"] == 21001 for record in statuses)

    requests = [record for record in records if record["event"] == "retarget_request"]
    applied = [record for record in records if record["event"] == "retarget_applied"]
    assert len(requests) == len(applied) == 1
    for record in (requests[0], applied[0]):
        assert record["owner"] == 13
        assert (record["old_target"], record["new_target"]) == (9401, 9601)
        assert record["probability"] == pytest.approx(0.9604)
        assert record["group"] == [13]
        assert record["stage"] == 2
        assert record["distance_km"] > 15.0
        assert record["reason"] == "current_target_delivered_destruction_threshold_met"

    before_reset = len(records)
    coordinator.reset_member(11)
    after_reset = audit_rows(path)
    assert len(after_reset) == before_reset + 4
    assert after_reset[before_reset] == {"event": "reset", "owner": 11, "step": 3}
    assert all(
        record["event"] == "initial_commitment"
        for record in after_reset[before_reset + 1:]
    )
    assert coordinator._audited_beliefs == {}
    assert coordinator._audited_retarget_status == {}


def test_audit_suppresses_unchanged_beliefs_and_retarget_status(tmp_path):
    path = tmp_path / "retarget_audit.jsonl"
    coordinator = TeamCoordinator(
        ((1, 11), (2, 12), (3, 13)), dynamic_public_targets(), Policy(),
        config(audit_path=path),
    )
    register_plan(coordinator, {11: 21000, 12: 21000, 13: 21001})
    for step in (1, 2):
        for owner, kind in ((11, 21000), (12, 21000), (13, 21001)):
            peers = tuple(item for item in (11, 12, 13) if item != owner)
            coordinator.submit(owner, observation(
                step, owner, kind, peers=peers,
                position={"lon": 0.5 if owner == 13 else 0.0, "lat": 0.0},
            ))
        coordinator.actions_for(11, 0, 0)
        if step == 1:
            first_step_rows = audit_rows(path)

    records = audit_rows(path)
    assert records == first_step_rows
    statuses = [
        record for record in records
        if record["event"] == "retarget_status" and record["owner"] == 13
    ]
    assert [record["reason"] for record in statuses] == [
        "target_probability_below_threshold"
    ]


@pytest.mark.parametrize(
    ("m_stage", "m_lon", "expected"),
    ((3, -0.2, "stage_too_late"), (2, 0.01, "too_close")),
)
def test_audit_records_retarget_rejection_reason(
    tmp_path, m_stage, m_lon, expected
):
    path = tmp_path / f"{expected}.jsonl"
    terminal_h_scenario(m_stage=m_stage, m_lon=m_lon, audit_path=path)

    statuses = [
        record for record in audit_rows(path)
        if record["event"] == "retarget_status" and record["owner"] == 13
    ]
    assert statuses[-1]["reason"] == expected
    assert statuses[-1]["stage"] == m_stage
    assert statuses[-1]["distance_km"] >= 0.0


def test_audit_records_no_eligible_candidate(tmp_path):
    path = tmp_path / "retarget_audit.jsonl"
    public = {"entities": {
        9401: {"type": 9400, "position": {"lon": 0.0, "lat": 0.0}},
    }}
    coordinator = TeamCoordinator(
        ((1, 11), (2, 12), (3, 13)), public, Policy(), config(audit_path=path),
    )
    register_plan(coordinator, {11: 21000, 12: 21000, 13: 21001})
    for owner, kind in ((11, 21000), (12, 21000), (13, 21001)):
        peers = tuple(item for item in (11, 12, 13) if item != owner)
        tracks = {9401: public_track(9401, 9400, 0.0, 1)}
        coordinator.submit(owner, observation(
            1, owner, kind, peers=peers, tracks=tracks,
            position={"lon": 0.5 if owner == 13 else 0.0, "lat": 0.0},
            stage=3 if kind == 21000 else 2,
            velocity={"x": -1.0, "y": 0.0} if kind == 21001 else None,
        ))
    coordinator.actions_for(11, 0, 0)
    coordinator.actions_for(13, 0, 0)
    coordinator.submit(11, observation(
        2, 11, 21000, peers=(12, 13),
        tracks={9401: public_track(9401, 9400, 0.0, 2)},
        position={"lon": 0.0, "lat": 0.0}, stage=3, health=0,
    ))
    coordinator.submit(12, observation(
        2, 12, 21000, peers=(11, 13),
        tracks={9401: public_track(9401, 9400, 0.0, 2)},
        position={"lon": 0.0, "lat": 0.0}, stage=3, health=0,
    ))
    coordinator.submit(13, observation(
        2, 13, 21001,
        tracks={9401: public_track(9401, 9400, 0.0, 2)},
        position={"lon": 0.5, "lat": 0.0}, stage=2,
        velocity={"x": -1.0, "y": 0.0},
    ))
    coordinator.actions_for(13, 0, 0)

    statuses = [
        record for record in audit_rows(path)
        if record["event"] == "retarget_status" and record["owner"] == 13
    ]
    assert statuses[-1]["reason"] == "no_eligible_candidate"


def test_audit_records_cooldown_after_first_retarget(tmp_path):
    path = tmp_path / "retarget_audit.jsonl"
    coordinator, _, _ = terminal_h_scenario(
        audit_path=path,
        dynamic=DynamicRetargetConfig(max_retargets=2),
    )
    coordinator.submit(13, observation(
        3, 13, 21001, position={"lon": 0.5, "lat": 0.0}, stage=2,
    ))
    coordinator.actions_for(13, 0, 0)

    statuses = [
        record for record in audit_rows(path)
        if record["event"] == "retarget_status" and record["owner"] == 13
    ]
    assert statuses[-1]["reason"] == "cooldown"
    assert statuses[-1]["last_retarget_step"] == 2


def test_disabled_audit_creates_no_file(tmp_path):
    path = tmp_path / "retarget_audit.jsonl"
    coordinator = TeamCoordinator(
        ((1, 11),), dynamic_public_targets(), Policy(), config(audit_path=None),
    )
    register_plan(coordinator, {11: 21001})
    coordinator.submit(11, observation(1, 11, 21001))
    coordinator.actions_for(11, 0, 0)
    assert not path.exists()


def test_audit_initialization_failure_is_not_silenced(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("block", encoding="utf-8")
    with pytest.raises(OSError):
        TeamCoordinator(
            ((1, 11),), dynamic_public_targets(), Policy(),
            config(audit_path=blocker / "retarget_audit.jsonl"),
        )
