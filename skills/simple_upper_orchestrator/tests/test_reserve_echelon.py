import json
from pathlib import Path

import numpy as np
import pytest

from hm_public_strike_packages_skill import PublicStrikeConfig
from l_satellite_strike_packages_skill import SatelliteStrikeConfig
from reserve_echelon_skill import ReservePackageSnapshot
from staggered_deployment_skill import StaggeredDeploymentConfig

from simple_upper_orchestrator_skill import __version__
from simple_upper_orchestrator_skill.config import OrchestratorConfig
from simple_upper_orchestrator_skill.coordinator import TeamCoordinator
from simple_upper_orchestrator_skill.deploy_agent import PreMissionPlan


class Policy:
    def encode_motion_observation(self, observation, **kwargs):
        return np.zeros(3)

    def prepare_action_mask(self, observation, mask):
        return mask

    def select_action(self, observation, mask):
        return 1


def test_release_version():
    assert __version__ == "2.2.0"


def configured(
    *, l_first_wave_per_target=0, deadline=300, max_stage=None,
    expected_target_count=None, audit_path=None, max_steps=1000,
):
    return OrchestratorConfig(
        Path("."),
        Path("checkpoint"),
        max_steps,
        deployment=StaggeredDeploymentConfig(deployment_gap_km=1),
        hm_strike=PublicStrikeConfig(),
        l_strike=SatelliteStrikeConfig(
            satellite_trigger_step=0,
            expected_target_count=expected_target_count,
        ),
        l_first_wave_per_target=l_first_wave_per_target,
        reserve_deadline_step=deadline,
        reserve_max_stage=max_stage,
        dynamic_retarget_enabled=False,
        terminal_guidance_enabled=False,
        audit_path=audit_path,
    )


def public_targets():
    return {"entities": {
        9401: {"type": 9400, "position": {"lon": 0.0, "lat": 0.0}},
        9601: {"type": 9600, "position": {"lon": 0.2, "lat": 0.0}},
        9602: {"type": 9600, "position": {"lon": 0.4, "lat": 0.0}},
    }}


def public_tracks(step):
    return {
        target_id: {
            "entity_id": target_id,
            "entity_type": target_type,
            "lla": {"x": lon, "y": 0.0},
            "time": step,
        }
        for target_id, target_type, lon in (
            (9401, 9400, 0.0),
            (9601, 9600, 0.2),
            (9602, 9600, 0.4),
        )
    }


def hidden_tracks(step, count=2):
    return {
        9501 + index: {
            "entity_id": 9501 + index,
            "entity_type": 9500,
            "lla": {"x": index * 0.2, "y": 0.0},
            "time": step,
        }
        for index in range(count)
    }


def observation(
    step, owner, kind, *, peers=(), tracks=None, stage=2, lon=-0.1,
    heading=90.0, speed=200.0,
):
    return {
        "step": step,
        "self": {
            "entity_id": owner,
            "type": kind,
            "health": 100,
            "stage": stage,
            "position": {"lon": lon, "lat": 0.0},
            "velocity": {"speed": speed, "heading": heading},
            "detectInfo": tracks or {},
            "commRangeInfo": {peer: {} for peer in peers},
        },
        "_orchestrator_launch_step": 0,
        "_orchestrator_maneuver_state": 0,
    }


def reserve_h_coordinator(
    *, deadline=300, max_stage=None, audit_path=None, max_steps=1000,
):
    members = ((1, 11), (2, 12), (3, 13))
    coordinator = TeamCoordinator(
        members,
        public_targets(),
        Policy(),
        configured(
            deadline=deadline,
            max_stage=max_stage,
            audit_path=audit_path,
            max_steps=max_steps,
        ),
    )
    package = ReservePackageSnapshot(
        package_id=13,
        member_ids=(13,),
        member_types=(21000,),
        state="STAGING",
        provisional_target_id=9601,
        fallback_target_id=9602,
        responsibility_region="center",
        dispatch_count=0,
    )
    coordinator.set_mission_plan(PreMissionPlan(
        hm_assignments={11: 9401, 12: 9401},
        entity_types={11: 21000, 12: 21000, 13: 21000},
        entity_core={
            owner: {"entity_id": owner, "type": 21000}
            for owner in (11, 12, 13)
        },
        l_roster=(),
        reserve_packages=(package,),
        reserve_target_regions={9401: "center", 9601: "center", 9602: "center"},
    ))
    return coordinator


def submit_h_step(
    coordinator, step, owners, *, reserve_stage=2, core_stage=2,
    same_group=True, reserve_heading=90.0,
):
    all_ids = (11, 12, 13)
    for owner in owners:
        peers = (
            tuple(item for item in all_ids if item != owner)
            if same_group
            else ((12,) if owner == 11 else ())
        )
        coordinator.submit(owner, observation(
            step,
            owner,
            21000,
            peers=peers,
            tracks=public_tracks(step),
            stage=reserve_stage if owner == 13 else core_stage,
            heading=reserve_heading if owner == 13 else 90.0,
        ))


def reserve_l_coordinator(*, audit_path=None):
    owners = (100, 101, 102, 103, 104)
    coordinator = TeamCoordinator(
        tuple((index, owner) for index, owner in enumerate(owners)),
        {"entities": {}},
        Policy(),
        configured(
            l_first_wave_per_target=2,
            expected_target_count=2,
            audit_path=audit_path,
        ),
    )
    coordinator.set_mission_plan(PreMissionPlan(
        {},
        {owner: 21002 for owner in owners},
        {owner: {"entity_id": owner, "type": 21002} for owner in owners},
        owners,
    ))
    return coordinator, owners


def test_core_h_early_loss_dispatches_same_group_reserve_and_routes_owner():
    coordinator = reserve_h_coordinator()
    submit_h_step(
        coordinator, 100, (11, 12, 13), reserve_stage=3, core_stage=2,
    )
    coordinator.actions_for(11, 0, 0)
    assert 13 in coordinator._reserve_holding._owners
    submit_h_step(
        coordinator, 101, (12, 13), reserve_stage=3, core_stage=2,
    )

    rows = coordinator.actions_for(13, 0, 0)

    loss = coordinator._hm_early_losses[11]
    deficits = coordinator._hm_reserve_deficits((11, 12, 13))
    assert loss.stage == 2
    assert loss.minimum_distance_metres > 10_000
    assert loss.nominal_damage == 16.0
    assert len(deficits) == 1
    assert deficits[0].target_id == 9401
    assert deficits[0].missing_damage == 8.0
    assert coordinator._assignments[13] == 9401
    assert [row for row in rows if row[0] == 4] == [(4, 13, 9401, 0)]
    assert all(row[1] == 13 for row in rows)
    assert coordinator._reserve_packages[13].state == "DISPATCHED"
    assert 13 not in coordinator._reserve_holding._owners
    assert 13 not in coordinator._reserve_holding_anchors
    assert set(coordinator._core_hm_assignments) == {11, 12}


def test_staging_h_reserve_uses_holding_action_without_type4(tmp_path):
    path = tmp_path / "holding.jsonl"
    coordinator = reserve_h_coordinator(audit_path=path)
    assert 13 not in coordinator._assignments
    assert 13 not in coordinator._goals

    submit_h_step(coordinator, 1, (11, 12, 13), reserve_stage=-1)
    core_rows = coordinator.actions_for(11, 0, 0)
    reserve_rows = coordinator.actions_for(13, 0, 0)

    assert [row for row in core_rows if row[0] == 4] == [(4, 11, 9401, 0)]
    assert not any(row[0] == 4 for row in reserve_rows)
    assert reserve_rows == ((0, 13, -1.0, 0),)
    assert 13 not in coordinator._assignments
    snapshot = coordinator._reserve_member_snapshots[13]
    assert snapshot.stage == 0
    assert snapshot.heading_east == pytest.approx(200.0)
    assert snapshot.heading_north == pytest.approx(0.0, abs=1e-9)
    assert snapshot.remaining_steps == 999
    assert coordinator._reserve_packages[13].state == "STAGING"
    holding = next(
        row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        )
        if row["event"] == "reserve_holding" and row["owner"] == 13
    )
    assert holding["phase"] == "left_leg"
    assert holding["cross_track"] == pytest.approx(0.0)
    assert holding["progress"] == pytest.approx(0.0)
    assert holding["baseline"] == 1
    assert holding["final"] == 0


def test_reserve_heading_uses_owner_local_position_window_estimate():
    coordinator = reserve_h_coordinator()
    for step, reserve_lon in ((1, -0.1), (2, -0.099)):
        for owner in (11, 12, 13):
            item = observation(
                step,
                owner,
                21000,
                peers=tuple(
                    candidate
                    for candidate in (11, 12, 13)
                    if candidate != owner
                ),
                tracks=public_tracks(step),
                lon=reserve_lon if owner == 13 else -0.1,
            )
            item["self"].pop("velocity")
            coordinator.submit(owner, item)
        coordinator.actions_for(13, 0, 0)

    snapshot = coordinator._reserve_member_snapshots[13]
    assert snapshot.heading_east is not None
    assert snapshot.heading_east > 100.0
    assert snapshot.heading_north == pytest.approx(0.0, abs=1e-6)
    assert snapshot.remaining_steps == 998


def test_round_reset_clears_holding_and_preserves_preplan():
    coordinator = reserve_h_coordinator()
    mission_plan = coordinator._mission_plan
    submit_h_step(coordinator, 1, (11, 12, 13))
    coordinator.actions_for(13, 0, 0)
    assert 13 in coordinator._reserve_holding._owners
    assert coordinator._reserve_holding_anchors[13] == (-0.1, 0.0)

    for owner in (11, 12, 13):
        coordinator.reset_member(owner)

    assert coordinator._mission_plan is mission_plan
    assert coordinator._reserve_packages[13].state == "STAGING"
    assert coordinator._reserve_holding._owners == {}
    assert coordinator._reserve_holding_anchors == {}


def test_outgroup_early_loss_does_not_dispatch_reserve():
    coordinator = reserve_h_coordinator()
    submit_h_step(coordinator, 1, (11, 12, 13), same_group=False)
    coordinator.actions_for(11, 0, 0)
    submit_h_step(coordinator, 2, (12, 13), same_group=False)

    rows = coordinator.actions_for(13, 0, 0)

    assert 13 not in coordinator._assignments
    assert not any(row[0] == 4 and row[2] == 9401 for row in rows)
    assert coordinator._reserve_packages[13].state == "STAGING"


def test_reserve_stage_later_than_maximum_is_rejected():
    coordinator = reserve_h_coordinator(max_stage=2)
    submit_h_step(coordinator, 1, (11, 12, 13), reserve_stage=3)
    coordinator.actions_for(11, 0, 0)
    submit_h_step(coordinator, 2, (12, 13), reserve_stage=3)

    rows = coordinator.actions_for(13, 0, 0)

    assert 13 not in coordinator._assignments
    assert not any(row[0] == 4 for row in rows)
    assert coordinator._reserve_packages[13].state == "STAGING"


def test_reachability_rejects_large_turn_and_audits_geometry(tmp_path):
    path = tmp_path / "turn-rejected.jsonl"
    coordinator = reserve_h_coordinator(audit_path=path)
    submit_h_step(
        coordinator, 1, (11, 12, 13), reserve_heading=270.0,
    )
    coordinator.actions_for(11, 0, 0)
    submit_h_step(
        coordinator, 2, (12, 13), reserve_heading=270.0,
    )

    rows = coordinator.actions_for(13, 0, 0)

    assert 13 not in coordinator._assignments
    assert not any(row[0] == 4 for row in rows)
    rejected = [
        row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        )
        if row["event"] == "reserve_rejected"
        and row["reason"] == "turn_angle_exceeded"
    ][-1]
    assert rejected["distance_km"] > 10
    assert rejected["turn_angle_degrees"] == pytest.approx(180.0)
    assert rejected["max_turn_angle_degrees"] == 60.0
    assert rejected["travel_steps"] > 0
    assert rejected["remaining_steps"] == 998


def test_h_early_loss_uses_minimum_not_last_target_distance():
    coordinator = reserve_h_coordinator()
    for step, core_lon in ((1, 0.004), (2, -1.0)):
        for owner in (11, 12, 13):
            coordinator.submit(owner, observation(
                step,
                owner,
                21000,
                peers=tuple(item for item in (11, 12, 13) if item != owner),
                tracks=public_tracks(step),
                stage=2,
                lon=core_lon if owner == 11 else -1.0,
            ))
        coordinator.actions_for(11, 0, 0)
    submit_h_step(coordinator, 3, (12, 13))

    rows = coordinator.actions_for(13, 0, 0)

    assert coordinator._hm_progress.get(11) is None
    assert 11 not in coordinator._hm_early_losses
    assert 13 not in coordinator._assignments
    assert not any(row[0] == 4 for row in rows)


def test_ledger_event_survives_loss_group_tuple_change():
    coordinator = reserve_h_coordinator()
    submit_h_step(coordinator, 1, (11, 12, 13))
    coordinator.actions_for(11, 0, 0)
    for owner in (12, 13):
        coordinator.submit(owner, observation(
            2,
            owner,
            21000,
            peers=((13,) if owner == 12 else (12,)),
            tracks=public_tracks(2),
            stage=2,
        ))

    rows = coordinator.actions_for(13, 0, 0)

    loss = coordinator._hm_early_losses[11]
    assert loss.communication_group == (11, 12, 13)
    assert coordinator._reserve_member_snapshots[13].group == (12, 13)
    assert loss.event_id in coordinator._owner_ledgers[13]
    assert coordinator._assignments[13] == 9401
    assert [row for row in rows if row[0] == 4] == [(4, 13, 9401, 0)]


def test_deadline_without_deficit_commits_fallback():
    coordinator = reserve_h_coordinator(deadline=3)
    submit_h_step(coordinator, 3, (11, 12, 13), reserve_stage=4)

    rows = coordinator.actions_for(13, 0, 0)

    assert coordinator._assignments[13] == 9602
    assert [row for row in rows if row[0] == 4] == [(4, 13, 9602, 0)]
    assert coordinator._reserve_packages[13].state == "FALLBACK"


def test_deadline_fallback_obeys_stage_gate():
    coordinator = reserve_h_coordinator(deadline=3, max_stage=2)
    submit_h_step(coordinator, 3, (11, 12, 13), reserve_stage=3)

    rows = coordinator.actions_for(13, 0, 0)

    assert 13 not in coordinator._assignments
    assert not any(row[0] == 4 for row in rows)
    assert coordinator._reserve_packages[13].state == "STAGING"


def test_l_first_six_per_target_leaves_twenty_five_single_reserves():
    owners = tuple(range(100, 185))
    coordinator = TeamCoordinator(
        tuple((index, owner) for index, owner in enumerate(owners)),
        {"entities": {}},
        Policy(),
        configured(l_first_wave_per_target=6, expected_target_count=10),
    )
    coordinator.set_mission_plan(PreMissionPlan(
        {},
        {owner: 21002 for owner in owners},
        {owner: {"entity_id": owner, "type": 21002} for owner in owners},
        owners,
    ))
    tracks = hidden_tracks(1, count=10)
    for owner in owners:
        coordinator.submit(owner, observation(1, owner, 21002, tracks=tracks))

    coordinator.actions_for(100, 0, 0)

    assert len(coordinator._l_core_ids) == 60
    l_packages = [
        package for package in coordinator._reserve_packages.values()
        if package.member_types == (21002,)
    ]
    assert len(l_packages) == 25
    assert all(len(package.member_ids) == 1 for package in l_packages)
    assert all(package.state == "STAGING" for package in l_packages)


def test_l_core_early_loss_dispatches_same_group_reserve():
    owners = (100, 101, 102, 103, 104)
    coordinator = TeamCoordinator(
        tuple((index, owner) for index, owner in enumerate(owners)),
        {"entities": {}},
        Policy(),
        configured(l_first_wave_per_target=2, expected_target_count=2),
    )
    coordinator.set_mission_plan(PreMissionPlan(
        {},
        {owner: 21002 for owner in owners},
        {owner: {"entity_id": owner, "type": 21002} for owner in owners},
        owners,
    ))
    tracks = hidden_tracks(1)
    for owner in owners:
        coordinator.submit(owner, observation(
            1,
            owner,
            21002,
            peers=tuple(item for item in owners if item != owner),
            tracks=tracks,
            stage=4 if owner == 104 else 2,
        ))
    coordinator.actions_for(100, 0, 0)
    for owner in (100, 102, 103, 104):
        coordinator.submit(owner, observation(
            2,
            owner,
            21002,
            peers=tuple(item for item in owners if item != owner),
            tracks=hidden_tracks(2),
            stage=4 if owner == 104 else 2,
        ))

    rows = coordinator.actions_for(104, 0, 0)

    loss = coordinator._l_early_losses[101]
    deficits = coordinator._l_reserve_deficits(owners)
    assert loss.stage == 2
    assert loss.minimum_distance_metres > 5_000
    assert loss.nominal_damage == 0.8
    assert len(deficits) == 1
    assert deficits[0].target_id == 9502
    assert deficits[0].missing_damage == pytest.approx(0.4)
    assert coordinator._assignments[104] == 9502
    assert [row for row in rows if row[0] == 4] == [(4, 104, 9502, 0)]
    assert all(row[1] == 104 for row in rows)


def test_reachability_rejects_when_remaining_time_is_insufficient(tmp_path):
    path = tmp_path / "time-rejected.jsonl"
    coordinator = reserve_h_coordinator(audit_path=path, max_steps=3)
    submit_h_step(coordinator, 1, (11, 12, 13))
    coordinator.actions_for(11, 0, 0)
    submit_h_step(coordinator, 2, (12, 13))

    rows = coordinator.actions_for(13, 0, 0)

    assert 13 not in coordinator._assignments
    assert not any(row[0] == 4 for row in rows)
    rejected = [
        row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        )
        if row["event"] == "reserve_rejected"
        and row["reason"] == "insufficient_time"
    ][-1]
    assert rejected["travel_steps"] > rejected["remaining_steps"]
    assert rejected["remaining_steps"] == 1


def test_l_core_stage_four_loss_does_not_create_deficit():
    coordinator, owners = reserve_l_coordinator()
    tracks = hidden_tracks(1)
    for owner in owners:
        coordinator.submit(owner, observation(
            1,
            owner,
            21002,
            peers=tuple(item for item in owners if item != owner),
            tracks=tracks,
            stage=4 if owner == 101 else 2,
        ))
    coordinator.actions_for(100, 0, 0)
    for owner in (100, 102, 103, 104):
        coordinator.submit(owner, observation(
            2,
            owner,
            21002,
            peers=tuple(item for item in owners if item != owner),
            tracks=hidden_tracks(2),
        ))

    rows = coordinator.actions_for(104, 0, 0)

    assert 101 not in coordinator._l_early_losses
    assert 104 not in coordinator._assignments
    assert not any(row[0] == 4 for row in rows)


def test_l_reserve_is_held_and_uses_persistent_owner_catalogue():
    coordinator, owners = reserve_l_coordinator()
    for owner in owners:
        coordinator.submit(owner, observation(
            1,
            owner,
            21002,
            peers=tuple(item for item in owners if item != owner),
            tracks=hidden_tracks(1),
        ))

    coordinator.actions_for(100, 0, 0)
    held_rows = coordinator.actions_for(104, 0, 0)

    assert 104 not in coordinator._assignments
    assert not any(row[0] == 4 for row in held_rows)
    assert set(coordinator._l_catalogues[104]) == {9501, 9502}

    living = (100, 102, 103, 104)
    for owner in living:
        tracks = {9501: hidden_tracks(2)[9501]}
        if owner == 100:
            tracks[9503] = hidden_tracks(2, count=3)[9503]
        coordinator.submit(owner, observation(
            2,
            owner,
            21002,
            peers=tuple(item for item in living if item != owner),
            tracks=tracks,
        ))

    rows = coordinator.actions_for(104, 0, 0)

    assert set(coordinator._l_catalogues[104]) == {9501, 9502}
    assert 9503 in coordinator._l_catalogues[100]
    assert 9503 not in coordinator._l_catalogues[104]
    assert coordinator._assignments[104] == 9502
    assert [row for row in rows if row[0] == 4] == [(4, 104, 9502, 0)]


def test_zero_reserve_keeps_original_l_rank_allocation():
    owners = (100, 101, 102)
    coordinator = TeamCoordinator(
        tuple((index, owner) for index, owner in enumerate(owners)),
        {"entities": {}},
        Policy(),
        configured(expected_target_count=2),
    )
    coordinator.set_mission_plan(PreMissionPlan(
        {},
        {owner: 21002 for owner in owners},
        {owner: {"entity_id": owner, "type": 21002} for owner in owners},
        owners,
    ))
    for owner in owners:
        coordinator.submit(owner, observation(
            1, owner, 21002, tracks=hidden_tracks(1),
        ))

    coordinator.actions_for(100, 0, 0)

    assert coordinator._assignments == {100: 9501, 101: 9502, 102: 9501}
    assert coordinator._reserve_packages == {}


def test_reserve_audit_and_reset_restore_preplan_staging(tmp_path):
    path = tmp_path / "reserve.jsonl"
    coordinator = reserve_h_coordinator(audit_path=path)
    mission_plan = coordinator._mission_plan
    submit_h_step(coordinator, 1, (11, 12, 13))
    coordinator.actions_for(11, 0, 0)
    submit_h_step(coordinator, 2, (12, 13))
    coordinator.actions_for(13, 0, 0)
    loss = coordinator._hm_early_losses[11]
    ledgers = {
        owner: dict(coordinator._owner_ledgers[owner])
        for owner in (11, 12, 13)
    }
    coordinator._record_early_loss(
        coordinator._hm_early_losses, loss, (11, 12, 13)
    )
    assert all(
        coordinator._owner_ledgers[owner] == ledger
        for owner, ledger in ledgers.items()
    )

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    events = {record["event"] for record in records}
    assert {
        "initial_reserve",
        "reserve_staging",
        "reserve_early_loss",
        "reserve_deficit",
        "reserve_dispatch",
        "reserve_action4",
    } <= events
    early = next(row for row in records if row["event"] == "reserve_early_loss")
    assert {
        "owner", "target", "type", "target_type", "nominal_damage",
        "stage", "min_distance_m", "probability", "last_group",
        "recipients", "event_id",
    } <= early.keys()
    deficit = next(row for row in records if row["event"] == "reserve_deficit")
    dispatch = next(row for row in records if row["event"] == "reserve_dispatch")
    action = next(row for row in records if row["event"] == "reserve_action4")
    assert deficit["event_ids"] == [early["event_id"]]
    assert deficit["target_health"] == 24.0
    assert deficit["initial_core_nominal"] == 32.0
    assert deficit["early_lost_nominal"] == 16.0
    assert deficit["missing"] == 8.0
    assert dispatch["evidence_event_ids"] == [early["event_id"]]
    assert action["owner"] == 13
    assert action["target"] == 9401

    for owner in (11, 12, 13):
        coordinator.reset_member(owner)

    assert coordinator._mission_plan is mission_plan
    assert coordinator._reserve_packages[13].state == "STAGING"
    assert 13 not in coordinator._assignments
    assert coordinator._hm_early_losses == {}
    assert coordinator._reserve_retarget_counts == {}


def test_every_reserve_rejection_is_audited(tmp_path):
    path = tmp_path / "rejected.jsonl"
    coordinator = reserve_h_coordinator(audit_path=path)
    submit_h_step(coordinator, 1, (11, 12, 13), same_group=False)
    coordinator.actions_for(11, 0, 0)
    submit_h_step(coordinator, 2, (12, 13), same_group=False)

    coordinator.actions_for(13, 0, 0)

    rejected = [
        row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        )
        if row["event"] == "reserve_rejected"
    ]
    assert rejected
    assert rejected[-1]["package"] == 13
    assert rejected[-1]["reason"] == "no_deficit"


def test_dispatched_reserve_terminal_does_not_create_recursive_deficit(tmp_path):
    path = tmp_path / "terminal.jsonl"
    coordinator = reserve_h_coordinator(audit_path=path)
    submit_h_step(coordinator, 1, (11, 12, 13))
    coordinator.actions_for(11, 0, 0)
    submit_h_step(coordinator, 2, (12, 13))
    coordinator.actions_for(13, 0, 0)
    assert coordinator._reserve_packages[13].state == "DISPATCHED"

    coordinator.submit(12, observation(
        3,
        12,
        21000,
        peers=(13,),
        tracks=public_tracks(3),
        stage=2,
    ))
    dead = observation(
        3,
        13,
        21000,
        peers=(12,),
        tracks=public_tracks(3),
        stage=2,
    )
    dead["self"]["health"] = 0
    coordinator.submit(13, dead)

    coordinator.actions_for(12, 0, 0)

    assert set(coordinator._hm_early_losses) == {11}
    assert 13 not in coordinator._core_hm_assignments
    assert 13 not in coordinator._reserve_packages
    assert 13 not in coordinator._reserve_seen_members
    assert 13 not in coordinator._reserve_member_snapshots
    assert coordinator._reserve_package_terminations[13]["status"] == "TERMINATED"
    assert coordinator._reserve_echelon.state == ()
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert sum(row["event"] == "reserve_early_loss" for row in records) == 1
    assert any(
        row["event"] == "terminal_evidence"
        and row["owner"] == 13
        and row["evidence_recorded"] is False
        for row in records
    )
    assert any(row["event"] == "reserve_terminal" for row in records)

    for owner in (11, 12, 13):
        coordinator.reset_member(owner)

    assert coordinator._reserve_packages[13].state == "STAGING"
    assert coordinator._reserve_package_terminations == {}
    assert 13 not in coordinator._assignments


def test_staging_reserve_terminal_is_removed_without_early_loss(tmp_path):
    path = tmp_path / "staging-terminal.jsonl"
    coordinator = reserve_h_coordinator(audit_path=path)
    submit_h_step(coordinator, 1, (11, 12, 13))
    coordinator.actions_for(13, 0, 0)
    for owner in (11, 12):
        coordinator.submit(owner, observation(
            2,
            owner,
            21000,
            peers=(12 if owner == 11 else 11, 13),
            tracks=public_tracks(2),
            stage=3,
        ))
    dead = observation(
        2,
        13,
        21000,
        peers=(11, 12),
        tracks=public_tracks(2),
        stage=-1,
    )
    dead["self"]["health"] = 0
    coordinator.submit(13, dead)

    coordinator.actions_for(11, 0, 0)

    assert coordinator._hm_early_losses == {}
    assert 13 not in coordinator._reserve_packages
    assert 13 not in coordinator._reserve_seen_members
    assert 13 not in coordinator._reserve_member_snapshots
    assert coordinator._reserve_package_terminations[13]["status"] == "TERMINATED"
    assert coordinator._reserve_echelon.state == ()
    assert any(
        json.loads(line)["event"] == "reserve_terminal"
        for line in path.read_text(encoding="utf-8").splitlines()
    )

def test_deadline_fallback_is_audited(tmp_path):
    path = tmp_path / "fallback.jsonl"
    coordinator = reserve_h_coordinator(deadline=3, audit_path=path)
    submit_h_step(coordinator, 3, (11, 12, 13))

    coordinator.actions_for(13, 0, 0)

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    fallback = next(row for row in records if row["event"] == "reserve_fallback")
    action = next(row for row in records if row["event"] == "reserve_action4")
    assert fallback["members"] == [13]
    assert fallback["old_target"] == 9601
    assert fallback["new_target"] == 9602
    assert action["owner"] == 13
    assert action["target"] == 9602
