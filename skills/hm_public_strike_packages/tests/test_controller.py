import pytest

from hm_public_strike_packages_skill import (
    HMMissileSnapshot,
    HMPublicStrikePackagesSkill,
    PublicStrikeConfig,
    PublicTargetTrack,
    expected_effective_damage,
    materialize_engine_actions,
    public_targets_from_directory,
    track_updates_from_isolated_observations,
)


def target(target_id, target_type, lon=0.0, lat=0.0, step=0.0):
    health, value = (24.0, 5.0) if target_type == 9400 else (18.0, 2.0)
    return PublicTargetTrack(target_id, target_type, lon, lat, step, health, value)


def missile(entity_id, missile_type, action=1, alive=True, launched=True):
    return HMMissileSnapshot(entity_id, missile_type, alive, launched, action)


def observation(entity_id, missile_type, detections):
    return {
        "self": {
            "entity_id": entity_id,
            "type": missile_type,
            "detectInfo": {
                item[0]: {
                    "entity_id": item[0],
                    "entity_type": item[1],
                    "time": item[4],
                    "lla": {"x": item[2], "y": item[3], "z": 0.0},
                }
                for item in detections
            },
        },
    }


def test_expected_effective_damage_uses_missile_and_target_type() -> None:
    assert expected_effective_damage(21000, 9400) == pytest.approx(16.0)
    assert expected_effective_damage(21001, 9400) == pytest.approx(4.0)
    assert expected_effective_damage(21000, 9600) == pytest.approx(12.0)
    assert expected_effective_damage(21001, 9600) == pytest.approx(3.0)


def test_covers_every_target_by_expected_damage_not_equal_missile_count() -> None:
    config = PublicStrikeConfig(safety_factor=1.0)
    skill = HMPublicStrikePackagesSkill(
        config,
        public_targets=(target(901, 9400), target(902, 9600)),
    )
    members = (
        missile(10, 21000), missile(11, 21000),
        missile(20, 21001), missile(21, 21001),
        missile(22, 21001), missile(23, 21001),
    )

    decision = skill.decide(step=0, members=members)

    assert [item.member_ids for item in decision.packages] == [
        (10, 20, 21), (11, 22, 23),
    ]
    assert [item.expected_damage for item in decision.packages] == pytest.approx([24.0, 18.0])
    assert all(item.expected_damage >= item.required_damage for item in decision.packages)


def test_feasible_mixed_inventory_is_planned_globally() -> None:
    config = PublicStrikeConfig(safety_factor=1.0)
    skill = HMPublicStrikePackagesSkill(
        config,
        public_targets=(target(901, 9400), target(902, 9400)),
    )
    decision = skill.decide(
        step=0,
        members=(
            missile(10, 21000), missile(11, 21000), missile(12, 21000),
            missile(20, 21001), missile(21, 21001),
        ),
    )

    assert len(decision.packages) == 2
    assert all(item.expected_damage >= 24.0 for item in decision.packages)


def test_inventory_shortage_gives_each_target_fair_deterministic_coverage() -> None:
    targets = (target(902, 9600), target(901, 9400))
    members = (missile(11, 21000), missile(10, 21000))

    first = HMPublicStrikePackagesSkill(public_targets=targets).decide(
        step=0, members=members,
    )
    second = HMPublicStrikePackagesSkill(public_targets=reversed(targets)).decide(
        step=0, members=reversed(members),
    )

    assert [(item.target.target_id, item.member_ids) for item in first.packages] == [
        (901, (10,)), (902, (11,)),
    ]
    assert first == second


def test_reserves_are_not_assigned() -> None:
    config = PublicStrikeConfig(reserve_h=1, reserve_m=1, safety_factor=1.0)
    skill = HMPublicStrikePackagesSkill(config, public_targets=(target(901, 9400),))
    decision = skill.decide(
        step=0,
        members=(
            missile(10, 21000), missile(11, 21000),
            missile(20, 21001), missile(21, 21001), missile(22, 21001),
        ),
    )

    assert decision.packages[0].member_ids == (10, 20, 21)
    assert decision.unassigned_member_ids == (11, 22)


def test_hm_baselines_pass_through_and_runtime_routes_by_owner() -> None:
    skill = HMPublicStrikePackagesSkill(public_targets=(target(901, 9400),))
    decision = skill.decide(
        step=0,
        members=(missile(10, 21000, action=0), missile(20, 21001, action=2)),
    )

    assert decision.baseline_actions == ((10, 0), (20, 2))
    assert materialize_engine_actions(decision) == {
        10: ((0.0, 10.0, -1.0, 0.0), (4.0, 10.0, 901.0, 0.0)),
        20: ((0.0, 20.0, 1.0, 0.0), (4.0, 20.0, 901.0, 0.0)),
    }


def test_legal_coordinate_refresh_reissues_attack_for_same_target() -> None:
    initial = (target(901, 9400, lon=1.0, lat=2.0),)
    skill = HMPublicStrikePackagesSkill(public_targets=initial)
    members = (missile(10, 21000),)
    first = skill.decide(step=0, members=members)
    updates = track_updates_from_isolated_observations(
        {10: observation(10, 21000, ((901, 9400, 3.0, 4.0, 1.0),))},
        public_targets=initial,
        step=1,
    )

    refreshed = skill.decide(step=1, members=members, track_updates=updates)

    assert first.attack_actions[0].target_id == 901
    assert not refreshed.allocation_changed
    assert [(item.entity_id, item.target_id) for item in refreshed.attack_actions] == [(10, 901)]
    assert refreshed.packages[0].target.lon == 3.0
    assert refreshed.packages[0].target.lat == 4.0


def test_hidden_and_unknown_detections_are_ignored_and_hidden_dto_is_rejected() -> None:
    initial = (target(901, 9400),)
    updates = track_updates_from_isolated_observations(
        {10: observation(10, 21001, (
            (950, 9500, 1.0, 1.0, 1.0),
            (999, 9600, 2.0, 2.0, 1.0),
        ))},
        public_targets=initial,
        step=1,
    )

    assert updates == ()
    with pytest.raises(ValueError, match="9400 or 9600"):
        PublicTargetTrack(950, 9500, 0.0, 0.0, 0.0, 10.0, 1.0)


def test_lock_holds_assignment_and_reset_restores_initial_directory() -> None:
    initial = (target(901, 9400, lon=1.0),)
    skill = HMPublicStrikePackagesSkill(public_targets=initial)
    first = skill.decide(step=0, members=(missile(10, 21000),))
    locked = skill.decide(step=1, members=(missile(11, 21000),))

    assert first.packages[0].member_ids == (10,)
    assert locked.packages == ()
    assert locked.unassigned_member_ids == (11,)

    skill.reset()
    fresh = skill.decide(step=2, members=(missile(11, 21000),))
    assert fresh.packages[0].member_ids == (11,)
    assert fresh.packages[0].target.lon == 1.0
    assert fresh.attack_actions[0].entity_id == 11


def test_unlocked_assignment_tracks_active_members() -> None:
    config = PublicStrikeConfig(lock_assignments=False)
    skill = HMPublicStrikePackagesSkill(config, public_targets=(target(901, 9400),))
    skill.decide(step=0, members=(missile(10, 21000),))

    changed = skill.decide(step=1, members=(missile(11, 21000),))

    assert changed.allocation_changed
    assert changed.packages[0].member_ids == (11,)


def test_directory_defaults_and_strict_validation() -> None:
    tracks = public_targets_from_directory({
        901: {"entity_id": 901, "entity_type": 9400, "lla": {"x": 1, "y": 2}},
        902: {"entity_id": 902, "entity_type": 9600, "lla": {"x": 3, "y": 4}},
    })
    assert [(item.initial_health, item.value) for item in tracks] == [(24.0, 5.0), (18.0, 2.0)]

    release_tracks = public_targets_from_directory({
        903: {"entity_id": 903, "entity_type": 9400, "position": {"lon": 12.1, "lat": 2.2}},
    })
    assert (release_tracks[0].lon, release_tracks[0].lat) == (12.1, 2.2)

    with pytest.raises(ValueError, match="missile_type"):
        HMMissileSnapshot(1, 21002)
    with pytest.raises(ValueError, match="baseline_action"):
        HMMissileSnapshot(1, 21000, baseline_action=True)
    with pytest.raises(ValueError, match="unique"):
        HMPublicStrikePackagesSkill(public_targets=(target(1, 9400), target(1, 9400)))
    with pytest.raises(ValueError, match="initial public directory"):
        HMPublicStrikePackagesSkill(public_targets=(target(1, 9400),)).decide(
            step=0,
            members=(missile(10, 21000),),
            track_updates=(target(2, 9400, step=1),),
        )


def test_dead_or_unlaunched_members_emit_no_owner_actions() -> None:
    skill = HMPublicStrikePackagesSkill(public_targets=(target(901, 9400),))
    decision = skill.decide(
        step=0,
        members=(
            missile(10, 21000, alive=False),
            missile(20, 21001, launched=False),
        ),
    )
    assert decision.baseline_actions == ()
    assert decision.attack_actions == ()
    assert materialize_engine_actions(decision) == {}
