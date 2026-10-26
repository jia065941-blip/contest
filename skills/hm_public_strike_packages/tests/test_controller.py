import pytest

from hm_public_strike_packages_skill import (
    HMMissileSnapshot,
    HMPublicStrikePackagesSkill,
    PublicStrikeConfig,
    PublicTargetTrack,
    expected_effective_damage,
    kill_probability,
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


def test_exact_kill_probability_known_values() -> None:
    assert kill_probability(2, 0, 9600) == pytest.approx(0.84)
    assert kill_probability(3, 0, 9600) == pytest.approx(0.936)
    assert kill_probability(2, 2, 9400) == pytest.approx(0.9472)


def test_covers_every_target_by_expected_damage_not_equal_missile_count() -> None:
    config = PublicStrikeConfig(allocation_mode="expected_damage", safety_factor=1.0)
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
    config = PublicStrikeConfig(allocation_mode="expected_damage", safety_factor=1.0)
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
    config = PublicStrikeConfig(allocation_mode="expected_damage")

    first = HMPublicStrikePackagesSkill(config, public_targets=targets).decide(
        step=0, members=members,
    )
    second = HMPublicStrikePackagesSkill(config, public_targets=reversed(targets)).decide(
        step=0, members=reversed(members),
    )

    assert [(item.target.target_id, item.member_ids) for item in first.packages] == [
        (901, (10,)), (902, (11,)),
    ]
    assert first == second


def test_sentry_minimum_h_is_round_robin_when_inventory_is_short() -> None:
    config = PublicStrikeConfig(
        allocation_mode="kill_probability", min_h_per_sentry=2,
    )
    targets = (target(903, 9600), target(901, 9600), target(902, 9600))
    members = tuple(missile(item, 21000) for item in (13, 10, 12, 11))

    first = HMPublicStrikePackagesSkill(config, public_targets=targets).decide(
        step=0, members=members,
    )
    second = HMPublicStrikePackagesSkill(
        config, public_targets=tuple(reversed(targets)),
    ).decide(step=0, members=tuple(reversed(members)))

    assert [(item.target.target_id, item.member_ids) for item in first.packages] == [
        (901, (10, 13)), (902, (11,)), (903, (12,)),
    ]
    assert first == second


def test_mother_h_cap_redirects_excess_h_and_assigns_all_feasible_inventory() -> None:
    mother = PublicTargetTrack(901, 9400, 0.0, 0.0, 0.0, 15.0, 5.0)
    config = PublicStrikeConfig(
        allocation_mode="kill_probability", max_h_per_mother=2,
    )
    decision = HMPublicStrikePackagesSkill(
        config, public_targets=(mother, target(902, 9600)),
    ).decide(
        step=0,
        members=tuple(missile(item, 21000) for item in (10, 11, 12, 13)),
    )

    packages = {item.target.target_id: item for item in decision.packages}
    assert packages[901].member_ids == (10, 11)
    assert packages[902].member_ids == (12, 13)
    assert decision.unassigned_member_ids == ()
    assert packages[901].kill_probability == pytest.approx(0.96)


def test_reserves_are_not_assigned() -> None:
    config = PublicStrikeConfig(
        allocation_mode="expected_damage",
        reserve_h=1,
        reserve_m=1,
        safety_factor=1.0,
    )
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
    skill = HMPublicStrikePackagesSkill(
        PublicStrikeConfig(allocation_mode="expected_damage"),
        public_targets=(target(901, 9400),),
    )
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


@pytest.mark.parametrize(
    "kwargs",
    (
        {"allocation_mode": "other"},
        {"allocation_mode": 1},
        {"target_kill_probability": 0.0},
        {"target_kill_probability": 1.01},
        {"target_kill_probability": True},
        {"min_h_per_mother": -1},
        {"allocation_mode": "kill_probability", "min_h_per_mother": 3},
        {"min_m_per_mother": True},
        {"min_h_per_sentry": -1},
        {"min_h_per_sentry": True},
        {"max_h_per_mother": -1},
        {"max_h_per_mother": 1.5},
        {"h_per_sentry_package": 0},
        {"h_per_sentry_package": True},
        {"m_per_h_mixed_package": 0},
        {"m_per_h_mixed_package": True},
        {"m_only_sentry_package": 0},
        {"m_only_sentry_package": 1.5},
        {"allow_m_only_sentry_packages": 1},
        {"target_priority_axis": "up"},
        {"m_target_policy": "other"},
        {"m_target_policy": 1},
    ),
)
def test_new_config_fields_are_strictly_validated(kwargs) -> None:
    with pytest.raises(ValueError):
        PublicStrikeConfig(**kwargs)


def test_new_config_defaults_select_full_coverage_mixed() -> None:
    config = PublicStrikeConfig()
    assert config.allocation_mode == "full_coverage_mixed"
    assert config.target_kill_probability == pytest.approx(0.85)
    assert config.min_h_per_mother == 2
    assert config.min_m_per_mother == 0
    assert config.min_h_per_sentry == 1
    assert config.max_h_per_mother == 2
    assert config.h_per_sentry_package == 2
    assert config.m_per_h_mixed_package == 2
    assert config.m_only_sentry_package == 6
    assert not config.allow_m_only_sentry_packages
    assert config.m_target_policy == "complete_h_sentries"


def test_kill_probability_validates_counts_and_package_reports_diagnostic() -> None:
    with pytest.raises(ValueError, match="h_count"):
        kill_probability(True, 0, 9600)
    with pytest.raises(ValueError, match="m_count"):
        kill_probability(0, -1, 9600)

    decision = HMPublicStrikePackagesSkill(
        public_targets=(target(901, 9600),),
    ).decide(
        step=0,
        members=(missile(10, 21000), missile(11, 21000)),
    )
    assert decision.packages[0].kill_probability == pytest.approx(0.84)
    assert decision.packages[0].nominal_applied_damage == pytest.approx(24.0)
    assert decision.packages[0].threshold_satisfied


def test_easy_v12_full_coverage_mixed_covers_every_sentry() -> None:
    targets = tuple(
        [target(900 + index, 9400) for index in range(4)]
        + [target(1000 + index, 9600) for index in range(7)]
    )
    members = tuple(
        [missile(index, 21000) for index in range(17)]
        + [missile(100 + index, 21001) for index in range(24)]
    )

    decision = HMPublicStrikePackagesSkill(
        PublicStrikeConfig(), public_targets=targets,
    ).decide(step=0, members=members)
    missile_types = {item.entity_id: item.missile_type for item in members}
    packages = {item.target.target_id: item for item in decision.packages}

    mother_h = [
        sum(missile_types[item] == 21000 for item in packages[target_id].member_ids)
        for target_id in range(900, 904)
    ]
    sentry_h = [
        sum(
            missile_types[item] == 21000
            for item in packages[target_id].member_ids
        ) if target_id in packages else 0
        for target_id in range(1000, 1007)
    ]
    sentry_m = [
        sum(
            missile_types[item] == 21001
            for item in packages[target_id].member_ids
        ) if target_id in packages else 0
        for target_id in range(1000, 1007)
    ]
    assert mother_h == [2, 2, 2, 2]
    assert sentry_h == [2, 2, 1, 1, 1, 1, 1]
    assert sentry_m == [0, 0, 5, 5, 5, 5, 4]
    assert all(item.threshold_satisfied for item in decision.packages)
    assert [packages[target_id].nominal_applied_damage for target_id in range(1000, 1007)] == [
        24.0, 24.0, 27.0, 27.0, 27.0, 27.0, 24.0,
    ]
    assert decision.unassigned_member_ids == ()


@pytest.mark.parametrize(
    (
        "h_count", "m_count", "mother_count", "sentry_count",
        "expected_sentry_h", "expected_sentry_m",
    ),
    (
        (27, 37, 6, 12, [2] * 3 + [1] * 9, [0] * 3 + [5] + [4] * 8),
        (48, 66, 8, 20, [2] * 12 + [1] * 8, [0] * 12 + [9, 9] + [8] * 6),
    ),
)
def test_medium_and_hard_v12_full_coverage_mixed_cover_every_sentry(
    h_count, m_count, mother_count, sentry_count,
    expected_sentry_h, expected_sentry_m,
) -> None:
    targets = tuple(
        [target(900 + index, 9400) for index in range(mother_count)]
        + [target(1000 + index, 9600) for index in range(sentry_count)]
    )
    members = tuple(
        [missile(index, 21000) for index in range(h_count)]
        + [missile(1000 + index, 21001) for index in range(m_count)]
    )
    decision = HMPublicStrikePackagesSkill(
        PublicStrikeConfig(), public_targets=targets,
    ).decide(step=0, members=members)
    missile_types = {item.entity_id: item.missile_type for item in members}
    packages = {item.target.target_id: item for item in decision.packages}

    assert [
        sum(missile_types[item] == 21000 for item in packages[target_id].member_ids)
        for target_id in range(900, 900 + mother_count)
    ] == [2] * mother_count
    assert [
        sum(missile_types[item] == 21000 for item in packages[target_id].member_ids)
        for target_id in range(1000, 1000 + sentry_count)
    ] == expected_sentry_h
    assert [
        sum(missile_types[item] == 21001 for item in packages[target_id].member_ids)
        for target_id in range(1000, 1000 + sentry_count)
    ] == expected_sentry_m
    assert all(packages[target_id].threshold_satisfied for target_id in packages)
    assert decision.unassigned_member_ids == ()


def test_full_coverage_mixed_uses_m_only_packages_when_h_is_short() -> None:
    targets = tuple(target(1000 + index, 9600) for index in range(4))
    members = tuple(
        [missile(index, 21000) for index in (11, 10)]
        + [missile(100 + index, 21001) for index in range(10)]
    )

    first = HMPublicStrikePackagesSkill(public_targets=targets).decide(
        step=0, members=members,
    )
    second = HMPublicStrikePackagesSkill(
        public_targets=tuple(reversed(targets)),
    ).decide(step=0, members=tuple(reversed(members)))
    missile_types = {item.entity_id: item.missile_type for item in members}
    packages = {item.target.target_id: item for item in first.packages}

    assert first == second
    assert [
        sum(missile_types[item] == 21000 for item in packages[target_id].member_ids)
        if target_id in packages else 0
        for target_id in range(1000, 1004)
    ] == [1, 1, 0, 0]
    assert [
        sum(missile_types[item] == 21001 for item in packages[target_id].member_ids)
        if target_id in packages else 0
        for target_id in range(1000, 1004)
    ] == [2, 2, 6, 0]
    assert sum(item.threshold_satisfied for item in first.packages) == 3
    assert first.unassigned_member_ids == ()


def test_full_coverage_mixed_completes_maximum_packages_when_m_is_short() -> None:
    targets = tuple(target(1000 + index, 9600) for index in range(3))
    members = tuple(
        [missile(index, 21000) for index in range(3)]
        + [missile(100 + index, 21001) for index in range(3)]
    )
    decision = HMPublicStrikePackagesSkill(public_targets=targets).decide(
        step=0, members=members,
    )
    missile_types = {item.entity_id: item.missile_type for item in members}
    packages = {item.target.target_id: item for item in decision.packages}

    assert [
        sum(missile_types[item] == 21001 for item in packages[target_id].member_ids)
        for target_id in range(1000, 1003)
    ] == [2, 1, 0]
    assert [packages[target_id].threshold_satisfied for target_id in range(1000, 1003)] == [
        True, False, False,
    ]
    assert decision.unassigned_member_ids == ()


def test_full_coverage_mixed_prioritizes_far_sentry_for_2h_upgrade() -> None:
    targets = (
        target(1001, 9600, lon=10.0),
        target(1002, 9600, lon=30.0),
        target(1003, 9600, lon=20.0),
    )
    members = tuple(missile(index, 21000) for index in range(4))
    decision = HMPublicStrikePackagesSkill(public_targets=targets).decide(
        step=0, members=members,
    )

    assert {item.target.target_id: len(item.member_ids) for item in decision.packages} == {
        1001: 1, 1002: 2, 1003: 1,
    }


def test_full_coverage_mixed_balances_redundant_m_after_universal_2h() -> None:
    targets = (target(1001, 9600), target(1002, 9600))
    members = tuple(
        [missile(index, 21000) for index in range(4)]
        + [missile(100 + index, 21001) for index in range(3)]
    )
    decision = HMPublicStrikePackagesSkill(public_targets=targets).decide(
        step=0, members=members,
    )
    missile_types = {item.entity_id: item.missile_type for item in members}
    packages = {item.target.target_id: item for item in decision.packages}

    assert [
        sum(missile_types[item] == 21001 for item in packages[target_id].member_ids)
        for target_id in (1001, 1002)
    ] == [2, 1]
    assert decision.unassigned_member_ids == ()


def test_allow_m_only_true_retains_v07_easy_allocation() -> None:
    targets = tuple(
        [target(900 + index, 9400) for index in range(4)]
        + [target(1000 + index, 9600) for index in range(7)]
    )
    members = tuple(
        [missile(index, 21000) for index in range(17)]
        + [missile(100 + index, 21001) for index in range(18)]
    )
    decision = HMPublicStrikePackagesSkill(
        PublicStrikeConfig(
            allocation_mode="deterministic_thresholds",
            allow_m_only_sentry_packages=True,
        ),
        public_targets=targets,
    ).decide(step=0, members=members)
    missile_types = {item.entity_id: item.missile_type for item in members}
    packages = {item.target.target_id: item for item in decision.packages}

    sentry_counts = [
        (
            sum(missile_types[item] == 21000 for item in packages[target_id].member_ids),
            sum(missile_types[item] == 21001 for item in packages[target_id].member_ids),
        )
        for target_id in range(1000, 1007)
    ]
    assert sentry_counts == [
        (2, 0), (2, 0), (2, 0), (2, 0), (1, 2), (0, 6), (0, 10),
    ]
    assert decision.unassigned_member_ids == ()


def test_threshold_balances_m_across_complete_h_sentries() -> None:
    targets = (
        target(901, 9400), target(1001, 9600), target(1002, 9600),
    )
    members = tuple(
        [missile(index, 21000) for index in range(6)]
        + [missile(100 + index, 21001) for index in range(5)]
    )

    decision = HMPublicStrikePackagesSkill(
        PublicStrikeConfig(allocation_mode="threshold_packages"),
        public_targets=targets,
    ).decide(step=0, members=members)
    packages = {item.target.target_id: item for item in decision.packages}

    assert packages[901].member_ids == (0, 1)
    assert packages[1001].member_ids == (2, 3, 100, 102, 104)
    assert packages[1002].member_ids == (4, 5, 101, 103)
    assert decision.unassigned_member_ids == ()


def test_explicit_kill_probability_mode_retains_legacy_easy_allocation() -> None:
    targets = tuple(
        [target(900 + index, 9400) for index in range(4)]
        + [target(1000 + index, 9600) for index in range(7)]
    )
    members = tuple(
        [missile(index, 21000) for index in range(17)]
        + [missile(100 + index, 21001) for index in range(24)]
    )
    decision = HMPublicStrikePackagesSkill(
        PublicStrikeConfig(allocation_mode="kill_probability"),
        public_targets=targets,
    ).decide(step=0, members=members)
    missile_types = {item.entity_id: item.missile_type for item in members}
    packages = {item.target.target_id: item for item in decision.packages}

    mother_counts = [
        (
            sum(missile_types[item] == 21000 for item in packages[target_id].member_ids),
            sum(missile_types[item] == 21001 for item in packages[target_id].member_ids),
        )
        for target_id in range(900, 904)
    ]
    sentry_h = sorted(
        sum(missile_types[item] == 21000 for item in packages[target_id].member_ids)
        for target_id in range(1000, 1007)
    )
    assert mother_counts == [(2, 6)] * 4
    assert sentry_h == [1, 1, 1, 1, 1, 2, 2]
    assert decision.unassigned_member_ids == ()


@pytest.mark.parametrize(
    (
        "h_count", "m_count", "mother_count", "sentry_count", "mother_h_count", "allow_m_only",
        "expected_sentry_h", "expected_sentry_m", "covered_sentries",
    ),
    (
        (
            27, 37, 6, 12, 2, False,
            [2] * 7 + [1] + [0] * 4,
            [5, 5, 5, 5, 4, 4, 4, 5] + [0] * 4,
            8,
        ),
        (
            27, 37, 6, 12, 3, False,
            [2] * 4 + [1] + [0] * 7,
            [8, 7, 7, 7, 8] + [0] * 7,
            5,
        ),
        (
            48, 66, 8, 20, 2, False,
            [2] * 16 + [0] * 4,
            [5, 5] + [4] * 14 + [0] * 4,
            16,
        ),
        (
            48, 66, 8, 20, 3, False,
            [2] * 12 + [0] * 8,
            [6] * 6 + [5] * 6 + [0] * 8,
            12,
        ),
        (
            27, 37, 6, 12, 2, True,
            [2] * 7 + [1] + [0] * 4,
            [0] * 7 + [2, 6, 6, 6, 17],
            12,
        ),
        (
            27, 37, 6, 12, 3, True,
            [2] * 4 + [1] + [0] * 7,
            [0] * 4 + [2, 6, 6, 6, 6, 11, 0, 0],
            10,
        ),
        (
            48, 66, 8, 20, 2, True,
            [2] * 16 + [0] * 4,
            [0] * 16 + [6, 6, 6, 48],
            20,
        ),
        (
            48, 66, 8, 20, 3, True,
            [2] * 12 + [0] * 8,
            [0] * 12 + [6] * 7 + [24],
            20,
        ),
    ),
)
def test_medium_and_hard_use_only_complete_deterministic_thresholds(
    h_count, m_count, mother_count, sentry_count, mother_h_count, allow_m_only,
    expected_sentry_h, expected_sentry_m, covered_sentries,
) -> None:
    targets = tuple(
        [target(900 + index, 9400) for index in range(mother_count)]
        + [target(1000 + index, 9600) for index in range(sentry_count)]
    )
    members = tuple(
        [missile(index, 21000) for index in range(h_count)]
        + [missile(1000 + index, 21001) for index in range(m_count)]
    )
    decision = HMPublicStrikePackagesSkill(
        PublicStrikeConfig(
            allocation_mode="deterministic_thresholds",
            min_h_per_mother=mother_h_count,
            allow_m_only_sentry_packages=allow_m_only,
        ),
        public_targets=targets,
    ).decide(step=0, members=members)
    missile_types = {item.entity_id: item.missile_type for item in members}
    packages = {item.target.target_id: item for item in decision.packages}

    mother_h = [
        sum(missile_types[item] == 21000 for item in packages[target_id].member_ids)
        for target_id in range(900, 900 + mother_count)
    ]
    sentry_h = [
        sum(
            missile_types[item] == 21000
            for item in packages[target_id].member_ids
        ) if target_id in packages else 0
        for target_id in range(1000, 1000 + sentry_count)
    ]
    sentry_m = [
        sum(
            missile_types[item] == 21001
            for item in packages[target_id].member_ids
        ) if target_id in packages else 0
        for target_id in range(1000, 1000 + sentry_count)
    ]
    mother_m = [
        sum(missile_types[item] == 21001 for item in packages[target_id].member_ids)
        for target_id in range(900, 900 + mother_count)
    ]
    assert mother_h == [mother_h_count] * mother_count
    assert mother_m == [0] * mother_count
    assert sentry_h == expected_sentry_h
    assert sentry_m == expected_sentry_m
    if not allow_m_only:
        assert all(
            not m_count or h_count
            for h_count, m_count in zip(sentry_h, sentry_m)
        )
    assert sum(
        item.threshold_satisfied
        for item in decision.packages
        if item.target.target_type == 9600
    ) == covered_sentries
    assert decision.unassigned_member_ids == ()


def test_incomplete_h_sentries_policy_retains_v05_m_targeting() -> None:
    targets = tuple(
        [target(900 + index, 9400) for index in range(4)]
        + [target(1000 + index, 9600) for index in range(7)]
    )
    members = tuple(
        [missile(index, 21000) for index in range(17)]
        + [missile(100 + index, 21001) for index in range(24)]
    )
    decision = HMPublicStrikePackagesSkill(
        PublicStrikeConfig(
            allocation_mode="threshold_packages",
            m_target_policy="incomplete_h_sentries",
        ),
        public_targets=targets,
    ).decide(step=0, members=members)
    missile_types = {item.entity_id: item.missile_type for item in members}
    packages = {item.target.target_id: item for item in decision.packages}

    sentry_m = [
        sum(missile_types[item] == 21001 for item in packages[target_id].member_ids)
        for target_id in range(1000, 1007)
    ]
    assert sentry_m == [0, 0, 0, 0, 8, 8, 8]


def test_mothers_policy_balances_m_across_mothers() -> None:
    targets = (
        target(901, 9400), target(902, 9400), target(1001, 9600),
    )
    members = tuple(
        [missile(index, 21000) for index in range(6)]
        + [missile(100 + index, 21001) for index in range(5)]
    )
    decision = HMPublicStrikePackagesSkill(
        PublicStrikeConfig(
            allocation_mode="threshold_packages", m_target_policy="mothers",
        ),
        public_targets=targets,
    ).decide(step=0, members=members)
    missile_types = {item.entity_id: item.missile_type for item in members}
    packages = {item.target.target_id: item for item in decision.packages}

    mother_m = [
        sum(missile_types[item] == 21001 for item in packages[target_id].member_ids)
        for target_id in (901, 902)
    ]
    assert mother_m == [3, 2]
    assert packages[1001].member_ids == (4, 5)


def test_default_m_policy_falls_back_to_partial_h_sentry_first() -> None:
    targets = (
        target(901, 9400), target(1001, 9600), target(1002, 9600),
    )
    members = (
        missile(10, 21000),
        missile(20, 21001), missile(21, 21001), missile(22, 21001),
    )
    decision = HMPublicStrikePackagesSkill(
        PublicStrikeConfig(
            allocation_mode="threshold_packages", min_h_per_mother=0,
        ),
        public_targets=targets,
    ).decide(step=0, members=members)

    assert [(item.target.target_id, item.member_ids) for item in decision.packages] == [
        (1001, (10, 20, 21, 22)),
    ]


def test_default_m_policy_uses_axis_priority_then_id_for_remainders() -> None:
    targets = (
        target(1001, 9600, lon=-10.0),
        target(1002, 9600, lon=20.0),
    )
    members = tuple(
        [missile(item, 21000) for item in range(10, 14)]
        + [missile(item, 21001) for item in range(20, 23)]
    )
    decision = HMPublicStrikePackagesSkill(
        PublicStrikeConfig(
            allocation_mode="threshold_packages", target_priority_axis="east",
        ),
        public_targets=targets,
    ).decide(step=0, members=members)
    packages = {item.target.target_id: item.member_ids for item in decision.packages}

    assert packages == {
        1001: (12, 13, 21),
        1002: (10, 11, 20, 22),
    }


def test_easy_v13_uses_three_h_mothers_and_does_not_count_redundancy() -> None:
    mother_count = 4
    sentry_count = 7
    targets = tuple(
        [target(900 + index, 9400) for index in range(mother_count)]
        + [target(1000 + index, 9600) for index in range(sentry_count)]
    )
    members = tuple(
        [missile(index, 21000) for index in range(17)]
        + [missile(1000 + index, 21001) for index in range(24)]
    )
    decision = HMPublicStrikePackagesSkill(
        PublicStrikeConfig(
            allocation_mode="deterministic_thresholds", min_h_per_mother=3,
        ),
        public_targets=targets,
    ).decide(step=0, members=members)
    missile_types = {item.entity_id: item.missile_type for item in members}
    packages = {item.target.target_id: item for item in decision.packages}

    mother_h = [
        sum(missile_types[item] == 21000 for item in packages[target_id].member_ids)
        for target_id in range(900, 900 + mother_count)
    ]
    sentry_counts = [
        (
            sum(missile_types[item] == 21000 for item in packages[target_id].member_ids),
            sum(missile_types[item] == 21001 for item in packages[target_id].member_ids),
        )
        if target_id in packages else (0, 0)
        for target_id in range(1000, 1000 + sentry_count)
    ]
    assert mother_h == [3] * mother_count
    assert sentry_counts == [
        (2, 8), (2, 8), (1, 8), (0, 0), (0, 0), (0, 0), (0, 0),
    ]
    assert sum(
        item.threshold_satisfied
        for item in decision.packages
        if item.target.target_type == 9600
    ) == 3
    assert all(target_id not in packages for target_id in range(1003, 1007))
    assert decision.unassigned_member_ids == ()


@pytest.mark.parametrize(
    ("axis", "expected_target_id"),
    (("east", 902), ("west", 901), ("north", 903), ("south", 904)),
)
def test_priority_axis_selects_the_first_complete_sentry_package(
    axis, expected_target_id,
) -> None:
    targets = (
        target(901, 9600, lon=-10.0),
        target(902, 9600, lon=20.0),
        target(903, 9600, lat=30.0),
        target(904, 9600, lat=-40.0),
    )
    decision = HMPublicStrikePackagesSkill(
        PublicStrikeConfig(
            allocation_mode="deterministic_thresholds", target_priority_axis=axis,
        ),
        public_targets=targets,
    ).decide(
        step=0,
        members=(missile(10, 21000), missile(11, 21000)),
    )

    assert len(decision.packages) == 1
    assert decision.packages[0].target.target_id == expected_target_id
    assert decision.packages[0].member_ids == (10, 11)


def test_threshold_shortage_favors_value_after_fair_mother_rounds() -> None:
    low_value = PublicTargetTrack(901, 9400, 0.0, 0.0, 0.0, 24.0, 1.0)
    high_value = PublicTargetTrack(902, 9400, 0.0, 0.0, 0.0, 24.0, 10.0)
    sentry = target(903, 9600)
    members = tuple(
        [missile(item, 21000) for item in (12, 10, 11)]
        + [missile(item, 21001) for item in (21, 20)]
    )

    decision = HMPublicStrikePackagesSkill(
        PublicStrikeConfig(
            allocation_mode="deterministic_thresholds",
            allow_m_only_sentry_packages=True,
        ),
        public_targets=(low_value, sentry, high_value),
    ).decide(step=0, members=members)
    packages = {item.target.target_id: item for item in decision.packages}
    missile_types = {item.entity_id: item.missile_type for item in members}

    assert [
        sum(missile_types[item] == 21000 for item in packages[target_id].member_ids)
        for target_id in (901, 902, 903)
    ] == [1, 2, 0]
    assert packages[903].member_ids == (20, 21)
    assert decision.unassigned_member_ids == ()


def test_threshold_allocation_is_deterministic_for_reversed_inputs() -> None:
    targets = tuple(
        [target(900 + index, 9400) for index in range(2)]
        + [target(1000 + index, 9600, lon=float(index)) for index in range(4)]
    )
    members = tuple(
        [missile(index, 21000) for index in range(9)]
        + [missile(100 + index, 21001) for index in range(11)]
    )
    config = PublicStrikeConfig(target_priority_axis="east")

    first = HMPublicStrikePackagesSkill(config, public_targets=targets).decide(
        step=0, members=members,
    )
    second = HMPublicStrikePackagesSkill(
        config, public_targets=tuple(reversed(targets)),
    ).decide(step=0, members=tuple(reversed(members)))

    assert first == second


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
