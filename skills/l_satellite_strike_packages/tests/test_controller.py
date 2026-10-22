import pytest

from l_satellite_strike_packages_skill import (
    LMissileSnapshot,
    LSatelliteStrikePackagesSkill,
    SatelliteStrikeConfig,
    materialize_engine_actions,
    tracks_from_isolated_observations,
)


def observation(entity_id, tracks=()):
    return {
        "step": 2,
        "self": {
            "entity_id": entity_id,
            "type": 21002,
            "detectInfo": {
                target_id: {
                    "entity_id": target_id,
                    "entity_type": target_type,
                    "time": observed_step,
                    "lla": {"x": lon, "y": lat, "z": 0},
                }
                for target_id, target_type, lon, lat, observed_step in tracks
            },
        },
    }


def members(count):
    return tuple(LMissileSnapshot(100 + index) for index in range(count))


def test_requests_satellite_once_and_preserves_local_actions() -> None:
    skill = LSatelliteStrikePackagesSkill()
    local_members = (LMissileSnapshot(100, baseline_action=0), LMissileSnapshot(101, baseline_action=2))

    first = skill.decide(
        step=1, members=local_members,
    )
    second = skill.decide(
        step=2, members=local_members,
    )

    assert first.baseline_actions == ((100, 0), (101, 2))
    assert first.satellite_request is not None
    assert first.satellite_request.requester_id == 100
    assert second.satellite_request is None


def test_equal_allocation_is_deterministic_and_targets_only_hidden_nests() -> None:
    skill = LSatelliteStrikePackagesSkill()
    tracks = (
        (903, 9500, 12.3, 2.3, 2),
        (901, 9500, 12.1, 2.1, 2),
        (800, 9600, 12.0, 2.0, 2),
    )

    decision = skill.decide(
        step=2,
        members=members(5),
        tracks=tracks_from_isolated_observations({100: observation(100, tracks)}, step=2),
    )

    assert [package.target.target_id for package in decision.packages] == [901, 903]
    assert [package.member_ids for package in decision.packages] == [(100, 101, 102), (103, 104)]
    assert decision.unassigned_member_ids == ()
    assert tuple((item.entity_id, item.target_id) for item in decision.attack_actions) == (
        (100, 901), (101, 901), (102, 901), (103, 903), (104, 903),
    )


def test_waits_for_expected_target_count_before_allocating() -> None:
    skill = LSatelliteStrikePackagesSkill(SatelliteStrikeConfig(expected_target_count=2))
    first = skill.decide(
        step=2,
        members=members(4),
        tracks=tracks_from_isolated_observations(
            {100: observation(100, ((901, 9500, 12, 2, 2),))}, step=2,
        ),
    )
    second = skill.decide(
        step=3,
        members=members(4),
        tracks=tracks_from_isolated_observations(
            {101: observation(101, ((902, 9500, 13, 2, 3),))}, step=3,
        ),
    )

    assert first.packages == ()
    assert first.unassigned_member_ids == (100, 101, 102, 103)
    assert [package.member_ids for package in second.packages] == [(100, 101), (102, 103)]


def test_fixed_size_and_reserve_leave_members_unassigned() -> None:
    skill = LSatelliteStrikePackagesSkill(SatelliteStrikeConfig(
        allocation_mode="fixed_size", package_size=2, reserve_count=1,
    ))
    tracks = ((901, 9500, 12, 2, 2), (902, 9500, 13, 2, 2))

    decision = skill.decide(
        step=2, members=members(6),
        tracks=tracks_from_isolated_observations({100: observation(100, tracks)}, step=2),
    )

    assert [package.member_ids for package in decision.packages] == [(100, 101), (102, 103)]
    assert decision.unassigned_member_ids == (104, 105)


def test_locked_assignment_ignores_later_tracks() -> None:
    skill = LSatelliteStrikePackagesSkill()
    first = skill.decide(
        step=2, members=members(2),
        tracks=tracks_from_isolated_observations(
            {100: observation(100, ((901, 9500, 12, 2, 2),))}, step=2,
        ),
    )
    second = skill.decide(
        step=3, members=members(2),
        tracks=tracks_from_isolated_observations(
            {100: observation(100, ((902, 9500, 13, 2, 3),))}, step=3,
        ),
    )

    assert first.packages == second.packages
    assert not second.allocation_changed
    assert second.attack_actions == ()
    assert [track.target_id for track in second.discovered_targets] == [901, 902]


def test_unlocked_assignment_rebalances_after_member_loss() -> None:
    skill = LSatelliteStrikePackagesSkill(SatelliteStrikeConfig(
        lock_assignments=False, reassign_lost_members=True,
    ))
    tracks = ((901, 9500, 12, 2, 2), (902, 9500, 13, 2, 2))
    skill.decide(
        step=2, members=members(4),
        tracks=tracks_from_isolated_observations({100: observation(100, tracks)}, step=2),
    )
    changed = skill.decide(
        step=3,
        members=(LMissileSnapshot(100, False), *members(4)[1:]),
        tracks=(),
    )

    assert changed.allocation_changed
    assert [package.member_ids for package in changed.packages] == [(101, 102), (103,)]


def test_rejects_invalid_local_action_and_invalid_config() -> None:
    with pytest.raises(ValueError, match="baseline_action"):
        LMissileSnapshot(100, baseline_action=4)
    with pytest.raises(ValueError, match="requires package_size"):
        SatelliteStrikeConfig(allocation_mode="fixed_size")


def test_observation_adapter_rejects_non_l_observations() -> None:
    invalid = observation(100)
    invalid["self"]["type"] = 21000
    with pytest.raises(ValueError, match="Only isolated L"):
        tracks_from_isolated_observations({100: invalid}, step=1)


def test_runtime_adapter_routes_only_owner_commands() -> None:
    skill = LSatelliteStrikePackagesSkill()
    tracks = tracks_from_isolated_observations(
        {100: observation(100, ((901, 9500, 12, 2, 1),))}, step=1,
    )
    decision = skill.decide(
        step=1,
        members=(LMissileSnapshot(100, baseline_action=0), LMissileSnapshot(101, baseline_action=2)),
        tracks=tracks,
    )

    routed = materialize_engine_actions(decision)

    assert routed[100] == (
        (0.0, 100.0, -1.0, 0.0),
        (3.0, 100.0, 0.0, 0.0),
        (4.0, 100.0, 901.0, 0.0),
    )
    assert routed[101] == (
        (0.0, 101.0, 1.0, 0.0),
        (4.0, 101.0, 901.0, 0.0),
    )
    assert all(int(row[1]) == entity_id for entity_id, rows in routed.items() for row in rows)


def test_dead_and_unlaunched_members_emit_no_actions() -> None:
    skill = LSatelliteStrikePackagesSkill()
    decision = skill.decide(
        step=1,
        members=(
            LMissileSnapshot(100, alive=False, baseline_action=0),
            LMissileSnapshot(101, launched=False, baseline_action=2),
        ),
    )

    assert decision.baseline_actions == ()
    assert decision.satellite_request is None
    assert materialize_engine_actions(decision) == {}


def test_reset_allows_a_new_satellite_request_and_allocation() -> None:
    skill = LSatelliteStrikePackagesSkill()
    tracks = tracks_from_isolated_observations(
        {100: observation(100, ((901, 9500, 12, 2, 1),))}, step=1,
    )
    skill.decide(step=1, members=members(2), tracks=tracks)

    skill.reset()
    fresh = skill.decide(step=1, members=members(2))

    assert fresh.satellite_request is not None
    assert fresh.packages == ()
    assert fresh.discovered_targets == ()


def test_duplicate_member_ids_are_rejected() -> None:
    skill = LSatelliteStrikePackagesSkill()
    with pytest.raises(ValueError, match="unique"):
        skill.decide(step=1, members=(LMissileSnapshot(100), LMissileSnapshot(100)))
