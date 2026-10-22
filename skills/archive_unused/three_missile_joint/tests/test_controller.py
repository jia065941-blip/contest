import pytest

from three_missile_joint_skill import (
    HStrikePackageSkill,
    JointMemberSnapshot,
    JointMissionContext,
    JointRole,
    JointThreat,
    ThreeMissileJointConfig,
    ThreeMissileJointSkill,
    Vector2,
)


def member(entity_id, north, north_velocity=0.0, action=1, threats=()):
    return JointMemberSnapshot(
        entity_id=entity_id,
        missile_type=21000,
        target_id=52,
        position=Vector2(0.0, north),
        velocity=Vector2(2000.0, north_velocity),
        goal_position=Vector2(100000.0, 0.0),
        threats=tuple(threats),
        baseline_action=action,
    )


def test_safe_group_passes_through_v13_actions():
    skill = ThreeMissileJointSkill()
    decision = skill.decide(
        10,
        (member(1, -1000.0, action=0), member(2, 0.0), member(3, 1000.0, action=2)),
    )
    assert decision.actions == (0, 1, 2)
    assert decision.overridden_entity_id is None
    assert decision.reason == "v13_passthrough"
    assert dict(decision.status.roles) == {
        1: JointRole.DECOY,
        2: JointRole.PRIMARY,
        3: JointRole.BACKUP,
    }
    assert decision.status.preferred_actions == ((1, 0), (2, 1), (3, 2))


def test_hard_crossing_changes_at_most_one_non_owner_action():
    skill = ThreeMissileJointSkill()
    decision = skill.decide(
        20,
        (
            member(1, -200.0, north_velocity=100.0),
            member(2, 200.0, north_velocity=-100.0),
            member(3, 1000.0),
        ),
    )
    changed = sum(
        final != baseline
        for final, baseline in zip(decision.actions, decision.baseline_actions, strict=True)
    )
    assert changed == 1
    assert decision.reason == "hard_peer_deconfliction"
    assert decision.selected_min_separation_metres > decision.baseline_min_separation_metres


def test_experimental_profile_relaxes_but_keeps_override_budget():
    config = ThreeMissileJointConfig.experimental()
    assert config.peer_horizon_seconds == 6.0
    assert config.peer_trigger_metres == 1000.0
    assert config.minimum_separation_improvement_metres == 100.0
    assert config.maximum_overrides_per_window == 2


def test_mission_aware_requires_peer_closing():
    skill = HStrikePackageSkill(ThreeMissileJointConfig.mission_aware())
    decision = skill.decide(
        60,
        (
            member(1, 0.0, north_velocity=20.0),
            member(2, 300.0, north_velocity=20.0),
        ),
    )
    assert decision.actions == decision.baseline_actions
    assert decision.overridden_entity_id is None


def test_mission_aware_protects_primary_and_disallows_direct_reversal():
    skill = HStrikePackageSkill(ThreeMissileJointConfig.mission_aware())
    decision = skill.decide(
        61,
        (
            member(1, -200.0, north_velocity=100.0, action=0),
            member(2, 200.0, north_velocity=-100.0, action=2),
        ),
    )
    assert decision.overridden_entity_id != 1
    if decision.overridden_entity_id == 2:
        assert decision.actions[1] == 1
        assert decision.actions[1] != 0


def test_mission_aware_profile_has_goal_and_eta_guards():
    config = ThreeMissileJointConfig.mission_aware()
    assert config.protect_primary is True
    assert config.allow_direct_reversal is False
    assert config.maximum_added_goal_error_degrees == 3.0
    assert config.maximum_eta_penalty_seconds == 1.5


def test_immediate_threat_protects_owner_from_peer_override():
    threat = JointThreat(
        threat_id=900,
        position=Vector2(4000.0, -100.0),
        velocity=Vector2(-2000.0, 50.0),
    )
    skill = ThreeMissileJointSkill()
    decision = skill.decide(
        30,
        (
            member(1, -100.0, north_velocity=50.0, threats=(threat,)),
            member(2, 100.0, north_velocity=-50.0),
            member(3, 1000.0),
        ),
    )
    if decision.overridden_entity_id is not None:
        assert decision.overridden_entity_id != 1


def test_confirmed_destroyed_target_produces_request_not_direct_retarget():
    decision = ThreeMissileJointSkill().decide(
        40,
        (member(1, -1000.0), member(2, 0.0), member(3, 1000.0)),
        JointMissionContext(
            target_expected_hp=0.0,
            target_confidence=1.0,
            target_confirmed_destroyed=True,
            authorized_fallback_target_ids=(53, 2552),
        ),
    )
    assert decision.actions == decision.baseline_actions
    assert decision.retarget_request is not None
    assert decision.retarget_request.from_target_id == 52
    assert decision.retarget_request.to_target_id == 53
    assert decision.retarget_request.member_ids == (1, 2, 3)


def test_unconfirmed_health_belief_does_not_request_retarget():
    decision = ThreeMissileJointSkill().decide(
        41,
        (member(1, -1000.0), member(2, 0.0), member(3, 1000.0)),
        JointMissionContext(
            target_expected_hp=0.0,
            target_confidence=0.5,
            authorized_fallback_target_ids=(53,),
        ),
    )
    assert decision.retarget_request is None


def test_dead_member_degrades_roles_without_replacement():
    third = JointMemberSnapshot(**{**member(3, 1000.0).__dict__, "alive": False})
    decision = ThreeMissileJointSkill().decide(
        42, (member(1, -1000.0), member(2, 0.0), third)
    )
    assert dict(decision.status.roles) == {
        1: JointRole.PRIMARY,
        2: JointRole.BACKUP,
        3: JointRole.LOST,
    }


@pytest.mark.parametrize("count", [2, 3, 4, 5, 6])
def test_variable_h_package_accepts_two_to_six_members(count):
    members = tuple(member(index + 1, index * 1000.0) for index in range(count))
    decision = HStrikePackageSkill().decide(50, members)
    assert len(decision.actions) == count
    assert len(decision.status.roles) == count


@pytest.mark.parametrize("count", [1, 7])
def test_variable_h_package_rejects_single_or_oversized_group(count):
    members = tuple(member(index + 1, index * 1000.0) for index in range(count))
    with pytest.raises(ValueError):
        HStrikePackageSkill().decide(51, members)


def test_four_member_roles_include_secondary():
    members = tuple(member(index + 1, index * 1000.0) for index in range(4))
    roles = dict(HStrikePackageSkill().decide(52, members).status.roles)
    assert roles == {
        1: JointRole.DECOY,
        2: JointRole.PRIMARY,
        3: JointRole.SECONDARY,
        4: JointRole.BACKUP,
    }


@pytest.mark.parametrize(
    "members",
    [
        (member(1, 0.0), member(2, 100.0)),
        (member(1, 0.0), member(1, 100.0), member(3, 200.0)),
        (
            member(1, 0.0),
            member(2, 100.0),
            JointMemberSnapshot(
                **{**member(3, 200.0).__dict__, "missile_type": 21001}
            ),
        ),
        (
            member(1, 0.0),
            member(2, 100.0),
            JointMemberSnapshot(
                **{**member(3, 200.0).__dict__, "target_id": 53}
            ),
        ),
    ],
)
def test_invalid_group_is_rejected(members):
    with pytest.raises(ValueError):
        ThreeMissileJointSkill().decide(0, members)
