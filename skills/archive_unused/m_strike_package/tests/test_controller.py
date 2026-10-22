import pytest

from m_strike_package_skill import (
    MMemberSnapshot,
    MPackageRole,
    MStrikeMissionContext,
    MStrikePackageSkill,
)
from target_damage_belief_skill import TargetHealthBelief


def belief(target=54, target_type=9600, expected=18.0, projected=18.0, confidence=0.0, destroyed=False):
    return TargetHealthBelief(
        target, target_type, 18.0, 0.0, expected, 18.0, projected,
        confidence, destroyed, (), (), (),
    )


def member(entity_id, *, target=54, target_type=9600, release=10, alive=True, launched=True, delivered=False):
    return MMemberSnapshot(
        entity_id, 21001, target, target_type, release,
        alive, launched, delivered, 1,
    )


def context(value, *, step=100, released=10, timeout=700, fallback=()):
    return MStrikeMissionContext(value, step, released, timeout, tuple(fallback))


@pytest.mark.parametrize("count", [2, 3, 4, 5, 6])
def test_accepts_two_to_six_and_never_overrides_motion(count):
    members = tuple(member(index + 1) for index in range(count))
    result = MStrikePackageSkill().decide(
        members=members, context=context(belief())
    )
    assert result.motion_overrides == ()
    assert len(result.status.roles) == count


@pytest.mark.parametrize("count", [1, 7])
def test_rejects_invalid_size(count):
    with pytest.raises(ValueError):
        MStrikePackageSkill().decide(
            members=tuple(member(index + 1) for index in range(count)),
            context=context(belief()),
        )


def test_pair_roles_are_primary_and_support():
    result = MStrikePackageSkill().decide(
        members=(member(1), member(2)), context=context(belief())
    )
    assert result.status.roles == (
        (1, MPackageRole.PRIMARY),
        (2, MPackageRole.SUPPORT),
    )


def test_projected_satisfaction_stops_only_unlaunched_members():
    members = (
        member(1, launched=True),
        member(2, launched=False),
        member(3, launched=False),
    )
    result = MStrikePackageSkill().decide(
        members=members,
        context=context(belief(expected=8.0, projected=0.0, confidence=0.8)),
    )
    assert result.stop_request.member_ids == (3,)
    assert result.retarget_request is None


def test_delivered_satisfaction_can_request_retarget_for_live_inflight_member():
    result = MStrikePackageSkill().decide(
        members=(member(1), member(2, delivered=True)),
        context=context(
            belief(expected=0.0, projected=0.0, confidence=0.9),
            fallback=(106,),
        ),
    )
    assert result.retarget_request.member_ids == (1,)
    assert result.retarget_request.authorized_target_ids == (106,)


def test_supplement_waits_for_timeout_and_uses_target_specific_lower_damage():
    package = MStrikePackageSkill()
    members = (member(1, target=52, target_type=9400), member(2, target=52, target_type=9400))
    value = belief(52, 9400, expected=8.0, projected=8.0)
    before = package.decide(
        members=members, context=context(value, step=709, released=10, timeout=700)
    )
    assert before.supplement_request is None
    after = package.decide(
        members=members, context=context(value, step=710, released=10, timeout=700)
    )
    assert after.supplement_request.full_unmet_count == 4
    assert after.supplement_request.recommended_grant_count == 2


def test_rejects_hidden_target_or_release_span_over_ten():
    with pytest.raises(ValueError):
        MStrikePackageSkill().decide(
            members=(member(1, target=168, target_type=9500), member(2, target=168, target_type=9500)),
            context=context(belief(168, 9500)),
        )
    with pytest.raises(ValueError):
        MStrikePackageSkill().decide(
            members=(member(1, release=0), member(2, release=11)),
            context=context(belief()),
        )
