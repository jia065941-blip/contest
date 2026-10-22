import pytest

from l_response_package_skill import LResponsePackageSkill, ResponseMemberSnapshot
from target_damage_belief_skill import TargetHealthBelief


def belief(expected=1.2, projected=1.2, confidence=0.0, destroyed=False):
    return TargetHealthBelief(
        168, 9500, 1.2, 0.0, expected, 1.2, projected, confidence, destroyed,
        (), (), (),
    )


def member(entity_id, *, launched=True, delivered=False, alive=True, eta=100.0):
    return ResponseMemberSnapshot(entity_id, 168, alive, launched, delivered, 50, eta, 1)


def test_two_member_package_passes_through_baselines():
    result = LResponsePackageSkill().decide(
        members=(member(1), member(2)), belief=belief()
    )
    assert result.actions == ((1, 1), (2, 1))
    assert result.status.roles[0][1].value == "primary"
    assert result.status.roles[1][1].value == "backup"


def test_confident_destroyed_releases_unlaunched_and_requests_authorized_target():
    result = LResponsePackageSkill().decide(
        members=(member(1), member(2, launched=False), member(3, launched=False)),
        belief=belief(expected=0.0, projected=0.0, confidence=1.0, destroyed=True),
        authorized_fallback_target_ids=(169,),
    )
    assert result.stop_commitment_request.release_member_ids == (2, 3)
    assert result.retarget_request.to_target_id == 169


def test_damage_shortfall_requests_supplement_when_package_has_room():
    result = LResponsePackageSkill().decide(
        members=(member(1, delivered=True), member(2, alive=False)),
        belief=belief(expected=0.4, projected=0.4, confidence=0.4),
    )
    assert result.supplement_request is not None
    assert result.supplement_request.requested_members == 1


@pytest.mark.parametrize("count", [1, 5])
def test_invalid_package_size_is_rejected(count):
    with pytest.raises(ValueError):
        LResponsePackageSkill().decide(
            members=tuple(member(index + 1) for index in range(count)),
            belief=belief(),
        )
