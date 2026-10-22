import pytest

from target_damage_belief_skill import (
    DeliveryEvidence,
    HitReceipt,
    TargetDamageBeliefSkill,
    TargetPriorEvidence,
)


def skill_with_target(target_id=52, target_type=9400, health=24.0):
    skill = TargetDamageBeliefSkill()
    skill.register_target(TargetPriorEvidence(target_id, target_type, health))
    return skill


def test_commitment_changes_projected_but_not_current_health():
    skill = skill_with_target()
    skill.commit_missile(missile_id=1, missile_type=21000, target_id=52)
    belief = skill.estimate(52)
    assert belief.expected_hp == 24.0
    assert belief.projected_expected_hp == 8.0


def test_delivery_creates_interval_not_false_exact_hit():
    skill = skill_with_target()
    skill.report_delivery(
        DeliveryEvidence(1, 21000, 52, 100, 4, 800.0)
    )
    belief = skill.estimate(52)
    assert belief.lower_hp <= belief.expected_hp <= belief.upper_hp
    assert belief.expected_hp == 24.0 - 16.0 * 0.85
    assert belief.upper_hp == 24.0
    assert belief.confirmed_destroyed is False


def test_confirmed_receipt_replaces_delivery_estimate():
    skill = skill_with_target()
    skill.report_delivery(
        DeliveryEvidence(1, 21000, 52, 100, 4, 50.0)
    )
    skill.report_hit_receipt(HitReceipt(1, 21000, 52, 101, 19.2))
    belief = skill.estimate(52)
    assert belief.lower_hp == pytest.approx(4.8)
    assert belief.expected_hp == pytest.approx(4.8)
    assert belief.upper_hp == pytest.approx(4.8)
    assert belief.receipt_missiles == (1,)


def test_hidden_target_uses_caller_supplied_prior():
    skill = skill_with_target(168, 9500, 1.2)
    skill.report_delivery(
        DeliveryEvidence(7, 21002, 168, 200, 4, 80.0)
    )
    belief = skill.estimate(168)
    assert belief.expected_hp < 1.2
    assert belief.confirmed_destroyed is False


def test_confirmed_destroyed_collapses_interval():
    skill = skill_with_target()
    skill.confirm_destroyed(52)
    belief = skill.estimate(52)
    assert belief.lower_hp == belief.expected_hp == belief.upper_hp == 0.0
    assert belief.confidence == 1.0


def test_retarget_transfer_removes_old_commitment_atomically():
    skill = TargetDamageBeliefSkill()
    skill.register_target(TargetPriorEvidence(52, 9400, 24.0))
    skill.register_target(TargetPriorEvidence(53, 9400, 24.0))
    skill.commit_missile(missile_id=1, missile_type=21000, target_id=52)
    skill.transfer_missile(1, 53)
    assert skill.estimate(52).committed_missiles == ()
    assert skill.estimate(53).committed_missiles == (1,)
    skill.transfer_missile(1, 53)
    assert skill.estimate(53).committed_missiles == (1,)


def test_duplicate_delivery_event_is_idempotent():
    skill = skill_with_target()
    evidence = DeliveryEvidence(
        1, 21000, 52, 100, 4, 100.0, event_id="delivery:1"
    )
    skill.report_delivery(evidence)
    first = skill.estimate(52)
    skill.report_delivery(evidence)
    second = skill.estimate(52)
    assert first == second
