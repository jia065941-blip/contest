import pytest

from target_damage_probability_skill import (
    CommitmentEvidence,
    DamageInterval,
    DeliveryEvidence,
    HitReceipt,
    TargetDamageProbabilitySkill,
    TargetPriorEvidence,
)


def tracker_with_target(target_id=52, target_type=9400, health=24.0):
    tracker = TargetDamageProbabilitySkill()
    tracker.register(TargetPriorEvidence(target_id, target_type, health))
    return tracker


def delivery(
    missile_id,
    *,
    stage=3,
    distance=50.0,
    event_id=None,
    probability=None,
    satellite_active=False,
    adjusted_damage_override=None,
):
    return DeliveryEvidence(
        missile_id,
        21000,
        52,
        100,
        stage,
        distance,
        event_id=event_id,
        probability=probability,
        satellite_active=satellite_active,
        adjusted_damage_override=adjusted_damage_override,
    )


def test_two_h_mother_at_100m_has_exact_convolved_probability():
    tracker = tracker_with_target()
    tracker.report_delivery(delivery(1))
    tracker.report_delivery(delivery(2))
    belief = tracker.estimate(52)
    assert belief.destruction_probability == pytest.approx(0.9604)
    assert belief.projected_destruction_probability == pytest.approx(0.9604)
    assert belief.lower_expected_damage == pytest.approx(18.816)
    assert belief.nominal_expected_damage == pytest.approx(31.36)
    assert belief.upper_expected_damage == pytest.approx(37.632)
    assert belief.lower_expected_damage < belief.initial_health
    assert belief.nominal_expected_damage >= belief.initial_health


def test_three_h_mother_lower_expected_damage_reaches_threshold():
    tracker = tracker_with_target()
    for missile_id in (1, 2, 3):
        tracker.report_delivery(delivery(missile_id))

    belief = tracker.estimate(52)
    assert belief.lower_expected_damage == pytest.approx(28.224)
    assert belief.lower_expected_damage >= belief.initial_health


def test_two_h_stage_four_has_exact_convolved_probability():
    tracker = tracker_with_target()
    tracker.report_delivery(delivery(1, stage=4, distance=800.0))
    tracker.report_delivery(delivery(2, stage=4, distance=800.0))
    assert tracker.estimate(52).destruction_probability == pytest.approx(0.7225)


def test_single_h_cannot_destroy_mother():
    tracker = tracker_with_target()
    tracker.report_delivery(delivery(1))
    assert tracker.estimate(52).destruction_probability == 0.0


def test_two_h_destroy_sentinel_with_same_delivery_probability():
    tracker = tracker_with_target(target_id=60, target_type=9600, health=24.0)
    for missile_id in (1, 2):
        tracker.report_delivery(
            DeliveryEvidence(missile_id, 21000, 60, 100, 3, 50.0)
        )
    assert tracker.estimate(60).destruction_probability == pytest.approx(0.9604)


def test_h_mother_reaches_twenty_only_with_explicit_satellite_evidence():
    tracker = tracker_with_target(health=20.0)
    tracker.report_delivery(delivery(1, probability=1.0))
    belief = tracker.estimate(52)
    assert belief.destruction_probability == 0.0
    assert belief.upper_expected_damage == 19.2

    satellite_tracker = tracker_with_target(health=20.0)
    satellite_tracker.report_delivery(
        delivery(1, probability=1.0, satellite_active=True)
    )
    belief = satellite_tracker.estimate(52)
    assert (
        belief.lower_expected_damage,
        belief.nominal_expected_damage,
        belief.upper_expected_damage,
    ) == (20.0, 20.0, 20.0)
    assert belief.destruction_probability == 1.0


def test_strict_damage_table_rejects_invalid_pairs():
    tracker = tracker_with_target(health=20.0)
    with pytest.raises(ValueError, match="unsupported missile/target"):
        tracker.commit(CommitmentEvidence(2, 21002, 52))


@pytest.mark.parametrize(
    ("missile_types", "expected_damage"),
    [
        ((21000, 21000), 24.0),
        ((21000, 21001, 21001), 18.0),
        ((21001,) * 6, 18.0),
    ],
)
def test_sentry_damage_is_fixed(missile_types, expected_damage):
    tracker = tracker_with_target(target_id=60, target_type=9600, health=100.0)
    for missile_id, missile_type in enumerate(missile_types, start=1):
        tracker.report_delivery(
            DeliveryEvidence(
                missile_id,
                missile_type,
                60,
                100,
                3,
                50.0,
                probability=1.0,
            )
        )

    belief = tracker.estimate(60)
    assert belief.lower_expected_damage == expected_damage
    assert belief.nominal_expected_damage == expected_damage
    assert belief.upper_expected_damage == expected_damage


def test_adjusted_damage_override_is_exact():
    tracker = tracker_with_target(health=17.5)
    tracker.report_delivery(
        delivery(1, probability=1.0, adjusted_damage_override=17.5)
    )

    belief = tracker.estimate(52)
    assert (
        belief.lower_expected_damage,
        belief.nominal_expected_damage,
        belief.upper_expected_damage,
    ) == (17.5, 17.5, 17.5)
    assert belief.destruction_probability == 1.0


def test_commitments_affect_only_projected_probability():
    tracker = TargetDamageProbabilitySkill(
        core_commitment_probability=0.9,
        flexible_commitment_probability=0.4,
    )
    tracker.register(TargetPriorEvidence(52, 9400, 24.0))
    tracker.commit(CommitmentEvidence(1, 21000, 52, "core"))
    tracker.commit(CommitmentEvidence(2, 21000, 52, "flexible"))
    belief = tracker.estimate(52)
    assert belief.destruction_probability == 0.0
    assert belief.projected_destruction_probability == pytest.approx(0.36)
    assert belief.confirmed_destroyed is False
    assert belief.core_committed_missiles == (1,)
    assert belief.flexible_committed_missiles == (2,)


def test_probability_override_and_alive_fallback():
    tracker = tracker_with_target()
    tracker.report_delivery(delivery(1, distance=5000.0, probability=1.0))
    tracker.report_delivery(
        DeliveryEvidence(2, 21000, 52, 100, 3, 5000.0, disappeared=False)
    )
    tracker.report_delivery(delivery(3, distance=5000.0))
    belief = tracker.estimate(52)
    assert belief.delivery_probabilities == ((1, 1.0), (2, 0.0), (3, 0.15))


def test_exact_receipt_replaces_delivery_probability():
    tracker = tracker_with_target()
    tracker.report_delivery(delivery(1, probability=0.0))
    tracker.report_receipt(HitReceipt(1, 21000, 52, 101, 24.0))
    belief = tracker.estimate(52)
    assert belief.destruction_probability == 1.0
    assert belief.exact_received_damage == 24.0
    assert belief.delivery_probabilities == ()
    assert belief.lower_expected_damage == 24.0
    assert belief.nominal_expected_damage == 24.0
    assert belief.upper_expected_damage == 24.0


def test_damage_interval_validates_order():
    assert DamageInterval(2.4, 4.0, 4.8).nominal == 4.0
    with pytest.raises(ValueError, match="lower <= nominal <= upper"):
        DamageInterval(4.0, 2.4, 4.8)


def test_duplicate_event_is_idempotent_but_collision_is_rejected():
    tracker = tracker_with_target()
    evidence = delivery(1, event_id="delivery:1")
    tracker.report_delivery(evidence)
    before = tracker.export()
    tracker.report_delivery(evidence)
    assert tracker.export() == before
    with pytest.raises(ValueError, match="reused with new payload"):
        tracker.report_delivery(delivery(2, event_id="delivery:1"))


def test_transfer_release_and_confirm_are_idempotent_events():
    tracker = TargetDamageProbabilitySkill()
    tracker.register(TargetPriorEvidence(52, 9400, 24.0))
    tracker.register(TargetPriorEvidence(53, 9400, 24.0))
    tracker.commit(CommitmentEvidence(1, 21000, 52))
    tracker.transfer(1, 53, event_id="transfer:1")
    tracker.transfer(1, 53, event_id="transfer:1")
    assert tracker.estimate(52).committed_missiles == ()
    assert tracker.estimate(53).committed_missiles == (1,)
    tracker.release(1, event_id="release:1")
    tracker.release(1, event_id="release:1")
    tracker.confirm(53, event_id="confirm:53")
    tracker.confirm(53, event_id="confirm:53")
    assert tracker.estimate(53).confirmed_destroyed is True


def test_export_and_replay_are_equivalent_and_preserve_event_ids():
    tracker = tracker_with_target()
    tracker.report_delivery(delivery(1, event_id="delivery:1"))
    tracker.commit(CommitmentEvidence(2, 21000, 52, "flexible", "commit:2"))
    snapshot = tracker.export()

    restored = TargetDamageProbabilitySkill.from_snapshot(snapshot)
    assert restored.snapshot() == tracker.snapshot()
    assert restored.export() == snapshot
    restored.report_delivery(delivery(1, event_id="delivery:1"))
    assert restored.export() == snapshot

    replayed = TargetDamageProbabilitySkill()
    replayed.replay(snapshot)
    assert replayed.snapshot() == tracker.snapshot()
    assert replayed.export() == snapshot


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TargetPriorEvidence(1, 9400, 0.0),
        lambda: CommitmentEvidence(1, 21000, 1, "reserve"),
        lambda: DeliveryEvidence(1, 21000, 1, 0, 0, -1.0),
        lambda: DeliveryEvidence(1, 21000, 1, 0, 0, None, probability=1.1),
        lambda: DeliveryEvidence(1, 21000, 1, 0, 0, None, satellite_active=1),
        lambda: DeliveryEvidence(
            1, 21000, 1, 0, 0, None, adjusted_damage_override=-1.0
        ),
        lambda: HitReceipt(1, 21000, 1, 0, float("nan")),
    ],
)
def test_invalid_evidence_is_rejected(factory):
    with pytest.raises(ValueError):
        factory()
