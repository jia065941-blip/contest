from mission_replan_skill import (
    MissionReplanConfig,
    MissionReplanSkill,
    MissionTargetBelief,
    ResponsePackagePlan,
)


def package(package_id, index, sequence, *members):
    return ResponsePackagePlan(package_id, index, sequence, tuple(members))


def belief(index, target, hp=1.2, confidence=0.0, destroyed=False):
    return MissionTargetBelief(index, target, hp, hp, confidence, destroyed)


def skill(timeout=10):
    return MissionReplanSkill(
        MissionReplanConfig(2, release_delay_steps=20, delivery_timeout_steps=timeout),
        (
            package("a0", 0, 0, 1, 2, 3),
            package("a1", 0, 1, 4, 5),
            package("b0", 1, 0, 6, 7, 8),
            package("b1", 1, 1, 9, 10),
        ),
    )


def test_first_package_per_target_releases_after_discovery_delay():
    value = skill()
    value.update_discoveries(step=31, target_ids=(100, 101))
    assert value.decide(step=50, beliefs=(belief(0, 100), belief(1, 101))).launch_member_ids == ()
    decision = value.decide(step=51, beliefs=(belief(0, 100), belief(1, 101)))
    assert decision.launch_package_ids == ("a0", "b0")
    assert decision.launch_member_ids == (1, 2, 3, 6, 7, 8)


def test_timeout_releases_next_package_only_for_unsatisfied_target():
    value = skill(timeout=10)
    value.update_discoveries(step=0, target_ids=(100, 101))
    value.decide(step=20, beliefs=(belief(0, 100), belief(1, 101)))
    decision = value.decide(
        step=30,
        beliefs=(belief(0, 100, hp=0.4), belief(1, 101, hp=0.0, confidence=0.9)),
    )
    assert decision.launch_package_ids == ("a1",)
    assert decision.cancelled_package_ids == ()


def test_only_high_confidence_overkill_cancels_unreleased_package():
    value = skill(timeout=10)
    value.update_discoveries(step=0, target_ids=(100, 101))
    value.decide(step=20, beliefs=(belief(0, 100), belief(1, 101)))
    satisfied = MissionTargetBelief(
        1,
        101,
        0.0,
        0.0,
        0.95,
        False,
        initial_health=1.2,
        delivered_expected_damage=1.6,
        confidence_source="heuristic_overkill_margin",
    )
    decision = value.decide(
        step=30,
        beliefs=(belief(0, 100, hp=0.4), satisfied),
    )
    assert decision.cancelled_package_ids == ("b1",)


def test_confirmed_destroyed_cancels_all_unreleased_packages():
    value = skill()
    value.update_discoveries(step=0, target_ids=(100, 101))
    decision = value.decide(
        step=20,
        beliefs=(belief(0, 100, destroyed=True), belief(1, 101)),
    )
    assert decision.cancelled_package_ids == ("a0", "a1")
    assert decision.launch_package_ids == ("b0",)


def test_reset_clears_release_and_discovery_state():
    value = skill()
    value.update_discoveries(step=0, target_ids=(100, 101))
    value.decide(step=20, beliefs=(belief(0, 100), belief(1, 101)))
    assert value.released_member_ids
    value.reset()
    assert not value.released_member_ids
    assert value.decide(step=100, beliefs=(belief(0, 100), belief(1, 101))).launch_member_ids == ()


def test_initial_release_covers_at_least_half_without_splitting_packages():
    value = MissionReplanSkill(
        MissionReplanConfig(1, release_delay_steps=0),
        (
            package("p0", 0, 0, 1, 2, 3, 4),
            package("p1", 0, 1, 5, 6, 7),
            package("p2", 0, 2, 8, 9, 10),
            package("p3", 0, 3, 11, 12),
            package("p4", 0, 4, 13, 14),
        ),
    )
    value.update_discoveries(step=0, target_ids=(100,))
    decision = value.decide(step=0, beliefs=(belief(0, 100),))
    assert decision.launch_package_ids == ("p0", "p1")
    assert len(decision.launch_member_ids) == 7
