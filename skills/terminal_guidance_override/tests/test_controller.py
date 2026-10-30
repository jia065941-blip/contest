from dataclasses import FrozenInstanceError, replace
import math

import pytest

from terminal_guidance_override_skill import (
    TerminalGuidanceConfig,
    TerminalGuidanceSkill,
    TerminalGuidanceSnapshot,
    TerminalTypeProfile,
    __version__,
)


METRES_PER_LATITUDE_DEGREE = 111_195.08


def snapshot(
    *,
    entity_id: int = 1,
    missile_type: int = 21001,
    stage: int = 4,
    baseline_action: int = 1,
    distance_m: float = 500.0,
    heading_error_deg: float = 20.0,
    track_source: str = "detect_fresh",
    track_age: float = 0.0,
    nearest_interceptor_distance_m: float | None = None,
    step: int = 10,
    velocity_source: str = "observed",
) -> TerminalGuidanceSnapshot:
    speed = 300.0
    heading = math.radians(-heading_error_deg)
    return TerminalGuidanceSnapshot(
        entity_id=entity_id,
        missile_type=missile_type,
        stage=stage,
        baseline_action=baseline_action,
        aimpoint_lon=0.0,
        aimpoint_lat=distance_m / METRES_PER_LATITUDE_DEGREE,
        track_source=track_source,
        track_age=track_age,
        nearest_interceptor_distance_m=nearest_interceptor_distance_m,
        step=step,
        position_lon=0.0,
        position_lat=0.0,
        velocity_east_mps=speed * math.sin(heading),
        velocity_north_mps=speed * math.cos(heading),
        velocity_source=velocity_source,
    )


def test_version_frozen_dtos_and_v04_default_profiles() -> None:
    item = snapshot()
    decision = TerminalGuidanceSkill().decide(item)
    profile = TerminalTypeProfile(capture_distance=100.0)

    with pytest.raises(FrozenInstanceError):
        item.stage = 5
    with pytest.raises(FrozenInstanceError):
        decision.final = 1
    with pytest.raises(FrozenInstanceError):
        profile.capture_distance = 200.0

    config = TerminalGuidanceConfig()
    profiles = dict(config.profiles)
    assert __version__ == "0.4.0"
    assert config.enabled_types == (21000, 21001, 21002)
    assert profiles[21000].capture_distance == pytest.approx(300.0)
    assert profiles[21000].final_straight_distance == pytest.approx(150.0)
    assert profiles[21000].max_track_age == 5
    assert profiles[21000].require_fresh_track is True
    assert profiles[21001].capture_distance == pytest.approx(600.0)
    assert profiles[21001].final_straight_distance == pytest.approx(200.0)
    assert profiles[21001].max_track_age == 5
    assert profiles[21001].require_fresh_track is True
    assert profiles[21002].capture_distance == pytest.approx(600.0)
    assert profiles[21002].final_straight_distance == pytest.approx(200.0)
    assert profiles[21002].max_track_age is None
    assert profiles[21002].require_fresh_track is False
    assert config.capture_distance == pytest.approx(600.0)
    assert config.final_straight_distance == pytest.approx(200.0)
    assert config.max_track_age == 5
    assert config.stale_grace_age == 8
    assert config.heading_enter_straight_deg == pytest.approx(6.0)
    assert config.heading_exit_straight_deg == pytest.approx(10.0)
    assert config.predicted_miss_threshold_m == pytest.approx(80.0)
    assert config.turn_pulse_steps == 1
    assert config.straight_recovery_steps == 2
    assert config.no_improvement_limit == 3
    assert config.latch is True


@pytest.mark.parametrize(
    ("missile_type", "capture_distance", "final_distance"),
    (
        (21000, 300.0, 150.0),
        (21001, 600.0, 200.0),
        (21002, 600.0, 200.0),
    ),
)
def test_default_profile_capture_and_final_boundaries(
    missile_type: int,
    capture_distance: float,
    final_distance: float,
) -> None:
    captured = TerminalGuidanceSkill().decide(snapshot(
        missile_type=missile_type,
        distance_m=capture_distance,
        heading_error_deg=90.0,
    ))
    outside = TerminalGuidanceSkill().decide(snapshot(
        missile_type=missile_type,
        distance_m=capture_distance + 0.01,
        heading_error_deg=90.0,
    ))
    final = TerminalGuidanceSkill().decide(snapshot(
        missile_type=missile_type,
        distance_m=final_distance,
        heading_error_deg=90.0,
        baseline_action=2,
    ))
    above_final = TerminalGuidanceSkill().decide(snapshot(
        missile_type=missile_type,
        distance_m=final_distance + 0.01,
        heading_error_deg=90.0,
    ))

    assert captured.control_state == "TURN_PULSE"
    assert captured.latched is True
    assert outside.reason == "outside_capture_distance"
    assert outside.control_state == "PASSTHROUGH"
    assert final.final == 1
    assert final.control_state == "FINAL_STRAIGHT"
    assert above_final.control_state == "TURN_PULSE"


def test_l_allows_stale_detect_track_but_requires_detect_track_and_aimpoint() -> None:
    allowed = TerminalGuidanceSkill().decide(snapshot(
        missile_type=21002,
        track_source="detect_stale",
        track_age=100_000,
        heading_error_deg=90.0,
    ))
    invalid_source = TerminalGuidanceSkill().decide(snapshot(
        missile_type=21002,
        track_source="estimated",
        track_age=0,
    ))
    missing_aimpoint = TerminalGuidanceSkill().decide(replace(
        snapshot(missile_type=21002, track_source="detect_stale", track_age=100_000),
        aimpoint_lon=None,
        aimpoint_lat=None,
    ))

    assert allowed.control_state == "TURN_PULSE"
    assert allowed.latched is True
    assert invalid_source.reason == "track_stale"
    assert invalid_source.control_state == "PASSTHROUGH"
    assert missing_aimpoint.reason == "aimpoint_missing"


@pytest.mark.parametrize(
    ("missile_type", "distance_m"),
    ((21000, 250.0), (21001, 500.0)),
)
def test_h_and_m_reject_stale_track_on_entry(
    missile_type: int, distance_m: float
) -> None:
    decision = TerminalGuidanceSkill().decide(snapshot(
        missile_type=missile_type,
        distance_m=distance_m,
        track_source="detect_stale",
        track_age=6,
        baseline_action=0,
    ))

    assert decision.final == 0
    assert decision.control_state == "PASSTHROUGH"
    assert decision.reason == "track_stale"


def test_same_owner_has_independent_state_for_each_profile() -> None:
    skill = TerminalGuidanceSkill()

    m_first = skill.decide(snapshot(entity_id=7, missile_type=21001, step=10))
    l_first = skill.decide(snapshot(entity_id=7, missile_type=21002, step=10))
    m_second = skill.decide(snapshot(entity_id=7, missile_type=21001, step=11))
    l_second = skill.decide(snapshot(entity_id=7, missile_type=21002, step=11))

    assert m_first.reason == "entered_turn_pulse_right"
    assert l_first.reason == "entered_turn_pulse_right"
    assert m_second.control_state == "STRAIGHT_RECOVERY"
    assert l_second.control_state == "STRAIGHT_RECOVERY"


def test_profiles_accept_mapping_or_tuple_and_inherit_global_defaults() -> None:
    inherited = TerminalTypeProfile(
        max_track_age=None,
        require_fresh_track=False,
    )
    mapping_config = TerminalGuidanceConfig(
        capture_distance=700.0,
        final_straight_distance=250.0,
        profiles={21001: inherited},
    )
    tuple_config = TerminalGuidanceConfig(
        profiles=((21002, TerminalTypeProfile(
            capture_distance=450.0,
            final_straight_distance=100.0,
            max_track_age=None,
            require_fresh_track=False,
        )),),
    )

    mapping_decision = TerminalGuidanceSkill(mapping_config).decide(snapshot(
        missile_type=21001,
        distance_m=700.0,
        track_source="detect_stale",
        track_age=100,
        heading_error_deg=90.0,
    ))
    tuple_outside = TerminalGuidanceSkill(tuple_config).decide(snapshot(
        missile_type=21002,
        distance_m=450.01,
    ))

    assert mapping_config.enabled_types == (21001,)
    assert tuple_config.enabled_types == (21002,)
    assert mapping_decision.control_state == "TURN_PULSE"
    assert tuple_outside.reason == "outside_capture_distance"


def test_capture_turn_pulse_then_two_recovery_steps_and_reestimate() -> None:
    skill = TerminalGuidanceSkill()

    decisions = [
        skill.decide(snapshot(step=10, heading_error_deg=20.0)),
        skill.decide(snapshot(step=11, heading_error_deg=18.0)),
        skill.decide(snapshot(step=12, heading_error_deg=16.0)),
        skill.decide(snapshot(step=13, heading_error_deg=14.0)),
    ]

    assert [item.final for item in decisions] == [2, 1, 1, 2]
    assert [item.control_state for item in decisions] == [
        "TURN_PULSE",
        "STRAIGHT_RECOVERY",
        "STRAIGHT_RECOVERY",
        "TURN_PULSE",
    ]
    assert decisions[0].reason == "entered_turn_pulse_right"
    assert decisions[0].turn_direction == "right"
    assert decisions[1].turn_direction is None


def test_heading_sign_cross_clears_turn_and_requires_straight_before_reverse() -> None:
    skill = TerminalGuidanceSkill()

    right = skill.decide(snapshot(step=10, heading_error_deg=20.0))
    crossed = skill.decide(snapshot(step=11, heading_error_deg=-20.0))
    recovery = skill.decide(snapshot(step=12, heading_error_deg=-18.0))
    left = skill.decide(snapshot(step=13, heading_error_deg=-16.0))

    assert [right.final, crossed.final, recovery.final, left.final] == [2, 1, 1, 0]
    assert crossed.control_state == "STRAIGHT_RECOVERY"
    assert crossed.reason == "heading_sign_cross"
    assert crossed.turn_direction is None
    assert left.control_state == "TURN_PULSE"
    assert left.turn_direction == "left"


def test_three_non_improving_observations_restart_straight_recovery() -> None:
    skill = TerminalGuidanceSkill()

    skill.decide(snapshot(step=10, heading_error_deg=20.0))
    first = skill.decide(snapshot(step=11, heading_error_deg=21.0))
    second = skill.decide(snapshot(step=12, heading_error_deg=22.0))
    third = skill.decide(snapshot(step=13, heading_error_deg=23.0))
    after_trigger = skill.decide(snapshot(step=14, heading_error_deg=22.0))

    assert first.no_improvement_count == 1
    assert second.no_improvement_count == 2
    assert third.final == 1
    assert third.control_state == "STRAIGHT_RECOVERY"
    assert third.reason == "no_improvement"
    assert third.no_improvement_count == 3
    assert after_trigger.control_state == "STRAIGHT_RECOVERY"
    assert after_trigger.no_improvement_count == 0


def test_heading_straight_hysteresis_enters_at_six_and_exits_at_ten() -> None:
    skill = TerminalGuidanceSkill(TerminalGuidanceConfig(
        predicted_miss_threshold_m=0.0
    ))

    entered = skill.decide(snapshot(step=10, heading_error_deg=6.0))
    held = skill.decide(snapshot(step=11, heading_error_deg=9.0))
    exited = skill.decide(snapshot(step=12, heading_error_deg=10.0))

    assert entered.control_state == "STRAIGHT"
    assert entered.reason == "heading_hysteresis_straight"
    assert held.control_state == "STRAIGHT"
    assert held.reason == "heading_hysteresis_straight"
    assert exited.control_state == "TURN_PULSE"
    assert exited.final == 2


def test_constant_velocity_predicted_miss_can_command_straight() -> None:
    decision = TerminalGuidanceSkill().decide(snapshot(
        distance_m=500.0,
        heading_error_deg=8.0,
    ))

    assert decision.distance == pytest.approx(500.0, rel=2e-4)
    assert decision.predicted_miss_distance == pytest.approx(
        500.0 * math.sin(math.radians(8.0)), rel=2e-4
    )
    assert decision.control_state == "STRAIGHT"
    assert decision.reason == "predicted_miss_straight"
    assert decision.final == 1


def test_perpendicular_constant_velocity_miss_is_current_range() -> None:
    decision = TerminalGuidanceSkill().decide(snapshot(heading_error_deg=90.0))

    assert decision.predicted_miss_distance == pytest.approx(
        decision.distance, rel=2e-4
    )
    assert decision.control_state == "TURN_PULSE"


def test_latched_track_uses_stale_grace_through_age_eight_then_releases() -> None:
    skill = TerminalGuidanceSkill()
    skill.decide(snapshot(step=10))

    age_six = skill.decide(snapshot(
        step=11,
        track_source="detect_stale",
        track_age=6,
        baseline_action=0,
    ))
    age_eight = skill.decide(snapshot(
        step=12,
        track_source="detect_stale",
        track_age=8,
        baseline_action=2,
    ))
    expired = skill.decide(snapshot(
        step=13,
        track_source="detect_stale",
        track_age=8.01,
        baseline_action=0,
    ))
    no_longer_latched = skill.decide(snapshot(
        step=14,
        distance_m=2500.0,
        baseline_action=2,
    ))

    assert age_six.final == 1
    assert age_six.control_state == "STALE_GRACE"
    assert age_six.reason == "stale_grace"
    assert age_six.latched is True
    assert age_eight.control_state == "STALE_GRACE"
    assert expired.final == 0
    assert expired.control_state == "PASSTHROUGH"
    assert expired.latched is False
    assert expired.reason == "track_stale"
    assert no_longer_latched.reason == "outside_capture_distance"


def test_unlatched_stale_track_cannot_enter_capture() -> None:
    decision = TerminalGuidanceSkill().decide(snapshot(
        track_source="detect_stale",
        track_age=6,
        baseline_action=0,
    ))

    assert decision.final == 0
    assert decision.control_state == "PASSTHROUGH"
    assert decision.reason == "track_stale"


def test_final_distance_forces_final_straight_even_when_misaligned() -> None:
    decision = TerminalGuidanceSkill().decide(snapshot(
        distance_m=200.0,
        heading_error_deg=90.0,
        baseline_action=2,
    ))

    assert decision.final == 1
    assert decision.mode == "straight"
    assert decision.control_state == "FINAL_STRAIGHT"
    assert decision.reason == "final_straight"
    assert decision.turn_direction is None


def test_threats_are_ignored_after_latch_even_with_deprecated_release_flag() -> None:
    skill = TerminalGuidanceSkill(TerminalGuidanceConfig(
        ignore_interceptors_after_latch=False
    ))

    skill.decide(snapshot())
    decision = skill.decide(snapshot(
        step=11,
        distance_m=700.0,
        nearest_interceptor_distance_m=1.0,
    ))

    assert decision.threat is True
    assert decision.latched is True
    assert decision.control_state == "STRAIGHT_RECOVERY"


def test_velocity_source_is_validated_and_copied_to_decision() -> None:
    decision = TerminalGuidanceSkill().decide(snapshot(velocity_source="estimated"))
    assert decision.velocity_source == "estimated"

    with pytest.raises(ValueError, match="velocity_source"):
        replace(snapshot(), velocity_source="")


def test_legacy_config_fields_map_to_v03_fields() -> None:
    config = TerminalGuidanceConfig(
        enter_distance_m=2500.0,
        straight_distance_m=400.0,
        exit_distance_m=3000.0,
        max_track_age_steps=4,
        heading_deadband_degrees=7.0,
        threat_distance_m=500.0,
    )

    assert config.capture_distance == pytest.approx(2500.0)
    assert config.capture_distance_m == pytest.approx(2500.0)
    assert config.enter_distance_m == pytest.approx(2500.0)
    assert config.final_straight_distance == pytest.approx(400.0)
    assert config.straight_distance_m == pytest.approx(400.0)
    assert config.max_track_age == 4
    assert config.max_track_age_steps == 4
    assert config.heading_enter_straight_deg == pytest.approx(7.0)
    assert config.heading_exit_straight_deg == pytest.approx(7.0)

    old_default_with_enter_alias = TerminalGuidanceConfig(
        capture_distance_m=3000.0,
        enter_distance_m=1200.0,
    )
    assert old_default_with_enter_alias.capture_distance == pytest.approx(1200.0)


def test_legacy_enabled_types_builds_one_profile_from_global_fields() -> None:
    config = TerminalGuidanceConfig(
        enabled_types=(21001,),
        capture_distance=900.0,
        final_straight_distance=250.0,
        max_track_age=4,
    )
    skill = TerminalGuidanceSkill(config)

    captured_m = skill.decide(snapshot(
        missile_type=21001,
        distance_m=900.0,
        heading_error_deg=90.0,
    ))
    unconfigured_h = skill.decide(snapshot(
        missile_type=21000,
        distance_m=250.0,
        heading_error_deg=90.0,
    ))

    assert config.enabled_types == (21001,)
    assert tuple(dict(config.profiles)) == (21001,)
    assert captured_m.control_state == "TURN_PULSE"
    assert unconfigured_h.reason == "missile_type_passthrough"


def test_reset_clears_one_owner_or_all_owner_state() -> None:
    skill = TerminalGuidanceSkill()
    skill.decide(snapshot(entity_id=1))
    skill.decide(snapshot(entity_id=2))

    skill.reset(1)
    owner_one = skill.decide(snapshot(entity_id=1, distance_m=700.0))
    owner_two = skill.decide(snapshot(entity_id=2, distance_m=700.0, step=11))
    assert owner_one.control_state == "PASSTHROUGH"
    assert owner_one.latched is False
    assert owner_two.control_state == "STRAIGHT_RECOVERY"
    assert owner_two.latched is True

    skill.reset()
    after_all = skill.decide(snapshot(entity_id=2, distance_m=700.0, step=12))
    assert after_all.control_state == "PASSTHROUGH"
    assert after_all.latched is False


def test_reset_owner_clears_that_owner_across_all_type_profiles() -> None:
    skill = TerminalGuidanceSkill()
    skill.decide(snapshot(entity_id=1, missile_type=21000, distance_m=250.0))
    skill.decide(snapshot(entity_id=1, missile_type=21001))
    skill.decide(snapshot(entity_id=2, missile_type=21002))

    skill.reset(1)
    h_owner_one = skill.decide(snapshot(
        entity_id=1,
        missile_type=21000,
        distance_m=350.0,
    ))
    m_owner_one = skill.decide(snapshot(
        entity_id=1,
        missile_type=21001,
        distance_m=700.0,
    ))
    l_owner_two = skill.decide(snapshot(
        entity_id=2,
        missile_type=21002,
        distance_m=700.0,
        step=11,
    ))

    assert h_owner_one.control_state == "PASSTHROUGH"
    assert m_owner_one.control_state == "PASSTHROUGH"
    assert l_owner_two.control_state == "STRAIGHT_RECOVERY"
    assert l_owner_two.latched is True


def test_unprofiled_type_stage_and_capture_gates_pass_through() -> None:
    wrong_type = TerminalGuidanceSkill().decide(snapshot(
        missile_type=21003,
        baseline_action=0,
    ))
    low_stage = TerminalGuidanceSkill().decide(snapshot(
        stage=3,
        baseline_action=0,
    ))
    too_far = TerminalGuidanceSkill().decide(snapshot(
        distance_m=601.0,
        baseline_action=0,
    ))

    assert wrong_type.reason == "missile_type_passthrough"
    assert low_stage.reason == "stage_below_minimum"
    assert too_far.reason == "outside_capture_distance"
    assert {wrong_type.control_state, low_stage.control_state, too_far.control_state} == {
        "PASSTHROUGH"
    }


def test_ecef_position_velocity_and_aimpoint_are_supported() -> None:
    radius = 6_378_137.0
    angle = math.radians(0.01)
    item = TerminalGuidanceSnapshot(
        entity_id=1,
        missile_type=21001,
        stage=4,
        baseline_action=1,
        aimpoint_ecf=(radius * math.cos(angle), radius * math.sin(angle), 0.0),
        position_ecf=(radius, 0.0, 0.0),
        velocity_ecf=(0.0, 0.0, 300.0),
        velocity_source="observed_ecf",
    )

    decision = TerminalGuidanceSkill(TerminalGuidanceConfig(
        enabled_types=(21001,),
        capture_distance=2000.0,
    )).decide(item)

    assert decision.distance == pytest.approx(1113.19, rel=1e-4)
    assert decision.heading_error == pytest.approx(90.0, abs=0.01)
    assert decision.predicted_miss_distance == pytest.approx(1113.19, rel=1e-4)
    assert decision.final == 2
    assert decision.velocity_source == "observed_ecf"


def test_invalid_inputs_are_rejected_before_or_without_state_changes() -> None:
    skill = TerminalGuidanceSkill()
    with pytest.raises(ValueError, match="baseline_action"):
        skill.decide(snapshot(baseline_action=3))
    assert skill.decide(snapshot(distance_m=2500.0)).latched is False

    with pytest.raises(ValueError, match="must not exceed"):
        TerminalGuidanceConfig(final_straight_distance=2001.0)
    with pytest.raises(ValueError, match="at least max_track_age"):
        TerminalGuidanceConfig(max_track_age=9, stale_grace_age=8)
    with pytest.raises(ValueError, match="must not exceed"):
        TerminalGuidanceConfig(
            heading_enter_straight_deg=11,
            heading_exit_straight_deg=10,
        )
    with pytest.raises(ValueError, match="positive integer"):
        TerminalGuidanceConfig(turn_pulse_steps=0)
    with pytest.raises(ValueError, match="conflicts"):
        TerminalGuidanceConfig(
            capture_distance=1800.0,
            capture_distance_m=1700.0,
        )
