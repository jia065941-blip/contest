from dataclasses import FrozenInstanceError

import pytest

from dynamic_retarget_skill import (
    DynamicRetargetConfig,
    DynamicRetargetSkill,
    RetargetMemberSnapshot,
    RetargetTargetSnapshot,
    __version__,
)


def member(
    entity_id: int,
    *,
    missile_type: int = 21001,
    current_target_id: int = 901,
    alive: bool = True,
    stage: int = 2,
    current_distance_km: float = 30.0,
    communication_group: object = "a",
    is_core: bool = False,
    retarget_count: int = 0,
    last_retarget_step: int | None = None,
    lon: float = 0.0,
    lat: float = 0.0,
    legal_target_ids: tuple[int, ...] = (901, 902, 903),
    fresh_target_ids: tuple[int, ...] = (901, 902, 903),
    velocity_x: float | None = 1.0,
    velocity_y: float | None = 0.0,
) -> RetargetMemberSnapshot:
    return RetargetMemberSnapshot(
        entity_id=entity_id,
        missile_type=missile_type,
        current_target_id=current_target_id,
        alive=alive,
        stage=stage,
        current_distance_km=current_distance_km,
        communication_group=communication_group,
        is_core=is_core,
        retarget_count=retarget_count,
        last_retarget_step=last_retarget_step,
        lon=lon,
        lat=lat,
        legal_target_ids=legal_target_ids,
        fresh_target_ids=fresh_target_ids,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
    )


def target(
    target_id: int,
    target_type: int,
    *,
    value: float,
    lon: float = 0.1,
    lat: float = 0.0,
    destruction_probability: float = 0.1,
    projected_probability: float = 0.2,
    assigned_count: int = 0,
) -> RetargetTargetSnapshot:
    return RetargetTargetSnapshot(
        target_id,
        target_type,
        value,
        lon,
        lat,
        destruction_probability,
        projected_probability,
        assigned_count,
    )


def public_targets() -> tuple[RetargetTargetSnapshot, ...]:
    return (
        target(
            901,
            9400,
            value=5.0,
            destruction_probability=0.95,
            projected_probability=1.0,
            assigned_count=6,
        ),
        target(902, 9600, value=2.0),
        target(903, 9600, value=2.0, lon=0.2, assigned_count=1),
    )


def test_frozen_dtos_and_default_config() -> None:
    snapshot = member(1)
    with pytest.raises(FrozenInstanceError):
        snapshot.stage = 3

    config = DynamicRetargetConfig()
    assert __version__ == "0.3.0"
    assert config.destruction_threshold == pytest.approx(0.90)
    assert config.max_stage == 2
    assert config.min_current_distance_km == pytest.approx(15.0)
    assert config.cooldown_steps == 25
    assert config.max_retargets == 1
    assert config.max_fraction_per_step == pytest.approx(0.2)
    assert 0.0 < config.distance_weight < 1.0
    assert config.max_target_separation_km == pytest.approx(15.0)
    assert config.max_turn_angle_degrees == pytest.approx(60.0)
    assert config.max_candidate_distance_km == pytest.approx(60.0)
    assert config.max_per_old_target_per_step == 1


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("max_target_separation_km", 0.0),
        ("max_target_separation_km", float("nan")),
        ("max_turn_angle_degrees", -1.0),
        ("max_turn_angle_degrees", float("inf")),
        ("max_candidate_distance_km", 0.0),
        ("max_candidate_distance_km", float("-inf")),
        ("max_per_old_target_per_step", 0),
        ("max_per_old_target_per_step", 1.5),
        ("max_per_old_target_per_step", True),
    ),
)
def test_new_config_limits_require_finite_positive_values(field, value) -> None:
    with pytest.raises(ValueError):
        DynamicRetargetConfig(**{field: value})


def test_turn_angle_cannot_exceed_half_circle() -> None:
    with pytest.raises(ValueError, match="at most 180"):
        DynamicRetargetConfig(max_turn_angle_degrees=180.01)


def test_snapshot_optional_tracking_fields_are_fail_closed() -> None:
    snapshot = RetargetMemberSnapshot(
        1, 21001, 901, True, 2, 30.0, "a", False,
        0, None, 0.0, 0.0, (901, 902),
    )
    assert snapshot.fresh_target_ids == ()
    assert snapshot.velocity_x is None
    assert snapshot.velocity_y is None

    decision = DynamicRetargetSkill().decide(
        step=100,
        members=(snapshot,),
        targets=public_targets()[:2],
    )
    assert decision.requests == ()
    assert decision.rejected_reasons[1] == "candidate_track_stale"


def test_snapshot_rejects_partial_or_nonfinite_velocity() -> None:
    with pytest.raises(ValueError, match="both be set"):
        member(1, velocity_x=1.0, velocity_y=None)
    with pytest.raises(ValueError, match="finite"):
        member(1, velocity_x=float("nan"), velocity_y=0.0)


def test_core_member_never_moves() -> None:
    decision = DynamicRetargetSkill().decide(
        step=100,
        members=(member(1, is_core=True),),
        targets=public_targets(),
    )
    assert decision.requests == ()
    assert decision.eligible_member_ids == ()
    assert decision.rejected_reasons[1] == "core_member"
    with pytest.raises(TypeError):
        decision.rejected_reasons[1] = "changed"


def test_projected_probability_alone_cannot_release_member() -> None:
    targets = (
        target(
            901,
            9400,
            value=5.0,
            destruction_probability=0.89,
            projected_probability=1.0,
        ),
        target(902, 9600, value=2.0),
    )
    decision = DynamicRetargetSkill().decide(
        step=100,
        members=(member(1),),
        targets=targets,
    )
    assert decision.requests == ()


@pytest.mark.parametrize(
    "snapshot",
    (
        member(1, stage=3),
        member(1, current_distance_km=14.99),
        member(1, last_retarget_step=76),
    ),
)
def test_stage_distance_and_cooldown_limits(snapshot) -> None:
    assert DynamicRetargetSkill().decide(
        step=100,
        members=(snapshot,),
        targets=public_targets(),
    ).requests == ()


def test_member_can_retarget_only_once_by_default() -> None:
    decision = DynamicRetargetSkill().decide(
        step=100,
        members=(member(1, retarget_count=1, last_retarget_step=1),),
        targets=public_targets(),
    )
    assert decision.requests == ()


def test_owner_legal_target_ids_filter_candidates() -> None:
    decision = DynamicRetargetSkill().decide(
        step=100,
        members=(member(1, legal_target_ids=(901, 903)),),
        targets=public_targets(),
    )
    assert [(item.old_target_id, item.new_target_id) for item in decision.requests] == [
        (901, 903),
    ]


def test_candidate_must_be_both_legal_and_fresh() -> None:
    decision = DynamicRetargetSkill().decide(
        step=100,
        members=(member(
            1,
            legal_target_ids=(901, 903),
            fresh_target_ids=(901, 902),
        ),),
        targets=public_targets(),
    )
    assert decision.requests == ()
    assert decision.rejected_reasons[1] == "candidate_track_stale"


def test_controller_rejects_cross_group_member_input() -> None:
    with pytest.raises(ValueError, match="one communication_group"):
        DynamicRetargetSkill().decide(
            step=100,
            members=(member(1, communication_group="a"), member(2, communication_group="b")),
            targets=public_targets(),
        )


def test_hashable_member_tuple_can_identify_communication_group() -> None:
    decision = DynamicRetargetSkill().decide(
        step=100,
        members=(member(1, communication_group=(1, 2)), member(2, communication_group=(1, 2))),
        targets=public_targets(),
    )
    assert decision.eligible_member_ids == (1, 2)


def test_per_step_budget_is_ceil_twenty_percent_and_at_least_one() -> None:
    members = tuple(member(entity_id) for entity_id in range(10, 20))
    skill = DynamicRetargetSkill(DynamicRetargetConfig(
        max_per_old_target_per_step=10,
    ))
    decision = skill.decide(
        step=100,
        members=members,
        targets=public_targets(),
    )
    assert decision.budget == 2
    assert [item.entity_id for item in decision.requests] == [10, 11]

    one = skill.decide(
        step=100,
        members=(member(99),),
        targets=public_targets(),
    )
    assert one.budget == 1
    assert len(one.requests) == 1


def test_ordering_is_deterministic_and_balances_effective_assignment_count() -> None:
    config = DynamicRetargetConfig(
        max_fraction_per_step=1.0,
        max_per_old_target_per_step=3,
    )
    members = (member(12), member(10), member(11))
    targets = (
        public_targets()[0],
        target(902, 9600, value=2.0, lon=0.1),
        target(903, 9600, value=2.0, lon=0.1),
    )
    first = DynamicRetargetSkill(config).decide(
        step=100,
        members=members,
        targets=targets,
    )
    second = DynamicRetargetSkill(config).decide(
        step=100,
        members=tuple(reversed(members)),
        targets=tuple(reversed(targets)),
    )
    assert first == second
    assert [(item.entity_id, item.new_target_id) for item in first.requests] == [
        (10, 902),
        (11, 903),
        (12, 902),
    ]


def test_default_allows_only_one_request_per_old_target_per_step() -> None:
    skill = DynamicRetargetSkill(DynamicRetargetConfig(max_fraction_per_step=1.0))
    first = skill.decide(
        step=100,
        members=(member(2), member(1)),
        targets=public_targets(),
    )
    second = skill.decide(
        step=100,
        members=(member(1), member(2)),
        targets=tuple(reversed(public_targets())),
    )
    assert first == second
    assert [request.entity_id for request in first.requests] == [1]
    assert first.eligible_member_ids == (1, 2)
    assert first.rejected_reasons[2] == "old_target_step_limit"


def test_flexible_m_moves_from_satisfied_mother_to_sentry() -> None:
    decision = DynamicRetargetSkill().decide(
        step=100,
        members=(member(20, missile_type=21001, current_target_id=901),),
        targets=public_targets(),
    )
    assert len(decision.requests) == 1
    request = decision.requests[0]
    assert request.entity_id == 20
    assert request.old_target_id == 901
    assert request.new_target_id == 902
    assert request.target_destruction_probability == pytest.approx(0.95)


def test_near_fresh_target_in_current_heading_is_selected() -> None:
    decision = DynamicRetargetSkill().decide(
        step=100,
        members=(member(
            20,
            legal_target_ids=(901, 902),
            fresh_target_ids=(902,),
            velocity_x=1.0,
            velocity_y=0.0,
        ),),
        targets=public_targets()[:2],
    )
    assert decision.requests[0].new_target_id == 902


def test_candidate_over_target_separation_limit_is_rejected() -> None:
    targets = (
        target(901, 9400, value=5.0, lon=0.0, destruction_probability=0.95),
        target(902, 9600, value=2.0, lon=0.2),
    )
    decision = DynamicRetargetSkill().decide(
        step=100,
        members=(member(1, legal_target_ids=(901, 902), fresh_target_ids=(902,)),),
        targets=targets,
    )
    assert decision.requests == ()
    assert decision.rejected_reasons[1] == "target_separation"


def test_candidate_outside_turn_angle_is_rejected() -> None:
    decision = DynamicRetargetSkill().decide(
        step=100,
        members=(member(1, velocity_x=-1.0, velocity_y=0.0),),
        targets=public_targets(),
    )
    assert decision.requests == ()
    assert decision.rejected_reasons[1] == "turn_angle"


def test_unknown_heading_is_rejected_as_turn_angle() -> None:
    decision = DynamicRetargetSkill().decide(
        step=100,
        members=(member(1, velocity_x=None, velocity_y=None),),
        targets=public_targets(),
    )
    assert decision.requests == ()
    assert decision.rejected_reasons[1] == "turn_angle"


def test_candidate_over_member_distance_limit_is_rejected() -> None:
    targets = (
        target(901, 9400, value=5.0, lon=0.50, destruction_probability=0.95),
        target(902, 9600, value=2.0, lon=0.55),
    )
    decision = DynamicRetargetSkill().decide(
        step=100,
        members=(member(1, legal_target_ids=(901, 902), fresh_target_ids=(902,)),),
        targets=targets,
    )
    assert decision.requests == ()
    assert decision.rejected_reasons[1] == "candidate_too_far"


def test_public_eligibility_reason_uses_decision_filtering() -> None:
    skill = DynamicRetargetSkill()
    snapshot = member(1, velocity_x=-1.0, velocity_y=0.0)
    targets = public_targets()
    assert skill.eligibility_reason(
        snapshot,
        targets[0],
        targets,
        step=100,
    ) == "turn_angle"


def test_projected_satisfied_candidate_is_not_selected() -> None:
    targets = (
        public_targets()[0],
        target(902, 9400, value=5.0, projected_probability=0.95),
        target(903, 9600, value=2.0, projected_probability=0.4),
    )
    decision = DynamicRetargetSkill().decide(
        step=100,
        members=(member(20),),
        targets=targets,
    )

    assert decision.requests[0].new_target_id == 903


def test_only_h_m_and_public_target_types_are_accepted() -> None:
    with pytest.raises(ValueError, match="H=21000 or M=21001"):
        member(1, missile_type=21002)
    with pytest.raises(ValueError, match="public 9400 or 9600"):
        target(950, 9500, value=10.0)
