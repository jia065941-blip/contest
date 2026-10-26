from dataclasses import FrozenInstanceError

import pytest

from reserve_echelon_skill import (
    H_MISSILE_TYPE,
    L_MISSILE_TYPE,
    ReserveEchelonConfig,
    ReserveEchelonSkill,
    ReserveMemberSnapshot,
    ReservePackageSnapshot,
    TargetDeficitSnapshot,
    __version__,
)


def member(
    entity_id: int,
    *,
    member_type: int = H_MISSILE_TYPE,
    stage: int = 1,
    group: object = "alpha",
    current_target: int = 900,
    legal_target_ids: tuple[int, ...] = (900, 901, 902, 950, 999),
    position: tuple[float, float] = (0.0, 0.0),
    retarget_count: int = 0,
    heading_east: float | None = 1.0,
    heading_north: float | None = 0.0,
    remaining_steps: int | None = None,
) -> ReserveMemberSnapshot:
    return ReserveMemberSnapshot(
        entity_id,
        member_type,
        stage,
        group,
        current_target,
        legal_target_ids,
        position,
        retarget_count,
        heading_east,
        heading_north,
        remaining_steps,
    )


def package(
    package_id: int = 1,
    *,
    member_ids: tuple[int, ...] = (10,),
    member_types: tuple[int, ...] = (H_MISSILE_TYPE,),
    region: object = "west",
    state: str = "STAGING",
    dispatch_count: int = 0,
    fallback_position: tuple[float, float] | None = (1.0, 0.0),
) -> ReservePackageSnapshot:
    return ReservePackageSnapshot(
        package_id,
        member_ids,
        member_types,
        state,
        900,
        999,
        region,
        dispatch_count,
        fallback_position,
    )


def deficit(
    target_id: int,
    *,
    target_type: int = 9600,
    weight: float = 2.0,
    missing_damage: float = 10.0,
    priority: float = 0.0,
    position: tuple[float, float] = (1.0, 0.0),
    region: object = "west",
    evidence_ids: tuple[int, ...] = (10,),
) -> TargetDeficitSnapshot:
    return TargetDeficitSnapshot(
        target_id,
        target_type,
        weight,
        missing_damage,
        priority,
        position,
        region,
        evidence_ids,
    )


def decide(
    skill: ReserveEchelonSkill,
    *,
    step: int = 100,
    members: tuple[ReserveMemberSnapshot, ...] = (member(10),),
    deficits: tuple[TargetDeficitSnapshot, ...] = (deficit(901),),
    packages: tuple[ReservePackageSnapshot, ...] = (package(),),
):
    return skill.decide(
        step=step,
        members=members,
        deficits=deficits,
        packages=packages,
    )


def test_frozen_dtos_version_and_default_config() -> None:
    snapshot = member(10)
    with pytest.raises(FrozenInstanceError):
        snapshot.stage = 2

    config = ReserveEchelonConfig()
    assert __version__ == "0.2.0"
    assert config.max_dispatch_stage is None
    assert config.max_dispatches == 1
    assert config.deadline_step == 300
    assert config.require_same_group_evidence is True
    assert config.max_target_distance_km is None
    assert config.h_max_distance_km == pytest.approx(80.0)
    assert config.l_max_distance_km == pytest.approx(100.0)
    assert config.h_max_turn_angle_degrees == pytest.approx(60.0)
    assert config.l_max_turn_angle_degrees == pytest.approx(75.0)
    assert config.seconds_per_step == pytest.approx(1.0)
    assert config.h_nominal_speed_mps == pytest.approx(200.0)
    assert config.l_nominal_speed_mps == pytest.approx(100.0)


def test_mother_deficit_wins_weighted_missing_damage_priority() -> None:
    mother = deficit(
        902,
        target_type=9400,
        weight=5.0,
        missing_damage=20.0,
        position=(4.0, 0.0),
    )
    sentry = deficit(
        901,
        target_type=9600,
        weight=2.0,
        missing_damage=10.0,
        position=(1.0, 0.0),
    )
    decision = decide(
        ReserveEchelonSkill(),
        deficits=(sentry, mother),
    )
    assert decision.requests[0].new_target_id == 902
    assert decision.requests[0].deficit == mother
    assert decision.packages[0].state == "DISPATCHED"


def test_default_config_dispatches_stage_three_and_four_package() -> None:
    members = (member(10, stage=3), member(11, stage=4))
    decision = decide(
        ReserveEchelonSkill(),
        members=members,
        packages=(package(
            member_ids=(10, 11),
            member_types=(H_MISSILE_TYPE, H_MISSILE_TYPE),
        ),),
    )
    assert decision.requests[0].new_target_id == 901
    assert decision.packages[0].state == "DISPATCHED"


def test_explicit_stage_limit_retains_legacy_gate() -> None:
    decision = decide(
        ReserveEchelonSkill(ReserveEchelonConfig(max_dispatch_stage=2)),
        members=(member(10, stage=3),),
    )
    assert decision.requests == ()
    assert decision.rejected_reasons[1] == "stage_too_late"
    assert decision.packages[0].state == "STAGING"


def test_deadline_without_deficit_commits_atomic_fallback() -> None:
    decision = decide(
        ReserveEchelonSkill(),
        step=300,
        members=(member(10, stage=4),),
        deficits=(),
    )
    assert len(decision.requests) == 1
    request = decision.requests[0]
    assert request.member_ids == (10,)
    assert (request.old_target_id, request.new_target_id) == (900, 999)
    assert request.reason == "deadline_no_deficit_fallback"
    assert request.deficit is None
    assert decision.packages[0].state == "FALLBACK"


def test_wrong_region_and_wrong_group_evidence_are_ineligible() -> None:
    other = member(20, group="bravo")
    wrong_region = deficit(901, region="east")
    wrong_group = deficit(902, evidence_ids=(20,))
    decision = decide(
        ReserveEchelonSkill(),
        members=(member(10), other),
        deficits=(wrong_group, wrong_region),
    )
    assert decision.requests == ()
    assert decision.rejected_reasons[1] == "no_eligible_deficit"


def test_same_group_evidence_can_be_disabled() -> None:
    skill = ReserveEchelonSkill(ReserveEchelonConfig(
        require_same_group_evidence=False,
    ))
    decision = decide(skill, deficits=(deficit(901, evidence_ids=()),))
    assert decision.requests[0].new_target_id == 901


def test_l_single_member_package_dispatches_only_to_dark_target() -> None:
    l_member = member(30, member_type=L_MISSILE_TYPE)
    l_package = package(
        member_ids=(30,),
        member_types=(L_MISSILE_TYPE,),
    )
    public = deficit(901, target_type=9400, evidence_ids=(30,))
    dark = deficit(
        950,
        target_type=9500,
        weight=1.0,
        missing_damage=1.0,
        evidence_ids=(30,),
    )
    decision = decide(
        ReserveEchelonSkill(),
        members=(l_member,),
        deficits=(public, dark),
        packages=(l_package,),
    )
    assert decision.requests[0].new_target_id == 950


def test_dispatch_is_persistent_and_occurs_only_once() -> None:
    skill = ReserveEchelonSkill()
    first = decide(skill)
    dispatched_member = member(10, current_target=901)
    second = decide(
        skill,
        members=(dispatched_member,),
        deficits=(deficit(902),),
    )

    assert len(first.requests) == 1
    assert second.requests == ()
    assert second.rejected_reasons[1] == "already_dispatched"
    assert skill.state[0].dispatch_count == 1

    skill.reset()
    assert skill.state == ()
    after_reset = decide(skill, deficits=(deficit(902),))
    assert after_reset.requests[0].new_target_id == 902


def test_no_deficit_before_deadline_keeps_package_staging() -> None:
    decision = decide(ReserveEchelonSkill(), step=299, deficits=())
    assert decision.requests == ()
    assert decision.rejected_reasons[1] == "no_deficit"
    assert decision.packages[0].state == "STAGING"


def test_all_members_must_allow_target_for_atomic_dispatch() -> None:
    members = (
        member(10),
        member(11, legal_target_ids=(900, 902, 999)),
    )
    decision = decide(
        ReserveEchelonSkill(),
        members=members,
        deficits=(deficit(901),),
        packages=(package(
            member_ids=(10, 11),
            member_types=(H_MISSILE_TYPE, H_MISSILE_TYPE),
        ),),
    )
    assert decision.requests == ()
    assert decision.packages[0].state == "STAGING"


def test_legacy_optional_distance_limit_is_an_additional_cap() -> None:
    skill = ReserveEchelonSkill(ReserveEchelonConfig(
        max_target_distance_km=5.0,
    ))
    too_far = deficit(901, position=(5.01, 0.0))
    decision = decide(skill, deficits=(too_far,))
    assert decision.requests == ()
    assert decision.rejected_reasons[1] == "distance_exceeded"


@pytest.mark.parametrize(
    ("member_type", "target_type", "target_id", "distance_limit"),
    (
        (H_MISSILE_TYPE, 9600, 901, 80.0),
        (L_MISSILE_TYPE, 9500, 950, 100.0),
    ),
)
def test_type_distance_limit_rejects_far_candidate(
    member_type, target_type, target_id, distance_limit
) -> None:
    decision = decide(
        ReserveEchelonSkill(),
        members=(member(10, member_type=member_type),),
        deficits=(deficit(
            target_id,
            target_type=target_type,
            position=(distance_limit + 0.01, 0.0),
        ),),
        packages=(package(member_types=(member_type,)),),
    )
    assert decision.requests == ()
    assert decision.rejected_reasons[1] == "distance_exceeded"


@pytest.mark.parametrize(
    ("member_type", "target_type", "target_id"),
    (
        (H_MISSILE_TYPE, 9600, 901),
        (L_MISSILE_TYPE, 9500, 950),
    ),
)
def test_turn_limit_rejects_large_heading_change(
    member_type, target_type, target_id
) -> None:
    decision = decide(
        ReserveEchelonSkill(),
        members=(member(10, member_type=member_type),),
        deficits=(deficit(
            target_id,
            target_type=target_type,
            position=(0.0, 10.0),
        ),),
        packages=(package(member_types=(member_type,)),),
    )
    assert decision.requests == ()
    assert decision.rejected_reasons[1] == "turn_angle_exceeded"


def test_missing_heading_fails_closed() -> None:
    decision = decide(
        ReserveEchelonSkill(),
        members=(member(10, heading_east=None, heading_north=None),),
    )
    assert decision.requests == ()
    assert decision.rejected_reasons[1] == "missing_heading"


def test_estimated_arrival_must_fit_remaining_steps() -> None:
    too_late = decide(
        ReserveEchelonSkill(),
        members=(member(10, remaining_steps=49),),
        deficits=(deficit(901, position=(10.0, 0.0)),),
    )
    on_time = decide(
        ReserveEchelonSkill(),
        members=(member(10, remaining_steps=50),),
        deficits=(deficit(901, position=(10.0, 0.0)),),
    )
    assert too_late.requests == ()
    assert too_late.rejected_reasons[1] == "insufficient_time"
    assert on_time.requests[0].new_target_id == 901


def test_deadline_fallback_uses_same_reachability_gate() -> None:
    decision = decide(
        ReserveEchelonSkill(),
        step=300,
        members=(member(10, heading_east=0.0, heading_north=1.0),),
        deficits=(),
    )
    assert decision.requests == ()
    assert decision.rejected_reasons[1] == "turn_angle_exceeded"
    assert decision.packages[0].state == "STAGING"


def test_ordering_is_deterministic() -> None:
    members = (member(11), member(10))
    reserve_package = package(
        member_ids=(10, 11),
        member_types=(H_MISSILE_TYPE, H_MISSILE_TYPE),
    )
    deficits = (
        deficit(902, weight=1.0, missing_damage=4.0, position=(1.0, 0.0)),
        deficit(901, weight=1.0, missing_damage=4.0, position=(1.0, 0.0)),
    )
    first = decide(
        ReserveEchelonSkill(),
        members=members,
        deficits=deficits,
        packages=(reserve_package,),
    )
    second = decide(
        ReserveEchelonSkill(),
        members=tuple(reversed(members)),
        deficits=tuple(reversed(deficits)),
        packages=(reserve_package,),
    )
    assert first == second
    assert first.requests[0].new_target_id == 901


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("max_dispatch_stage", -1),
        ("max_dispatches", 0),
        ("deadline_step", -1),
        ("require_same_group_evidence", 1),
        ("max_target_distance_km", 0.0),
        ("max_target_distance_km", float("nan")),
        ("h_max_distance_km", 0.0),
        ("l_max_distance_km", float("inf")),
        ("h_max_turn_angle_degrees", 180.01),
        ("l_max_turn_angle_degrees", 0.0),
        ("seconds_per_step", 0.0),
        ("h_nominal_speed_mps", -1.0),
        ("l_nominal_speed_mps", float("nan")),
    ),
)
def test_config_validation_is_strict(field, value) -> None:
    with pytest.raises(ValueError):
        ReserveEchelonConfig(**{field: value})


def test_package_validation_rejects_future_hm_composition() -> None:
    mixed = ReservePackageSnapshot(
        1,
        (10, 11),
        (H_MISSILE_TYPE, 21001),
        "STAGING",
        900,
        999,
        "west",
        0,
    )
    with pytest.raises(ValueError, match="H-only and L-only"):
        decide(
            ReserveEchelonSkill(),
            members=(member(10), member(11, member_type=21001)),
            packages=(mixed,),
        )
