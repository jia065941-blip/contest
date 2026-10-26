from dataclasses import FrozenInstanceError, replace
import math

import pytest

from reserve_holding_maneuver_skill import (
    H_MISSILE_TYPE,
    LEFT,
    RIGHT,
    STRAIGHT,
    HoldingSnapshot,
    ReserveHoldingManeuverConfig,
    ReserveHoldingManeuverSkill,
    __version__,
)


KM_PER_DEGREE = 111.195


def snapshot(
    *,
    entity_id: int = 1,
    entity_type: int = H_MISSILE_TYPE,
    step: int = 0,
    position: tuple[float, float] = (0.01, 0.0),
    velocity: tuple[float, float] = (200.0, 0.0),
    anchor: tuple[float, float] = (0.0, 0.0),
    forward_target: tuple[float, float] = (0.1, 0.0),
    baseline_action: int = STRAIGHT,
    state: str = "STAGING",
) -> HoldingSnapshot:
    return HoldingSnapshot(
        entity_id=entity_id,
        type=entity_type,
        step=step,
        position=position,
        velocity=velocity,
        anchor=anchor,
        forward_target=forward_target,
        baseline_action=baseline_action,
        state=state,
    )


def test_version_frozen_dtos_and_default_config() -> None:
    item = snapshot()
    decision = ReserveHoldingManeuverSkill().decide(item)

    with pytest.raises(FrozenInstanceError):
        item.step = 2
    with pytest.raises(FrozenInstanceError):
        decision.action = STRAIGHT

    config = ReserveHoldingManeuverConfig()
    assert __version__ == "0.1.0"
    assert config.enabled_type == 21000
    assert config.corridor_half_width_km == pytest.approx(6.0)
    assert config.heading_offset_deg == pytest.approx(35.0)
    assert config.max_turn_steps == 6
    assert config.straight_transition_steps == 2
    assert config.latch is True


@pytest.mark.parametrize(
    ("forward_target", "position", "velocity"),
    (
        ((0.1, 0.0), (0.01, 0.0), (200.0, 0.0)),
        ((-0.1, 0.0), (-0.01, 0.0), (-200.0, 0.0)),
        ((0.0, 0.1), (0.0, 0.01), (0.0, 200.0)),
        ((0.0, -0.1), (0.0, -0.01), (0.0, -200.0)),
    ),
)
def test_four_forward_axes_advance_and_start_left_leg(
    forward_target, position, velocity
) -> None:
    decision = ReserveHoldingManeuverSkill().decide(snapshot(
        forward_target=forward_target,
        position=position,
        velocity=velocity,
    ))

    assert decision.forward_progress == pytest.approx(1.11195, rel=2e-4)
    assert decision.cross_track == pytest.approx(0.0, abs=1e-9)
    assert decision.phase == "left_leg"
    assert decision.action == LEFT


@pytest.mark.parametrize(
    ("forward_target", "position", "velocity"),
    (
        ((0.1, 0.0), (0.01, 1.0 / KM_PER_DEGREE), (200.0, 0.0)),
        ((-0.1, 0.0), (-0.01, -1.0 / KM_PER_DEGREE), (-200.0, 0.0)),
        ((0.0, 0.1), (-1.0 / KM_PER_DEGREE, 0.01), (0.0, 200.0)),
        ((0.0, -0.1), (1.0 / KM_PER_DEGREE, -0.01), (0.0, -200.0)),
    ),
)
def test_cross_track_positive_is_left_of_each_forward_axis(
    forward_target, position, velocity
) -> None:
    decision = ReserveHoldingManeuverSkill().decide(snapshot(
        forward_target=forward_target,
        position=position,
        velocity=velocity,
    ))

    assert decision.forward_progress > 1.0
    assert decision.cross_track == pytest.approx(1.0, rel=2e-4)


def test_upper_boundary_has_two_straight_steps_before_right_leg() -> None:
    skill = ReserveHoldingManeuverSkill()
    initial = skill.decide(snapshot(step=0))
    upper = (0.02, 6.1 / KM_PER_DEGREE)
    first_transition = skill.decide(snapshot(step=1, position=upper))
    second_transition = skill.decide(snapshot(step=2, position=upper))
    reversed_leg = skill.decide(snapshot(step=3, position=upper))

    assert initial.phase == "left_leg"
    assert initial.action == LEFT
    assert [first_transition.action, second_transition.action] == [
        STRAIGHT,
        STRAIGHT,
    ]
    assert [first_transition.phase, second_transition.phase] == [
        "transition",
        "transition",
    ]
    assert first_transition.reason == "upper_boundary_transition"
    assert second_transition.reason == "transition"
    assert reversed_leg.phase == "right_leg"
    assert reversed_leg.action == RIGHT


def test_lower_boundary_flips_right_leg_back_through_transition() -> None:
    skill = ReserveHoldingManeuverSkill()
    upper = (0.02, 6.1 / KM_PER_DEGREE)
    lower = (0.04, -6.1 / KM_PER_DEGREE)

    skill.decide(snapshot(step=0))
    skill.decide(snapshot(step=1, position=upper))
    skill.decide(snapshot(step=2, position=upper))
    skill.decide(snapshot(step=3, position=upper))
    first = skill.decide(snapshot(step=4, position=lower))
    second = skill.decide(snapshot(step=5, position=lower))
    left_again = skill.decide(snapshot(step=6, position=lower))

    assert first.phase == "transition"
    assert first.reason == "lower_boundary_transition"
    assert second.phase == "transition"
    assert left_again.phase == "left_leg"
    assert left_again.action == LEFT


def test_consecutive_turns_never_exceed_configured_limit() -> None:
    skill = ReserveHoldingManeuverSkill(ReserveHoldingManeuverConfig(
        max_turn_steps=2,
    ))

    decisions = [
        skill.decide(snapshot(step=step))
        for step in range(5)
    ]

    assert [item.action for item in decisions] == [
        LEFT,
        LEFT,
        STRAIGHT,
        LEFT,
        LEFT,
    ]
    assert decisions[2].reason == "max_turn_steps_guard"


@pytest.mark.parametrize("entity_type", (21001, 21002, 9400))
def test_non_h_types_pass_through(entity_type: int) -> None:
    decision = ReserveHoldingManeuverSkill().decide(snapshot(
        entity_type=entity_type,
        baseline_action=RIGHT,
    ))

    assert decision.action == RIGHT
    assert decision.overridden is False
    assert decision.reason == "type_passthrough"


@pytest.mark.parametrize("state", ("DISPATCHED", "FALLBACK"))
def test_non_staging_h_passes_through(state: str) -> None:
    decision = ReserveHoldingManeuverSkill().decide(snapshot(
        state=state,
        baseline_action=RIGHT,
    ))

    assert decision.action == RIGHT
    assert decision.overridden is False
    assert decision.reason == "state_passthrough"


def test_reset_clears_owner_transition_state() -> None:
    skill = ReserveHoldingManeuverSkill()
    upper = (0.02, 6.1 / KM_PER_DEGREE)
    skill.decide(snapshot(step=0))
    assert skill.decide(snapshot(step=1, position=upper)).phase == "transition"

    skill.reset(1)
    after_reset = skill.decide(snapshot(step=2, position=upper))

    assert after_reset.phase == "right_leg"
    assert after_reset.action == RIGHT


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("enabled_type", 21001),
        ("corridor_half_width_km", 0.0),
        ("corridor_half_width_km", math.nan),
        ("heading_offset_deg", 0.0),
        ("heading_offset_deg", 90.0),
        ("max_turn_steps", 0),
        ("straight_transition_steps", 0),
        ("latch", 1),
    ),
)
def test_config_validation_is_strict(field, value) -> None:
    with pytest.raises(ValueError):
        ReserveHoldingManeuverConfig(**{field: value})


@pytest.mark.parametrize(
    "invalid",
    (
        {"entity_id": True},
        {"step": -1},
        {"position": [0.0, 0.0]},
        {"velocity": (0.0, 0.0)},
        {"anchor": (181.0, 0.0)},
        {"forward_target": (0.0, 0.0)},
        {"baseline_action": 3},
        {"state": "ACTIVE"},
    ),
)
def test_snapshot_validation_is_strict(invalid) -> None:
    with pytest.raises(ValueError):
        snapshot(**invalid)


def test_decide_and_reset_reject_wrong_types() -> None:
    skill = ReserveHoldingManeuverSkill()
    with pytest.raises(ValueError, match="HoldingSnapshot"):
        skill.decide(object())
    with pytest.raises(ValueError, match="nonnegative integer"):
        skill.reset(True)


def test_latch_false_discards_owner_on_passthrough() -> None:
    skill = ReserveHoldingManeuverSkill(ReserveHoldingManeuverConfig(latch=False))
    upper = (0.02, 6.1 / KM_PER_DEGREE)
    skill.decide(snapshot(step=0))
    skill.decide(snapshot(step=1, position=upper))
    skill.decide(snapshot(step=2, state="DISPATCHED"))

    resumed = skill.decide(snapshot(step=3))

    assert resumed.phase == "left_leg"
    assert resumed.action == LEFT
