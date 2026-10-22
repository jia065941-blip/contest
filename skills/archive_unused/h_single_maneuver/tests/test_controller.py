import pytest

from skill_generation.errors import StateError, ValidationError
from skill_generation.skills import ManeuverAction
from h_single_maneuver_skill import (
    HMissileSnapshot,
    HSingleManeuverConfig,
    HSingleManeuverSkill,
    ManeuverPhase,
    ThreatTrack,
    Vector2,
)


def snapshot(
    step: int,
    *,
    missile_id: int = 10,
    position: Vector2 = Vector2(0.0, 0.0),
    velocity: Vector2 = Vector2(2000.0, 0.0),
    goal: Vector2 = Vector2(100000.0, 0.0),
    threats=(),
):
    return HMissileSnapshot(
        step=step,
        missile_id=missile_id,
        position=position,
        velocity=velocity,
        goal_position=goal,
        threats=tuple(threats),
    )


def inbound(threat_id: int, east: float, north: float) -> ThreatTrack:
    return ThreatTrack(
        entity_id=threat_id,
        position=Vector2(east, north),
        velocity=Vector2(-2000.0, 0.0),
    )


def test_no_threat_delegates_to_stop_skill():
    skill = HSingleManeuverSkill()
    decision = skill.step(snapshot(0))
    assert decision.action is ManeuverAction.STOP
    assert decision.engine_acc_z == 0.0
    assert decision.phase is ManeuverPhase.CRUISE
    assert decision.reason == "no imminent conflict"


def test_left_side_threat_breaks_right_for_exact_duration_then_settles():
    skill = HSingleManeuverSkill(HSingleManeuverConfig(break_steps=2))
    threat = inbound(50, 12000.0, 1000.0)

    first = skill.step(snapshot(0, threats=(threat,)))
    second = skill.step(snapshot(1, threats=(threat,)))
    settle = skill.step(snapshot(2, threats=(threat,)))

    assert [first.action, second.action, settle.action] == [
        ManeuverAction.RIGHT,
        ManeuverAction.RIGHT,
        ManeuverAction.STOP,
    ]
    assert first.phase is second.phase is ManeuverPhase.EVADE
    assert settle.phase is ManeuverPhase.SETTLE


def test_right_side_threat_breaks_left():
    skill = HSingleManeuverSkill()
    decision = skill.step(snapshot(0, threats=(inbound(51, 12000.0, -1000.0),)))
    assert decision.action is ManeuverAction.LEFT
    assert decision.engine_acc_z == -1.0


def test_fixed_escape_direction_overrides_geometry():
    skill = HSingleManeuverSkill(HSingleManeuverConfig(escape_direction="right"))
    decision = skill.step(snapshot(0, threats=(inbound(53, 12000.0, -1000.0),)))
    assert decision.action is ManeuverAction.RIGHT


def test_recovery_turns_toward_goal_after_settle():
    config = HSingleManeuverConfig(break_steps=1, settle_steps=1, cooldown_steps=20)
    skill = HSingleManeuverSkill(config)
    threat = inbound(52, 12000.0, 1000.0)

    assert skill.step(snapshot(0, threats=(threat,))).action is ManeuverAction.RIGHT
    assert skill.step(snapshot(1)).action is ManeuverAction.STOP
    recovery = skill.step(
        snapshot(2, velocity=Vector2(1900.0, -400.0), goal=Vector2(100000.0, 0.0))
    )
    assert recovery.phase is ManeuverPhase.RECOVER
    assert recovery.action is ManeuverAction.LEFT
    assert recovery.reason == "bounded goal recovery"


def test_side_lock_prevents_rapid_escape_reversal():
    config = HSingleManeuverConfig(
        break_steps=1, settle_steps=1, cooldown_steps=0, side_lock_steps=8
    )
    skill = HSingleManeuverSkill(config)
    left_threat = inbound(60, 12000.0, 1000.0)
    right_threat = inbound(61, 12000.0, -1000.0)

    assert skill.step(snapshot(0, threats=(left_threat,))).action is ManeuverAction.RIGHT
    assert skill.step(snapshot(1)).action is ManeuverAction.STOP
    decision = skill.step(snapshot(2, threats=(right_threat,)))
    assert decision.action is ManeuverAction.RIGHT


def test_break_budget_ignores_later_non_emergency_tracks():
    config = HSingleManeuverConfig(
        break_steps=1,
        settle_steps=1,
        cooldown_steps=0,
        side_lock_steps=0,
        maximum_breaks=1,
    )
    skill = HSingleManeuverSkill(config)
    assert skill.step(snapshot(0, threats=(inbound(80, 12000.0, 1000.0),))).action is ManeuverAction.RIGHT
    assert skill.step(snapshot(1)).action is ManeuverAction.STOP
    decision = skill.step(snapshot(2, threats=(inbound(81, 12000.0, -1000.0),)))
    assert decision.action is ManeuverAction.STOP
    assert decision.reason == "goal corridor reacquired"


def test_terminal_phase_rejects_marginal_cpa_but_keeps_emergency():
    skill = HSingleManeuverSkill(HSingleManeuverConfig(break_steps=1))
    marginal = inbound(70, 8000.0, 2500.0)
    terminal = snapshot(0, goal=Vector2(10000.0, 0.0), threats=(marginal,))
    assert skill.step(terminal).action is ManeuverAction.STOP

    skill.reset()
    emergency = inbound(71, 500.0, 0.0)
    assert skill.step(snapshot(0, goal=Vector2(10000.0, 0.0), threats=(emergency,))).action in {
        ManeuverAction.LEFT,
        ManeuverAction.RIGHT,
    }


def test_one_instance_cannot_switch_missiles_or_replay_steps():
    skill = HSingleManeuverSkill()
    skill.step(snapshot(4, missile_id=10))
    with pytest.raises(StateError, match="missile_changed"):
        skill.step(snapshot(5, missile_id=11))
    with pytest.raises(StateError, match="non_monotonic_step"):
        skill.step(snapshot(4, missile_id=10))


@pytest.mark.parametrize(
    "values",
    [
        {"break_steps": 0},
        {"terminal_risk_scale": 1.1},
        {"prediction_horizon_seconds": float("nan")},
        {"unknown": 1},
    ],
)
def test_invalid_config_is_rejected(values):
    with pytest.raises(ValidationError):
        HSingleManeuverConfig.from_mapping(values)
