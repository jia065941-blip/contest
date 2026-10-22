import pytest

from skill_generation.errors import StateError, ValidationError
from skill_generation.skills import (ManeuverAction, ManeuverState,
                                      TimedManeuverSkill)


def test_exact_execution_duration_and_completion():
    skill = TimedManeuverSkill(ManeuverAction.LEFT, -1.0).start(3)
    commands = [skill.step() for _ in range(3)]
    assert [command.action for command in commands] == [ManeuverAction.LEFT] * 3
    assert [command.engine_acc_z for command in commands] == [-1.0] * 3
    assert [command.remaining_steps for command in commands] == [2, 1, 0]
    assert [command.completed for command in commands] == [False, False, True]
    assert skill.state is ManeuverState.COMPLETE
    with pytest.raises(StateError):
        skill.step()


@pytest.mark.parametrize("value", [0, -1, 1.5, True, None])
def test_execution_steps_must_be_positive_integer(value):
    skill = TimedManeuverSkill(ManeuverAction.STOP, 0.0)
    with pytest.raises(ValidationError):
        skill.start(value)


def test_reset_and_restart_are_explicit():
    skill = TimedManeuverSkill(ManeuverAction.RIGHT, 1.0).start(2)
    with pytest.raises(StateError):
        skill.start(1)
    skill.reset()
    assert skill.state is ManeuverState.IDLE
    command = skill.start(1).step()
    assert command.completed and command.action is ManeuverAction.RIGHT
