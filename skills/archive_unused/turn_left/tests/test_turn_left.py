from skill_generation.skills import ManeuverAction, ManeuverState
from turn_left_skill import TurnLeftSkill


def test_turn_left_for_requested_steps():
    skill = TurnLeftSkill().start(2)
    first, second = skill.step(), skill.step()
    assert first.action is second.action is ManeuverAction.LEFT
    assert first.engine_acc_z == second.engine_acc_z == -1.0
    assert not first.completed and second.completed
    assert skill.state is ManeuverState.COMPLETE
