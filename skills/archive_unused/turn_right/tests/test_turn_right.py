from skill_generation.skills import ManeuverAction, ManeuverState
from turn_right_skill import TurnRightSkill


def test_turn_right_for_requested_steps():
    skill = TurnRightSkill().start(2)
    first, second = skill.step(), skill.step()
    assert first.action is second.action is ManeuverAction.RIGHT
    assert first.engine_acc_z == second.engine_acc_z == 1.0
    assert not first.completed and second.completed
    assert skill.state is ManeuverState.COMPLETE
