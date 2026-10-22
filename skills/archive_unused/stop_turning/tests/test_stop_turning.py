from skill_generation.skills import ManeuverAction, ManeuverState
from stop_turning_skill import StopTurningSkill


def test_stop_turning_for_requested_steps():
    skill = StopTurningSkill().start(2)
    first, second = skill.step(), skill.step()
    assert first.action is second.action is ManeuverAction.STOP
    assert first.engine_acc_z == second.engine_acc_z == 0.0
    assert not first.completed and second.completed
    assert skill.state is ManeuverState.COMPLETE
