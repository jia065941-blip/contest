"""Fixed-duration stop-turning Skill."""

from skill_generation.skills import ManeuverAction, TimedManeuverSkill


class StopTurningSkill(TimedManeuverSkill):
    def __init__(self) -> None:
        super().__init__(ManeuverAction.STOP, 0.0)


__all__ = ["StopTurningSkill"]
__version__ = "0.1.0"
