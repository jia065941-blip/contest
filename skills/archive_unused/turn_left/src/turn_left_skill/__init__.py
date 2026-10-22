"""Fixed-duration left maneuver Skill."""

from skill_generation.skills import ManeuverAction, TimedManeuverSkill


class TurnLeftSkill(TimedManeuverSkill):
    def __init__(self) -> None:
        super().__init__(ManeuverAction.LEFT, -1.0)


__all__ = ["TurnLeftSkill"]
__version__ = "0.1.0"
