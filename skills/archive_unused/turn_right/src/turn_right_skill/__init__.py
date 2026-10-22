"""Fixed-duration right maneuver Skill."""

from skill_generation.skills import ManeuverAction, TimedManeuverSkill


class TurnRightSkill(TimedManeuverSkill):
    def __init__(self) -> None:
        super().__init__(ManeuverAction.RIGHT, 1.0)


__all__ = ["TurnRightSkill"]
__version__ = "0.1.0"
