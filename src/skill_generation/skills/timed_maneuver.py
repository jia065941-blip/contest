"""Reusable lifecycle for deterministic, fixed-duration maneuver Skills."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

from ..errors import StateError, ValidationError


class ManeuverAction(IntEnum):
    LEFT = 0
    STOP = 1
    RIGHT = 2


class ManeuverState(str, Enum):
    IDLE = "IDLE"
    ACTIVE = "ACTIVE"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True)
class ManeuverCommand:
    action: ManeuverAction
    engine_acc_z: float
    remaining_steps: int
    completed: bool


class TimedManeuverSkill:
    """Emit one fixed maneuver command for an exact number of simulator steps."""

    def __init__(self, action: ManeuverAction, engine_acc_z: float) -> None:
        self.action = action
        self.engine_acc_z = float(engine_acc_z)
        self.state = ManeuverState.IDLE
        self.total_steps = 0
        self.remaining_steps = 0

    def start(self, execution_steps: int) -> "TimedManeuverSkill":
        if type(execution_steps) is not int or execution_steps <= 0:
            raise ValidationError("invalid_execution_steps", "execution_steps",
                                  "positive integer", type(execution_steps).__name__)
        if self.state is ManeuverState.ACTIVE:
            raise StateError("already_active", "state", "IDLE or COMPLETE", self.state.value)
        self.total_steps = execution_steps
        self.remaining_steps = execution_steps
        self.state = ManeuverState.ACTIVE
        return self

    def step(self) -> ManeuverCommand:
        if self.state is not ManeuverState.ACTIVE:
            raise StateError("not_active", "state", ManeuverState.ACTIVE.value, self.state.value)
        self.remaining_steps -= 1
        completed = self.remaining_steps == 0
        if completed:
            self.state = ManeuverState.COMPLETE
        return ManeuverCommand(
            action=self.action,
            engine_acc_z=self.engine_acc_z,
            remaining_steps=self.remaining_steps,
            completed=completed,
        )

    def reset(self) -> None:
        self.total_steps = 0
        self.remaining_steps = 0
        self.state = ManeuverState.IDLE
