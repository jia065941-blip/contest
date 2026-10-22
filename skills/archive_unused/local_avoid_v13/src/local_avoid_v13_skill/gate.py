"""Mission-v12 action gate kept outside the numerical Backend."""

from dataclasses import dataclass
import math

import numpy as np

from .encoder import EnvironmentConfig


LEFT, STOP, RIGHT = 0, 1, 2


@dataclass(frozen=True)
class MissionGateConfig:
    recovery_deadband_deg: float = 4.0
    terminal_goal_distance_m: tuple[float, float, float] = (12_000.0, 10_000.0, 8_000.0)
    late_flight_fraction: float = 0.85
    engage_distance_m: tuple[float, float, float] = (1_000.0, 850.0, 600.0)
    engage_t_cpa_s: tuple[float, float, float] = (45.0, 32.0, 20.0)
    engage_d_cpa_m: tuple[float, float, float] = (3_000.0, 2_000.0, 900.0)
    emergency_distance_m: tuple[float, float, float] = (750.0, 600.0, 450.0)
    terminal_risk_scale: float = 0.65


def _only(action: int) -> np.ndarray:
    result = np.zeros(3, dtype=np.bool_)
    result[action] = True
    return result


def prepare_action_mask(observation: np.ndarray, environment: EnvironmentConfig,
                        recovery_pending: bool,
                        gate: MissionGateConfig = MissionGateConfig()) -> tuple[np.ndarray, bool]:
    encoded_type = observation[30:33]
    type_slot = int(np.argmax(encoded_type)) if encoded_type.max(initial=0.0) > 0.5 else 0
    goal_distance = float(observation[3]) * environment.goal_distance_scale_m
    terminal = goal_distance <= gate.terminal_goal_distance_m[type_slot] or observation[29] >= gate.late_flight_fraction
    scale = gate.terminal_risk_scale if terminal else 1.0
    urgent = False
    for offset in (6, 15):
        if observation[offset] <= 0.5:
            continue
        distance = float(observation[offset + 1]) * environment.threat_distance_scale_m
        closing = float(observation[offset + 5]) * environment.relative_speed_scale_mps
        time_to_cpa = float(observation[offset + 6]) * environment.t_cpa_scale_s
        cpa_distance = float(observation[offset + 7]) * environment.threat_distance_scale_m
        urgent = urgent or distance <= gate.emergency_distance_m[type_slot]
        urgent = urgent or distance <= gate.engage_distance_m[type_slot] * scale
        urgent = urgent or (closing > 0.0 and 0.0 <= time_to_cpa <= gate.engage_t_cpa_s[type_slot] * scale
                            and cpa_distance <= gate.engage_d_cpa_m[type_slot] * scale)
    if urgent:
        return np.ones(3, dtype=np.bool_), recovery_pending
    if not recovery_pending:
        return _only(STOP), False
    goal_error = math.atan2(float(observation[4]), float(observation[5]))
    if abs(goal_error) <= math.radians(gate.recovery_deadband_deg):
        return _only(STOP), False
    return _only(LEFT if goal_error > 0.0 else RIGHT), True
