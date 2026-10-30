"""Configuration owned by the release adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

from dynamic_retarget_skill import DynamicRetargetConfig
from hm_public_strike_packages_skill import PublicStrikeConfig
from l_satellite_strike_packages_skill import SatelliteStrikeConfig
from staggered_deployment_skill import StaggeredDeploymentConfig
from terminal_guidance_override_skill import TerminalGuidanceConfig


@dataclass(frozen=True)
class OrchestratorConfig:
    runtime_root: Path
    checkpoint: Path
    max_steps: int
    deployment: StaggeredDeploymentConfig = field(default_factory=StaggeredDeploymentConfig)
    deployment_batch_size: int = 10
    l_strike: SatelliteStrikeConfig = field(default_factory=SatelliteStrikeConfig)
    hm_strike: PublicStrikeConfig = field(default_factory=PublicStrikeConfig)
    reserve_h_count: int = 0
    l_first_wave_per_target: int = 0
    reserve_deadline_step: int = 450
    reserve_l_deadline_step: int = 250
    reserve_max_stage: int | None = None
    reserve_h_max_distance_km: float = 80.0
    reserve_l_max_distance_km: float = 100.0
    reserve_h_max_turn_angle_degrees: float | None = None
    reserve_l_max_turn_angle_degrees: float | None = None
    reserve_h_turn_rate_degrees_per_second: float = 20.0
    reserve_l_turn_rate_degrees_per_second: float = 15.0
    reserve_turn_safety_seconds: float = 5.0
    reserve_holding_half_width_km: float = 6.0
    reserve_holding_heading_offset_degrees: float = 35.0
    reserve_holding_max_turn_steps: int = 6
    reserve_holding_straight_steps: int = 2
    dynamic_retarget: DynamicRetargetConfig = field(default_factory=DynamicRetargetConfig)
    terminal_guidance: TerminalGuidanceConfig = field(
        default_factory=TerminalGuidanceConfig
    )
    retarget_track_max_age_steps: float = 5
    dynamic_retarget_enabled: bool = True
    terminal_guidance_enabled: bool = False
    audit_path: Path | None = None
    formation_audit_interval_steps: int = 25
    missile_trace_interval_steps: int = 0
    missile_terminal_history: int = 20
    seconds_per_step: float = 1.0
    velocity_window_steps: int = 5

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.deployment_batch_size <= 0:
            raise ValueError("deployment_batch_size must be positive")
        for name in (
            "reserve_h_count",
            "l_first_wave_per_target",
            "reserve_deadline_step",
            "reserve_l_deadline_step",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.reserve_max_stage is not None and (
            isinstance(self.reserve_max_stage, bool)
            or not isinstance(self.reserve_max_stage, int)
            or self.reserve_max_stage < 0
        ):
            raise ValueError("reserve_max_stage must be a nonnegative integer or None")
        for name in (
            "reserve_h_max_distance_km",
            "reserve_l_max_distance_km",
            "reserve_h_turn_rate_degrees_per_second",
            "reserve_l_turn_rate_degrees_per_second",
            "reserve_holding_half_width_km",
            "reserve_holding_heading_offset_degrees",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
        for name in (
            "reserve_h_max_turn_angle_degrees",
            "reserve_l_max_turn_angle_degrees",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
                or value > 180
            ):
                raise ValueError(f"{name} must be in (0, 180] or None")
        if (
            isinstance(self.reserve_turn_safety_seconds, bool)
            or not isinstance(self.reserve_turn_safety_seconds, (int, float))
            or not math.isfinite(float(self.reserve_turn_safety_seconds))
            or self.reserve_turn_safety_seconds < 0
        ):
            raise ValueError("reserve_turn_safety_seconds must be nonnegative and finite")
        if self.reserve_holding_heading_offset_degrees >= 90:
            raise ValueError("reserve_holding_heading_offset_degrees must be below 90")
        for name in (
            "reserve_holding_max_turn_steps",
            "reserve_holding_straight_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hm_strike.reserve_m != 0:
            raise ValueError("reserve echelon 0.4.0 does not support M reserves")
        if (
            self.reserve_h_count
            and self.hm_strike.reserve_h not in (0, self.reserve_h_count)
        ):
            raise ValueError("reserve_h_count conflicts with hm_strike.reserve_h")
        if (
            isinstance(self.retarget_track_max_age_steps, bool)
            or not isinstance(self.retarget_track_max_age_steps, (int, float))
            or not math.isfinite(float(self.retarget_track_max_age_steps))
            or self.retarget_track_max_age_steps < 0
        ):
            raise ValueError("retarget_track_max_age_steps must be nonnegative and finite")
        if not isinstance(self.dynamic_retarget_enabled, bool):
            raise ValueError("dynamic_retarget_enabled must be a boolean")
        if not isinstance(self.terminal_guidance, TerminalGuidanceConfig):
            raise ValueError("terminal_guidance must be a TerminalGuidanceConfig")
        if not isinstance(self.terminal_guidance_enabled, bool):
            raise ValueError("terminal_guidance_enabled must be a boolean")
        if self.audit_path is not None and not isinstance(self.audit_path, Path):
            raise ValueError("audit_path must be a Path or None")
        if (
            isinstance(self.formation_audit_interval_steps, bool)
            or not isinstance(self.formation_audit_interval_steps, int)
            or self.formation_audit_interval_steps <= 0
        ):
            raise ValueError("formation_audit_interval_steps must be a positive integer")
        if (
            isinstance(self.missile_trace_interval_steps, bool)
            or not isinstance(self.missile_trace_interval_steps, int)
            or self.missile_trace_interval_steps < 0
        ):
            raise ValueError("missile_trace_interval_steps must be a nonnegative integer")
        if (
            isinstance(self.missile_terminal_history, bool)
            or not isinstance(self.missile_terminal_history, int)
            or self.missile_terminal_history <= 0
        ):
            raise ValueError("missile_terminal_history must be a positive integer")
        if (
            isinstance(self.seconds_per_step, bool)
            or not isinstance(self.seconds_per_step, (int, float))
            or not math.isfinite(float(self.seconds_per_step))
            or self.seconds_per_step <= 0
        ):
            raise ValueError("seconds_per_step must be positive and finite")
        if (
            isinstance(self.velocity_window_steps, bool)
            or not isinstance(self.velocity_window_steps, int)
            or self.velocity_window_steps < 2
        ):
            raise ValueError("velocity_window_steps must be an integer at least 2")
