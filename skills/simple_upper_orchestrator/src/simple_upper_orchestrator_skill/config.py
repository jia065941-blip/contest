"""Configuration owned by the release adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from hm_public_strike_packages_skill import PublicStrikeConfig
from l_satellite_strike_packages_skill import SatelliteStrikeConfig
from staggered_deployment_skill import StaggeredDeploymentConfig


@dataclass(frozen=True)
class OrchestratorConfig:
    runtime_root: Path
    checkpoint: Path
    max_steps: int
    deployment: StaggeredDeploymentConfig = field(default_factory=StaggeredDeploymentConfig)
    deployment_batch_size: int = 10
    l_strike: SatelliteStrikeConfig = field(default_factory=SatelliteStrikeConfig)
    hm_strike: PublicStrikeConfig = field(default_factory=PublicStrikeConfig)

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.deployment_batch_size <= 0:
            raise ValueError("deployment_batch_size must be positive")
