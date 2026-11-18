"""场景检查、配置加载、部署约束和比赛计分。"""

from .config import DeploymentConstraints, ScenarioConfig

from .loader import (
    CASES_ROOT,
    SCENARIO_ROOT,
    load_scenario,
    read_profile,
    resolve_scenario_path,
)
from .deployment import resolve_deployment_constraints

__all__ = [
    "CASES_ROOT", "SCENARIO_ROOT", "ScenarioConfig", "DeploymentConstraints",
    "load_scenario", "read_profile", "resolve_scenario_path",
    "resolve_deployment_constraints",
]
