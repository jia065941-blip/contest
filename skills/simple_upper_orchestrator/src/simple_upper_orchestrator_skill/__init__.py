"""Official-release wiring for the simple H/M/L upper layer."""

from .config import OrchestratorConfig

__all__ = ["OrchestratorConfig", "PreMissionPlan"]

__version__ = "2.7.0"


def __getattr__(name):
    if name != "PreMissionPlan":
        raise AttributeError(name)
    from .deploy_agent import PreMissionPlan

    return PreMissionPlan
