from .base_agent import base_agent
from .attack_missile_agent import AttackMissileAgent
from .deploy_agent import DeployAgent

from pathlib import Path


# Expose the blue strategy package without copying plugin policy code into the
# engine package. The policy implementation remains owned by the plugin.
_plugin_agents = Path(__file__).resolve().parents[2] / "plugins" / "blue_baselines" / "src" / "user_agents"
if _plugin_agents.is_dir():
    __path__.append(str(_plugin_agents))
