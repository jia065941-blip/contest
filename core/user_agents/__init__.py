from .base_agent import base_agent
from .attack_missile_agent import AttackMissileAgent
from .deploy_agent import DeployAgent
"""Core agents plus locally registered extension packages."""

from pathlib import Path


# Keep extension source outside the engine tree.  This allows the blue policy
# package to evolve independently while legacy imports (``user_agents.*``)
# continue to work.
_plugin_agents = Path(__file__).resolve().parents[2] / "plugins" / "blue_baselines" / "src" / "user_agents"
if _plugin_agents.is_dir():
    __path__.append(str(_plugin_agents))
