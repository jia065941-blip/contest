"""Read-only bridge to the author's original V13 deployment interfaces."""

from __future__ import annotations

import importlib
import hashlib
from pathlib import Path
import sys
from types import ModuleType

from .checkpoint import load_verified_checkpoint


def build_original_policy(contest_root: str | Path, checkpoint: str | Path,
                          *, max_steps: int):
    source = Path(contest_root).resolve(strict=True) / "policies/red/skills/evasion_ppo"
    if not source.is_dir():
        raise FileNotFoundError(f"Original evasion_ppo package is missing: {source}")
    package_name = "_local_avoid_v13_original"
    package = sys.modules.get(package_name)
    if package is None:
        package = ModuleType(package_name)
        package.__path__ = [str(source)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    local_ppo = importlib.import_module(f"{package_name}.local_ppo")
    local_env = importlib.import_module(f"{package_name}.local_env")
    local_runtime = importlib.import_module(f"{package_name}.local_runtime")
    payload = load_verified_checkpoint(checkpoint)
    ppo_values = dict(payload["ppo_config"])
    ppo_values["hidden_dims"] = tuple(ppo_values["hidden_dims"])
    ppo_values["device"] = "cpu"
    model = local_ppo.LocalAvoidPPO(local_ppo.LocalPPOConfig(**ppo_values))
    model.network.load_state_dict(payload["network"], strict=True)
    model.actor_rms.load_state_dict(payload["actor_rms"])
    model.critic_rms.load_state_dict(payload["critic_rms"])
    model.network.eval()
    environment_values = dict(payload["environment_config"])
    environment_values["unit_speeds_mps"] = tuple(environment_values["unit_speeds_mps"])
    environment_values["unit_turn_rates_deg_s"] = tuple(environment_values["unit_turn_rates_deg_s"])
    policy = local_runtime.LocalAvoidRuntimePolicy(max_steps=max_steps)
    policy.model = model
    policy.environment_config = local_env.LocalAvoidConfig(**environment_values)
    return policy


def build_external_policy(contest_root: str | Path, checkpoint: str | Path,
                          *, max_steps: int):
    """Load another compatible local runtime without V13's fixed hash gate."""
    root = Path(contest_root).resolve(strict=True)
    source = root / "policies/red/skills/evasion_ppo"
    if not source.is_dir():
        raise FileNotFoundError(f"External evasion_ppo package is missing: {source}")
    # Newer runtimes retain absolute imports rooted at their checkout (for
    # example ``policies.red.learning``), not just imports within evasion_ppo.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    suffix = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:12]
    package_name = f"_local_avoid_external_{suffix}"
    package = sys.modules.get(package_name)
    if package is None:
        package = ModuleType(package_name)
        package.__path__ = [str(source)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    local_runtime = importlib.import_module(f"{package_name}.local_runtime")
    policy = local_runtime.LocalAvoidRuntimePolicy(max_steps=int(max_steps))
    policy.load(str(Path(checkpoint).resolve(strict=True)))
    policy.set_training(False)
    return policy
