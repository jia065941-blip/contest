"""Load compatible caller-supplied local-avoidance runtimes and checkpoints."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class LocalAvoidancePolicy(Protocol):
    max_steps: int

    def load(self, checkpoint: str) -> None: ...
    def set_training(self, training: bool) -> None: ...
    def register_agent(self, agent_id: int, entity_id: int, max_steps: int | None = None) -> None: ...
    def encode_motion_observation(
        self, observation: Mapping[str, Any], *, agent_id: int, entity_id: int,
        launch_step: int, maneuver_state: int, goal_position: Any,
    ) -> Any: ...
    def prepare_action_mask(self, observation: Any, base_mask: Any) -> Any: ...
    def select_action(self, observation: Any, action_mask: Any) -> int: ...


def load_external_policy(
    runtime_root: str | Path,
    checkpoint: str | Path,
    *,
    max_steps: int,
) -> LocalAvoidancePolicy:
    """Load an external runtime without imposing a model name or checkpoint hash."""
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    root = Path(runtime_root).expanduser().resolve(strict=True)
    checkpoint_path = Path(checkpoint).expanduser().resolve(strict=True)
    source = root / "policies/red/skills/evasion_ppo"
    if not source.is_dir():
        raise FileNotFoundError(f"External evasion_ppo package is missing: {source}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    suffix = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:16]
    package_name = f"_local_avoidance_external_{suffix}"
    package = sys.modules.get(package_name)
    if package is None:
        package = ModuleType(package_name)
        package.__path__ = [str(source)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    local_runtime = importlib.import_module(f"{package_name}.local_runtime")
    runtime_type = getattr(local_runtime, "LocalAvoidRuntimePolicy", None)
    if runtime_type is None:
        raise RuntimeError("External runtime does not define LocalAvoidRuntimePolicy")
    policy = runtime_type(max_steps=max_steps)
    for method in ("load", "set_training", "register_agent", "encode_motion_observation", "prepare_action_mask", "select_action"):
        if not callable(getattr(policy, method, None)):
            raise RuntimeError(f"External policy is missing required method: {method}")
    policy.load(str(checkpoint_path))
    policy.set_training(False)
    return policy
