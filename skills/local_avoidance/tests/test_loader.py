from pathlib import Path

import pytest

from local_avoidance_skill import load_external_policy


def runtime_tree(tmp_path: Path, body: str) -> tuple[Path, Path]:
    source = tmp_path / "policies/red/skills/evasion_ppo"
    source.mkdir(parents=True)
    (source / "local_runtime.py").write_text(body, encoding="utf-8")
    checkpoint = tmp_path / "network.pt"
    checkpoint.write_bytes(b"caller-owned")
    return tmp_path, checkpoint


def test_loads_caller_runtime_and_checkpoint_without_model_gate(tmp_path: Path) -> None:
    root, checkpoint = runtime_tree(tmp_path, '''
class LocalAvoidRuntimePolicy:
    def __init__(self, max_steps): self.max_steps=max_steps; self.loaded=None; self.training=True
    def load(self, checkpoint): self.loaded=checkpoint
    def set_training(self, training): self.training=training
    def register_agent(self, *args, **kwargs): pass
    def encode_motion_observation(self, *args, **kwargs): return ()
    def prepare_action_mask(self, observation, mask): return mask
    def select_action(self, observation, mask): return 1
''')

    policy = load_external_policy(root, checkpoint, max_steps=3000)

    assert policy.max_steps == 3000
    assert policy.loaded == str(checkpoint.resolve())
    assert policy.training is False


def test_rejects_missing_contract_method(tmp_path: Path) -> None:
    root, checkpoint = runtime_tree(tmp_path, '''
class LocalAvoidRuntimePolicy:
    def __init__(self, max_steps): self.max_steps=max_steps
    def load(self, checkpoint): pass
    def set_training(self, training): pass
''')
    with pytest.raises(RuntimeError, match="register_agent"):
        load_external_policy(root, checkpoint, max_steps=10)


def test_validates_steps_and_runtime_layout(tmp_path: Path) -> None:
    checkpoint = tmp_path / "network.pt"
    checkpoint.write_bytes(b"x")
    with pytest.raises(ValueError, match="positive integer"):
        load_external_policy(tmp_path, checkpoint, max_steps=0)
    with pytest.raises(FileNotFoundError, match="evasion_ppo"):
        load_external_policy(tmp_path, checkpoint, max_steps=1)
