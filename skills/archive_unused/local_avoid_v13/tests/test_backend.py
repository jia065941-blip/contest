import json

import numpy as np
import pytest

from skill_generation.artifacts import load_bundle

from local_avoid_v13_skill import LocalAvoidV13Skill, create_v13_registry, default_bundle_path
from local_avoid_v13_skill.checkpoint import load_verified_checkpoint


def test_bundle_identity_and_static_contract():
    bundle = load_bundle(default_bundle_path())
    assert bundle.manifest.model_id == "local-avoid-v13"
    assert bundle.manifest.version == "13.0.0"
    assert len(bundle.fingerprint) == 64
    create_v13_registry().validate(bundle)
    config = json.loads((bundle.root / "artifacts/environment_config.json").read_text())
    assert config["source_checkpoint_sha256"] == (
        "632662d47fb35edfdb10d680a359cced03fab75af09cff666631a97f0ccf87f3")
    assert config["stage"] == 3 and config["total_steps"] == 65536


def test_installed_actor_has_stable_known_output_and_owned_result():
    with LocalAvoidV13Skill() as skill:
        first = skill.infer({"observation": np.zeros(33, dtype=np.float32)})
        saved = first.outputs["logits"].copy()
        second = skill.infer({"observation": np.ones(33, dtype=np.float32)})
    np.testing.assert_allclose(saved, [-1.0860736, 1.6324025, -0.99082255],
                               rtol=0, atol=1e-7)
    np.testing.assert_array_equal(first.outputs["logits"], saved)
    assert not np.shares_memory(first.outputs["logits"], second.outputs["logits"])
    assert int(np.argmax(first.outputs["logits"])) == 1


def test_input_is_not_modified():
    observation = np.linspace(-1, 1, 33, dtype=np.float32)
    original = observation.copy()
    with LocalAvoidV13Skill() as skill:
        skill.infer({"observation": observation})
    np.testing.assert_array_equal(observation, original)


def test_checkpoint_reader_rejects_any_other_file():
    artifact = default_bundle_path() / "artifacts" / "actor_parameters.npz"
    with pytest.raises(ValueError, match="Unexpected V13 checkpoint SHA-256"):
        load_verified_checkpoint(artifact)
