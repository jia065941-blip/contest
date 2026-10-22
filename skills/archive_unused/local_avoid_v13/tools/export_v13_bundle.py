"""Convert the verified V13 checkpoint to a non-pickle Actor-only bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from numpy._core.multiarray import _reconstruct, scalar


EXPECTED_SHA256 = "632662d47fb35edfdb10d680a359cced03fab75af09cff666631a97f0ccf87f3"
ACTOR_KEYS = (
    "actor.0.weight", "actor.0.bias", "actor.2.weight", "actor.2.bias",
    "actor.4.weight", "actor.4.bias", "actor.6.weight", "actor.6.bias",
)


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def _safe_payload(path: Path) -> dict:
    safe = [_reconstruct, scalar, np.ndarray, np.dtype,
            type(np.dtype(np.float64)), type(np.dtype(np.float32))]
    with torch.serialization.safe_globals(safe):
        return torch.load(path, map_location="cpu", weights_only=True)


def export(checkpoint: Path, output: Path) -> None:
    checkpoint = checkpoint.resolve(strict=True)
    actual_sha = _digest(checkpoint)
    if actual_sha != EXPECTED_SHA256:
        raise ValueError(f"Unexpected V13 checkpoint SHA-256: {actual_sha}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite bundle: {output}")
    payload = _safe_payload(checkpoint)
    if payload.get("algorithm") != "local_avoid_asymmetric_ppo_v1":
        raise ValueError("Unexpected checkpoint algorithm")
    ppo = dict(payload["ppo_config"])
    expected_config = {
        "actor_obs_dim": 33, "critic_state_dim": 43, "action_dim": 3,
        "hidden_dims": [256, 256, 128], "normalize_observation": True,
    }
    for name, expected in expected_config.items():
        actual = list(ppo[name]) if name == "hidden_dims" else ppo[name]
        if actual != expected:
            raise ValueError(f"Unexpected ppo_config.{name}: {actual!r}")
    network = payload["network"]
    if not set(ACTOR_KEYS) <= set(network):
        raise ValueError("Checkpoint Actor state is incomplete")
    actor_rms = payload["actor_rms"]
    arrays = {name: network[name].detach().cpu().numpy().astype(np.float32, copy=True)
              for name in ACTOR_KEYS}
    arrays.update({
        "actor_rms.mean": np.asarray(actor_rms["mean"], dtype=np.float64),
        "actor_rms.var": np.asarray(actor_rms["var"], dtype=np.float64),
        "actor_rms.count": np.asarray(actor_rms["count"], dtype=np.float64),
    })
    artifacts = output / "artifacts"
    artifacts.mkdir(parents=True)
    parameters = artifacts / "actor_parameters.npz"
    np.savez(parameters, **arrays)
    config = artifacts / "environment_config.json"
    config.write_text(json.dumps({
        "source_checkpoint_sha256": actual_sha,
        "algorithm": payload["algorithm"],
        "stage": int(payload["stage"]),
        "total_steps": int(payload["total_steps"]),
        "ppo_config": ppo,
        "environment_config": dict(payload["environment_config"]),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "format_version": 1,
        "model_id": "local-avoid-v13",
        "version": "13.0.0",
        "backend": {"id": "local-avoid-v13-torch", "interface_version": "1"},
        "artifacts": [
            {"name": "actor_parameters", "path": "artifacts/actor_parameters.npz",
             "sha256": _digest(parameters)},
            {"name": "environment_config", "path": "artifacts/environment_config.json",
             "sha256": _digest(config)},
        ],
        "inputs": [{"name": "observation", "dtype": "float32", "shape": [33], "finite": True}],
        "outputs": [{"name": "logits", "dtype": "float32", "shape": [3], "finite": True}],
        "contract_version": "local-avoid-v13-logits-v1",
        "preprocessing": {"required": True, "artifact": "actor_parameters",
                          "version": "running-mean-std-v1"},
        "model_format": "local-avoid-v13-actor-npz-v1",
        "dependencies": {"numpy": np.__version__, "torch": torch.__version__},
        "source": f"Actor-only safe conversion of V13 checkpoint SHA-256 {actual_sha}",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
