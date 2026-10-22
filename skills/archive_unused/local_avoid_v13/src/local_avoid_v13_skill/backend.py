"""Trusted CPU backend for the converted, data-only V13 Actor."""

from __future__ import annotations

import io
from typing import Mapping

import numpy as np
import torch
from torch import nn

from skill_generation.artifacts import CheckedBundle, read_artifact
from skill_generation.errors import BackendError, DependencyCompatibilityError, StateError


BACKEND_ID = "local-avoid-v13-torch"
MODEL_FORMAT = "local-avoid-v13-actor-npz-v1"
CONTRACT_VERSION = "local-avoid-v13-logits-v1"
PARAMETER_KEYS = (
    "actor.0.weight", "actor.0.bias", "actor.2.weight", "actor.2.bias",
    "actor.4.weight", "actor.4.bias", "actor.6.weight", "actor.6.bias",
    "actor_rms.mean", "actor_rms.var", "actor_rms.count",
)
EXPECTED_SHAPES = {
    "actor.0.weight": (256, 33), "actor.0.bias": (256,),
    "actor.2.weight": (256, 256), "actor.2.bias": (256,),
    "actor.4.weight": (128, 256), "actor.4.bias": (128,),
    "actor.6.weight": (3, 128), "actor.6.bias": (3,),
    "actor_rms.mean": (33,), "actor_rms.var": (33,),
    "actor_rms.count": (),
}


def validate_v13_contract(bundle: CheckedBundle) -> None:
    manifest = bundle.manifest
    expected = {
        "backend_id": BACKEND_ID,
        "backend_interface_version": "1",
        "model_format": MODEL_FORMAT,
        "contract_version": CONTRACT_VERSION,
    }
    for field, value in expected.items():
        if getattr(manifest, field) != value:
            raise BackendError("unsupported_contract", field, value, getattr(manifest, field))
    if dict(manifest.preprocessing) != {
        "required": True,
        "artifact": "actor_parameters",
        "version": "running-mean-std-v1",
    }:
        raise BackendError("unsupported_preprocessing", "preprocessing",
                           "running-mean-std-v1", dict(manifest.preprocessing))
    if {item.name for item in manifest.artifacts} != {"actor_parameters", "environment_config"}:
        raise BackendError("invalid_artifacts", "artifacts",
                           "actor_parameters and environment_config", "artifact mismatch")
    if len(manifest.inputs) != 1 or len(manifest.outputs) != 1:
        raise BackendError("invalid_io_contract", "inputs/outputs", "one input and output", "count mismatch")
    actor_input, logits = manifest.inputs[0], manifest.outputs[0]
    if (actor_input.name, actor_input.dtype, actor_input.shape) != ("observation", "float32", (33,)):
        raise BackendError("invalid_io_contract", "inputs.observation", "float32[33]", "mismatch")
    if (logits.name, logits.dtype, logits.shape) != ("logits", "float32", (3,)):
        raise BackendError("invalid_io_contract", "outputs.logits", "float32[3]", "mismatch")
    installed = {"numpy": np.__version__, "torch": torch.__version__}
    if set(manifest.dependencies) != set(installed):
        raise BackendError("invalid_dependencies", "dependencies", sorted(installed), sorted(manifest.dependencies))
    for package, actual in installed.items():
        required = manifest.dependencies[package]
        if required != actual:
            raise DependencyCompatibilityError(package, required, actual)


class LocalAvoidV13Backend:
    stateless = True
    interface_version = "1"
    version = "1.0"
    device = "cpu"
    inference_mode = "deterministic"

    @staticmethod
    def validate_bundle(bundle: CheckedBundle) -> None:
        validate_v13_contract(bundle)

    def __init__(self) -> None:
        self._network: nn.Sequential | None = None
        self._mean: np.ndarray | None = None
        self._var: np.ndarray | None = None

    def load(self, bundle: CheckedBundle) -> None:
        if self._network is not None:
            raise StateError("already_loaded", "backend", "closed backend", "loaded")
        validate_v13_contract(bundle)
        raw = read_artifact(bundle, "actor_parameters")
        try:
            with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
                if set(archive.files) != set(PARAMETER_KEYS):
                    raise BackendError("invalid_parameter_names", "actor_parameters",
                                       sorted(PARAMETER_KEYS), sorted(archive.files))
                arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
            for name, shape in EXPECTED_SHAPES.items():
                value = arrays[name]
                expected_dtype = np.float64 if name.startswith("actor_rms") else np.float32
                if value.shape != shape or value.dtype != expected_dtype or not np.isfinite(value).all():
                    raise BackendError("invalid_parameter", name,
                                       f"{expected_dtype.__name__}{shape} finite",
                                       f"{value.dtype}{value.shape}")
            if np.any(arrays["actor_rms.var"] < 0) or float(arrays["actor_rms.count"]) <= 0:
                raise BackendError("invalid_normalizer", "actor_rms", "nonnegative variance and positive count", "invalid")
            network = nn.Sequential(
                nn.Linear(33, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(),
                nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 3),
            )
            state = {
                name.removeprefix("actor."): torch.from_numpy(arrays[name])
                for name in PARAMETER_KEYS if name.startswith("actor.")
            }
            network.load_state_dict(state, strict=True)
            network.eval()
        except BackendError:
            raise
        except Exception as cause:
            raise BackendError("parameter_load_failed", "actor_parameters",
                               "valid data-only V13 Actor", type(cause).__name__) from cause
        self._network = network
        self._mean = arrays["actor_rms.mean"]
        self._var = arrays["actor_rms.var"]

    def infer(self, inputs: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        if self._network is None or self._mean is None or self._var is None:
            raise StateError("not_loaded", "backend", "loaded backend", "unloaded")
        observation = inputs["observation"]
        normalized = np.clip(
            (observation - self._mean) / np.sqrt(self._var + 1e-8), -10.0, 10.0,
        ).astype(np.float32)
        with torch.inference_mode():
            logits = self._network(torch.from_numpy(normalized)).numpy().astype(np.float32, copy=True)
        return {"logits": logits}

    def close(self) -> None:
        self._network = None
        self._mean = None
        self._var = None
