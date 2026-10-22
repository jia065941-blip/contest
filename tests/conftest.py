"""Independent, data-only fixtures; all generated files live under tmp_path."""

import hashlib
import json

import numpy as np
import pytest


@pytest.fixture
def bundle_factory(tmp_path):
    def make(*, version="test-1", scale=1.0, dtype="float64", mutate=None,
             parameters=None):
        root = tmp_path / f"bundle-{len(list(tmp_path.glob('bundle-*')))}"
        root.mkdir()
        (root / "artifacts").mkdir()
        artifact = root / "artifacts" / "parameters.npz"
        if parameters is None:
            parameters = {
                "weight": np.array([[2, -1], [0.5, 3]], dtype=dtype) * scale,
                "bias": np.array([1, -2], dtype=dtype),
            }
        np.savez(artifact, **parameters)
        manifest = {
            "format_version": 1, "model_id": "independent-affine", "version": version,
            "backend": {"id": "numpy-affine", "interface_version": "1"},
            "artifacts": [{"name": "parameters", "path": "artifacts/parameters.npz",
                           "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}],
            "inputs": [{"name": "x", "dtype": dtype, "shape": [2], "finite": True}],
            "outputs": [{"name": "y", "dtype": dtype, "shape": [2], "finite": True}],
            "contract_version": "affine-v1", "preprocessing": {"required": False},
            "model_format": "numpy-affine-npz-v1", "dependencies": {"numpy": np.__version__},
            "source": "Independent deterministic test arithmetic, not a task model",
        }
        if mutate is not None:
            mutate(manifest)
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return root
    return make
