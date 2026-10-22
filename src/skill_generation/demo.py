"""Deterministic harmless bundle generation; no training or random fallback."""

import hashlib
import json
from pathlib import Path
import numpy as np


def create_demo_bundle(root: Path, *, version: str = "1.0.0", scale: float = 1.0) -> Path:
    """Create a new bundle only; never overwrite an existing version directory."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    (root / "artifacts").mkdir()
    path = root / "artifacts" / "parameters.npz"
    weight = np.array([[1., 2.], [3., 4.]], dtype=np.float64) * scale
    bias = np.array([0.5, -0.5], dtype=np.float64)
    np.savez(path, weight=weight, bias=bias)
    manifest = {
        "format_version": 1, "model_id": "harmless-affine", "version": version,
        "backend": {"id": "numpy-affine", "interface_version": "1"},
        "artifacts": [{"name": "parameters", "path": "artifacts/parameters.npz",
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}],
        "inputs": [{"name": "x", "dtype": "float64", "shape": [2], "finite": True}],
        "outputs": [{"name": "y", "dtype": "float64", "shape": [2], "finite": True}],
        "contract_version": "affine-v1", "preprocessing": {"required": False},
        "model_format": "numpy-affine-npz-v1", "dependencies": {"numpy": np.__version__},
        "source": "Locally generated deterministic arithmetic fixture; no task model.",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (root / "README.md").write_text("# Harmless Affine Fixture\n\nOnly y = x @ weight + bias. Not a trained task model.\n", encoding="utf-8")
    return root
