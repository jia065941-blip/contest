"""A stateless, explicitly shaped affine transform with data-only parameters."""

from __future__ import annotations

import io
import zipfile
from typing import Mapping

import numpy as np

from ..artifacts import CheckedBundle, read_artifact
from ..contracts import TensorSpec
from ..errors import BackendError, DependencyCompatibilityError, SkillGenerationError, StateError
from ..validator import validate_tensors


def validate_affine_contract(bundle: CheckedBundle) -> None:
    m = bundle.manifest
    expected = {"backend_id": "numpy-affine", "backend_interface_version": "1", "model_format": "numpy-affine-npz-v1", "contract_version": "affine-v1"}
    for field, value in expected.items():
        if getattr(m, field) != value:
            raise BackendError("unsupported_contract", field, value, getattr(m, field))
    if dict(m.preprocessing) != {"required": False}:
        raise BackendError("preprocessing_forbidden", "preprocessing", {"required": False}, "preprocessing requested")
    if set(m.dependencies) != {"numpy"} or m.dependencies["numpy"] != np.__version__:
        required = m.dependencies.get("numpy", "exactly one declared numpy version")
        raise DependencyCompatibilityError("numpy", required, np.__version__)
    if len(m.artifacts) != 1 or m.artifacts[0].name != "parameters" or set(bundle.artifacts) != {"parameters"}:
        raise BackendError("invalid_artifacts", "artifacts", "parameters only", "artifact mismatch")
    if len(m.inputs) != 1 or len(m.outputs) != 1 or m.inputs[0].name != "x" or m.outputs[0].name != "y":
        raise BackendError("invalid_io_contract", "inputs/outputs", "x -> y", "tensor names mismatch")
    x, y = m.inputs[0], m.outputs[0]
    if len(x.shape) != 1 or len(y.shape) != 1 or x.shape[0] <= 0 or y.shape[0] <= 0:
        raise BackendError("invalid_affine_shape", "inputs/outputs", "positive fixed vectors (n,) -> (m,)", "invalid shape")
    if x.dtype not in ("float32", "float64") or x.dtype != y.dtype:
        raise BackendError("invalid_affine_dtype", "inputs/outputs", "matching float32 or float64", "dtype mismatch")


class NumpyAffineBackend:
    stateless = True
    interface_version = "1"
    version = "1.0"
    device = "cpu"
    inference_mode = "deterministic"

    @staticmethod
    def validate_bundle(bundle: CheckedBundle) -> None:
        validate_affine_contract(bundle)

    def __init__(self) -> None:
        self._bundle: CheckedBundle | None = None
        self._weight: np.ndarray | None = None
        self._bias: np.ndarray | None = None

    def load(self, bundle: CheckedBundle) -> None:
        if self._bundle is not None:
            raise StateError("already_loaded", "backend", "close before reloading", "loaded")
        validate_affine_contract(bundle)
        x, y = bundle.manifest.inputs[0], bundle.manifest.outputs[0]
        raw = read_artifact(bundle, "parameters")
        try:
            # Reject duplicate members, directories, and uncontracted hidden arrays.
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                members = archive.infolist()
                if len(members) != 2 or {i.filename for i in members} != {"weight.npy", "bias.npy"}:
                    raise BackendError("invalid_parameter_members", "parameters", "weight.npy and bias.npy only", "invalid members")
                sizes = {"weight.npy": x.shape[0] * y.shape[0], "bias.npy": y.shape[0]}
                for member in members:
                    if member.file_size > sizes[member.filename] * np.dtype(x.dtype).itemsize + 16384:
                        raise BackendError("parameter_size_exceeded", "parameters", "contract-sized arrays", "oversized member")
                    # Validate headers before NumPy can allocate from untrusted shapes.
                    with archive.open(member) as stream:
                        version = np.lib.format.read_magic(stream)
                        if version == (1, 0):
                            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                        elif version == (2, 0):
                            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                        else:
                            raise BackendError("unsupported_npy_version", "parameters", "NPY 1.0 or 2.0", version)
                        expected_shape = (x.shape[0], y.shape[0]) if member.filename == "weight.npy" else y.shape
                        if shape != expected_shape or dtype != np.dtype(x.dtype):
                            raise BackendError("parameter_header_mismatch", "parameters", "contract shape and dtype", "header mismatch")
                        if member.file_size != stream.tell() + sizes[member.filename] * dtype.itemsize:
                            raise BackendError("parameter_length_mismatch", "parameters", "exact array bytes", "truncated or trailing data")
            with np.load(io.BytesIO(raw), allow_pickle=False) as parameters:
                weight, bias = parameters["weight"], parameters["bias"]
            validate_tensors({"weight": weight, "bias": bias}, (
                TensorSpec("weight", x.dtype, (x.shape[0], y.shape[0])),
                TensorSpec("bias", x.dtype, y.shape),
            ), phase="parameters")
            weight.setflags(write=False)
            bias.setflags(write=False)
        except SkillGenerationError:
            raise
        except Exception as exc:
            raise BackendError("parameter_load_failed", "parameters", "valid numeric NPZ", type(exc).__name__) from exc
        self._weight, self._bias, self._bundle = weight, bias, bundle

    def infer(self, inputs: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        if self._bundle is None:
            raise StateError("not_loaded", "backend", "loaded backend", "unloaded")
        validate_tensors(inputs, self._bundle.manifest.inputs)
        try:
            with np.errstate(over="raise", invalid="raise"):
                y = inputs["x"] @ self._weight + self._bias
        except Exception as exc:
            raise BackendError("inference_failed", "output.y", "valid affine result", type(exc).__name__) from exc
        result = {"y": y}
        validate_tensors(result, self._bundle.manifest.outputs, phase="output")
        return result

    def close(self) -> None:
        self._bundle = None
        self._weight = None
        self._bias = None
