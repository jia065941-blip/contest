"""Immutable, domain-independent model and tensor contracts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, Protocol, runtime_checkable

from .errors import ManifestError

if TYPE_CHECKING:
    import numpy as np

    from .artifacts import CheckedBundle


DTYPES = frozenset({"bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64"})


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ManifestError("invalid_string", field, "nonempty string", type(value).__name__)
    return value


def _keys(value: Any, required: set[str], optional: set[str], field: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ManifestError("invalid_object", field, "object", type(value).__name__)
    if not required <= value.keys() or value.keys() - required - optional:
        raise ManifestError("invalid_fields", field, sorted(required), "missing or unknown fields")
    return value


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    finite: bool = True
    minimum: int | float | None = None
    maximum: int | float | None = None

    def __post_init__(self) -> None:
        _text(self.name, "tensor.name")
        if not isinstance(self.dtype, str) or self.dtype not in DTYPES:
            raise ManifestError("invalid_dtype", "tensor.dtype", sorted(DTYPES), "unsupported dtype")
        if not isinstance(self.shape, (tuple, list)) or any(type(d) is not int or d < 0 for d in self.shape):
            raise ManifestError("invalid_shape", "tensor.shape", "fixed nonnegative integer dimensions", "invalid shape")
        object.__setattr__(self, "shape", tuple(self.shape))
        if type(self.finite) is not bool:
            raise ManifestError("invalid_boolean", "tensor.finite", "boolean", type(self.finite).__name__)
        for name in ("minimum", "maximum"):
            value = getattr(self, name)
            if value is not None and (type(value) not in (int, float) or (type(value) is float and not math.isfinite(value))):
                raise ManifestError("invalid_bound", f"tensor.{name}", "finite number or null", "invalid bound")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ManifestError("invalid_range", "tensor", "minimum <= maximum", "reversed bounds")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TensorSpec:
        data = _keys(value, {"name", "dtype", "shape"}, {"finite", "minimum", "maximum"}, "tensor")
        return cls(**data)


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        _text(self.name, "artifact.name")
        _text(self.path, "artifact.path")
        if not isinstance(self.sha256, str) or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise ManifestError("invalid_sha256", "artifact.sha256", "64 lowercase hex digits", "invalid digest")


@dataclass(frozen=True)
class ModelManifest:
    format_version: int
    model_id: str
    version: str
    backend_id: str
    backend_interface_version: str
    artifacts: tuple[ArtifactSpec, ...]
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    contract_version: str
    preprocessing: Mapping[str, Any]
    model_format: str
    dependencies: Mapping[str, str]
    source: str

    def __post_init__(self) -> None:
        if type(self.format_version) is not int or self.format_version != 1:
            raise ManifestError("unsupported_format_version", "format_version", 1, self.format_version)
        for name in ("model_id", "version", "backend_id", "backend_interface_version", "contract_version", "model_format", "source"):
            _text(getattr(self, name), name)
        if self.backend_interface_version != "1":
            raise ManifestError("unsupported_interface_version", "backend.interface_version", "1", self.backend_interface_version)
        for name, kind in (("artifacts", ArtifactSpec), ("inputs", TensorSpec), ("outputs", TensorSpec)):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)) or not values or any(not isinstance(v, kind) for v in values):
                raise ManifestError("invalid_collection", name, "nonempty contract sequence", "invalid collection")
            if len({v.name for v in values}) != len(values):
                raise ManifestError("duplicate_name", name, "unique names", "duplicates")
            object.__setattr__(self, name, tuple(values))
        if len({a.path for a in self.artifacts}) != len(self.artifacts):
            raise ManifestError("duplicate_path", "artifacts", "unique paths", "duplicates")
        preprocessing = _keys(self.preprocessing, {"required"}, {"artifact", "version"}, "preprocessing")
        if type(preprocessing["required"]) is not bool:
            raise ManifestError("invalid_boolean", "preprocessing.required", "boolean", "invalid type")
        if preprocessing["required"]:
            for key in ("artifact", "version"):
                _text(preprocessing.get(key), f"preprocessing.{key}")
            if preprocessing["artifact"] not in {a.name for a in self.artifacts}:
                raise ManifestError("missing_preprocessing_artifact", "preprocessing.artifact", "declared artifact name", preprocessing["artifact"])
        elif set(preprocessing) != {"required"}:
            raise ManifestError("unexpected_preprocessing_metadata", "preprocessing", {"required": False}, "extra metadata")
        object.__setattr__(self, "preprocessing", MappingProxyType(dict(preprocessing)))
        if not isinstance(self.dependencies, Mapping):
            raise ManifestError("invalid_dependencies", "dependencies", "version mapping", "invalid type")
        for key, value in self.dependencies.items():
            _text(key, "dependencies.name")
            _text(value, "dependencies.version")
        object.__setattr__(self, "dependencies", MappingProxyType(dict(self.dependencies)))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ModelManifest:
        fields = {"format_version", "model_id", "version", "backend", "artifacts", "inputs", "outputs", "contract_version", "preprocessing", "model_format", "dependencies", "source"}
        data = dict(_keys(value, fields, set(), "manifest"))
        backend = _keys(data.pop("backend"), {"id", "interface_version"}, set(), "backend")
        for name in ("artifacts", "inputs", "outputs"):
            if not isinstance(data[name], list):
                raise ManifestError("invalid_collection", name, "array", type(data[name]).__name__)
        data["artifacts"] = tuple(ArtifactSpec(**_keys(v, {"name", "path", "sha256"}, set(), "artifact")) for v in data["artifacts"])
        for name in ("inputs", "outputs"):
            data[name] = tuple(TensorSpec.from_dict(v) for v in data[name])
        return cls(backend_id=backend["id"], backend_interface_version=backend["interface_version"], **data)


@dataclass(frozen=True)
class InferenceResult:
    outputs: Mapping[str, np.ndarray]
    call_id: str
    session_id: str
    model_id: str
    model_version: str
    model_fingerprint: str
    backend_id: str
    elapsed_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))


@runtime_checkable
class ModelBackend(Protocol):
    """Trusted stateless backend; instances must support weak references.

    Slot-based implementations must include ``__weakref__`` in their slots
    so the registry can detect accidental resource sharing without retaining it.
    """
    stateless: ClassVar[bool] = True
    interface_version: ClassVar[str] = "1"
    version: ClassVar[str] = "1.0"
    device: ClassVar[str]
    inference_mode: ClassVar[str]

    def load(self, bundle: CheckedBundle) -> None: ...
    def infer(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def close(self) -> None: ...
