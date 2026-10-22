"""Validate tensor contracts without coercion, copies, or mutation."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .contracts import TensorSpec
from .errors import ValidationError


def validate_tensors(values: Mapping[str, np.ndarray], specs: Sequence[TensorSpec], phase: str = "input") -> None:
    specs = tuple(specs)
    if any(not isinstance(spec, TensorSpec) for spec in specs) or len({s.name for s in specs}) != len(specs):
        raise ValidationError("invalid_specs", phase, "unique TensorSpec sequence", "invalid specs")
    if not isinstance(values, Mapping):
        raise ValidationError("invalid_tensors", phase, "mapping", type(values).__name__)
    if set(values) != {s.name for s in specs}:
        names = tuple(key if isinstance(key, str) else f"<{type(key).__name__}>" for key in values)
        raise ValidationError("tensor_names_mismatch", phase, tuple(s.name for s in specs), names)
    for spec in specs:
        value = values[spec.name]
        field = f"{phase}.{spec.name}"
        # Subclasses may execute custom array operations; require plain arrays.
        if type(value) is not np.ndarray:
            raise ValidationError("invalid_tensor_type", field, "numpy.ndarray", type(value).__name__)
        if value.shape != spec.shape:
            raise ValidationError("shape_mismatch", field, spec.shape, value.shape)
        if value.dtype != np.dtype(spec.dtype):
            raise ValidationError("dtype_mismatch", field, spec.dtype, str(value.dtype))
        if spec.finite and not bool(np.isfinite(value).all()):
            raise ValidationError("nonfinite_tensor", field, "finite values", "nonfinite value present")
        # Python scalars preserve large integer precision and do not round a
        # manifest bound down to the array's float16/float32 dtype.
        if spec.minimum is not None and value.size and not value.min().item() >= spec.minimum:
            raise ValidationError("below_minimum", field, spec.minimum, "range violation")
        if spec.maximum is not None and value.size and not value.max().item() <= spec.maximum:
            raise ValidationError("above_maximum", field, spec.maximum, "range violation")
