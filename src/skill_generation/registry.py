"""Explicit trusted backend registration; static validation never calls factories."""

from __future__ import annotations

from builtins import id as builtins_id
from threading import RLock
from typing import Callable
from weakref import WeakValueDictionary, ref

from .artifacts import CheckedBundle
from .contracts import ModelBackend
from .errors import BackendError, SkillGenerationError


_issued: WeakValueDictionary[int, ModelBackend] = WeakValueDictionary()
_issued_lock = RLock()


class BackendRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], ModelBackend]] = {}
        self._validators: dict[str, Callable[[CheckedBundle], None]] = {}

    def register(self, id: str, factory: Callable[[], ModelBackend], *,
                 validator: Callable[[CheckedBundle], None] | None = None) -> None:
        if not isinstance(id, str) or not id.strip() or not callable(factory):
            raise BackendError("invalid_registration", "backend", "id and callable factory", "invalid registration")
        if id in self._factories:
            raise BackendError("duplicate_backend", "backend.id", "unregistered id", id)
        self._factories[id] = factory
        check = validator if validator is not None else getattr(factory, "validate_bundle", None)
        if callable(check):
            self._validators[id] = check

    def create(self, id: str) -> ModelBackend:
        if id not in self._factories:
            raise BackendError("unknown_backend", "backend.id", "registered backend", id)
        try:
            backend = self._factories[id]()
            if (getattr(backend, "stateless", None) is not True or getattr(backend, "interface_version", None) != "1"
                    or not isinstance(getattr(backend, "version", None), str)
                    or not isinstance(getattr(backend, "device", None), str)
                    or not isinstance(getattr(backend, "inference_mode", None), str)
                    or any(not callable(getattr(backend, name, None)) for name in ("load", "infer", "close"))):
                raise BackendError("invalid_backend", "backend",
                                   "stateless interface v1 with device and inference_mode",
                                   "incompatible backend")
            try:
                ref(backend)
            except TypeError as exc:
                raise BackendError("backend_not_weak_referenceable", "backend.ownership",
                                   "weak-referenceable instance (include __weakref__ in slots)",
                                   type(backend).__name__) from exc
            with _issued_lock:
                if builtins_id(backend) in _issued:
                    raise BackendError("backend_reused", "backend", "fresh exclusive instance", "already issued instance")
                _issued[builtins_id(backend)] = backend
            return backend
        except SkillGenerationError:
            raise
        except Exception as exc:
            raise BackendError("backend_creation_failed", "backend", "constructible backend", type(exc).__name__) from exc

    def validate(self, bundle: CheckedBundle) -> None:
        id = bundle.manifest.backend_id
        if id not in self._factories:
            raise BackendError("unknown_backend", "backend.id", "registered backend", id)
        if id not in self._validators:
            raise BackendError("unsupported_static_contract", "backend.id", "known static contract", id)
        try:
            self._validators[id](bundle)
        except SkillGenerationError:
            raise
        except Exception as exc:
            raise BackendError("contract_validation_failed", "backend", "valid static contract", type(exc).__name__) from exc


def create_default_registry() -> BackendRegistry:
    from .backends.numpy_affine import NumpyAffineBackend

    registry = BackendRegistry()
    registry.register("numpy-affine", NumpyAffineBackend)
    return registry
