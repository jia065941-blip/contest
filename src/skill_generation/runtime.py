"""Sequential runtime with exclusive backend ownership per session."""

from collections import deque
from enum import Enum
from pathlib import Path
from time import perf_counter
from uuid import uuid4
from typing import Mapping

import numpy as np

from .artifacts import load_bundle
from .contracts import InferenceResult
from .errors import BackendError, SkillGenerationError, StateError
from .monitor import Monitor
from .registry import BackendRegistry, create_default_registry
from .validator import validate_tensors


class SessionState(str, Enum):
    NEW = "NEW"
    READY = "READY"
    ERROR = "ERROR"
    CLOSED = "CLOSED"


class Runtime:
    def __init__(self, bundle_dir: str | Path, *, registry: BackendRegistry | None = None,
                 monitor: Monitor | None = None, max_error_history: int = 100):
        if type(max_error_history) is not int or max_error_history < 0:
            raise ValueError("max_error_history must be a nonnegative integer")
        self.bundle_dir = Path(bundle_dir)
        self.registry = registry if registry is not None else create_default_registry()
        self.monitor = monitor if monitor is not None else Monitor()
        self.session_id = uuid4().hex
        self.state = SessionState.NEW
        self.call_count = 0
        self.success_count = 0
        self.errors: deque[dict[str, str | None]] = deque(maxlen=max_error_history)
        self.total_error_count = 0
        self.bundle = None
        self.backend = None
        self.backend_version: str | None = None
        self._record("created")

    def _record(self, event: str, **metadata: object) -> None:
        manifest = self.bundle.manifest if self.bundle is not None else None
        try:
            self.monitor.record(event, session_id=self.session_id, state=self.state.value,
                                model_id=manifest.model_id if manifest else None,
                                model_version=manifest.version if manifest else None,
                                model_fingerprint=self.bundle.fingerprint if self.bundle else None,
                                backend_id=manifest.backend_id if manifest else None,
                                backend_version=self.backend_version,
                                backend_device=getattr(self.backend, "device", None),
                                inference_mode=getattr(self.backend, "inference_mode", None),
                                call_count=self.call_count, success_count=self.success_count,
                                total_error_count=self.total_error_count, **metadata)
        except Exception:
            # Monitoring is observational and must never alter lifecycle or numerics.
            return

    def _error(self, exc: SkillGenerationError, call_id: str | None = None) -> None:
        self.total_error_count += 1
        self.errors.append({"code": exc.code, "call_id": call_id})
        self._record("error", error_code=exc.code, call_id=call_id)

    def _require(self, state: SessionState) -> None:
        if self.state != state:
            exc = StateError("INVALID_SESSION_STATE", field="state",
                             expected=state.value, actual=self.state.value)
            self._error(exc)
            raise exc

    def open(self) -> "Runtime":
        self._require(SessionState.NEW)
        start = perf_counter()
        try:
            self.bundle = load_bundle(self.bundle_dir)
            self.registry.validate(self.bundle)
            self.backend = self.registry.create(self.bundle.manifest.backend_id)
            self.backend_version = self.backend.version
            if getattr(self.backend, "stateless", False) is not True:
                raise BackendError("STATEFUL_BACKEND_UNSUPPORTED", field="stateless",
                                   expected=True, actual=False)
            self.backend.load(self.bundle)
        except Exception as cause:
            self.state = SessionState.ERROR
            exc = cause if isinstance(cause, SkillGenerationError) else BackendError(
                "BACKEND_LOAD_FAILED", field="backend", expected="successful load",
                actual=type(cause).__name__)
            self._error(exc)
            self._record("load_failed", load_seconds=perf_counter() - start)
            backend, self.backend = self.backend, None
            if backend is not None:
                try:
                    backend.close()
                except Exception as cleanup_cause:
                    self._record("load_cleanup_failed",
                                 cleanup_error=type(cleanup_cause).__name__)
            if exc is cause:
                raise
            raise exc from cause
        self.state = SessionState.READY
        self._record("loaded", load_seconds=perf_counter() - start)
        return self

    def infer(self, inputs: Mapping[str, np.ndarray]) -> InferenceResult:
        self._require(SessionState.READY)
        self.call_count += 1
        call_id = f"{self.session_id}:{self.call_count}"
        assert self.bundle is not None and self.backend is not None
        try:
            validate_tensors(inputs, self.bundle.manifest.inputs, phase="input")
        except SkillGenerationError as exc:
            self._error(exc, call_id)
            raise
        start = perf_counter()
        try:
            # Backends receive owned copies, so even a faulty implementation cannot
            # mutate caller arrays. Returned arrays are copied once after validation.
            backend_inputs = {name: value.copy() for name, value in inputs.items()}
            outputs = self.backend.infer(backend_inputs)
            validate_tensors(outputs, self.bundle.manifest.outputs, phase="output")
        except Exception as cause:
            self.state = SessionState.ERROR
            exc = cause if isinstance(cause, SkillGenerationError) else BackendError(
                "BACKEND_INFERENCE_FAILED", field="backend", expected="valid outputs",
                actual=type(cause).__name__)
            self._error(exc, call_id)
            self._record("inference_failed", call_id=call_id,
                         inference_seconds=perf_counter() - start)
            if exc is cause:
                raise
            raise exc from cause
        elapsed = perf_counter() - start
        owned_outputs = {name: value.copy() for name, value in outputs.items()}
        self.success_count += 1
        self._record("inference_completed", call_id=call_id, inference_seconds=elapsed)
        return InferenceResult(outputs=owned_outputs, call_id=call_id,
                               session_id=self.session_id,
                               model_id=self.bundle.manifest.model_id,
                               model_version=self.bundle.manifest.version,
                               model_fingerprint=self.bundle.fingerprint,
                               backend_id=self.bundle.manifest.backend_id,
                               elapsed_seconds=elapsed)

    def close(self) -> None:
        if self.state == SessionState.CLOSED:
            return
        backend, self.backend = self.backend, None
        try:
            if backend is not None:
                backend.close()
        except Exception as cause:
            self.state = SessionState.CLOSED
            exc = BackendError("BACKEND_CLOSE_FAILED", field="backend",
                               expected="resource release", actual=type(cause).__name__)
            self._error(exc)
            raise exc from cause
        finally:
            self.state = SessionState.CLOSED
            self._record("closed")

    def __enter__(self) -> "Runtime":
        self._require(SessionState.NEW)
        return self.open()

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc is None:
            self.close()
        else:
            try:
                self.close()
            except SkillGenerationError:
                pass  # The original inference/load exception remains primary.
