import json
from datetime import datetime
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import skill_generation

from skill_generation.artifacts import load_bundle
from skill_generation.backends.numpy_affine import NumpyAffineBackend
from skill_generation.errors import (BackendError, DependencyCompatibilityError,
                                     SkillGenerationError, StateError, ValidationError)
from skill_generation.monitor import Monitor
from skill_generation.registry import BackendRegistry
from skill_generation.runtime import SessionState
from skill_generation.skills import InferenceSkill


def injected_registry(backend):
    registry = BackendRegistry()
    registry.register("numpy-affine", lambda: backend,
                      validator=NumpyAffineBackend.validate_bundle)
    return registry


class FaultBackend(NumpyAffineBackend):
    def __init__(self, *, fail=None, output=None):
        super().__init__()
        self.fail = fail
        self.output = output
        self.close_count = 0
        self.infer_count = 0

    def load(self, bundle):
        if self.fail == "load":
            raise RuntimeError("controlled load failure")
        super().load(bundle)

    def infer(self, inputs):
        self.infer_count += 1
        if self.fail == "infer":
            raise RuntimeError("controlled inference failure")
        if self.output is not None:
            return self.output
        return super().infer(inputs)

    def close(self):
        self.close_count += 1
        super().close()
        if self.fail == "close":
            raise RuntimeError("controlled close failure")


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("values,expected", [([2, 4], [7, 8]), ([0, 0], [1, -2]),
                                             ([-2, 1], [-2.5, 3])])
def test_real_inference_and_direct_backend(bundle_factory, dtype, values, expected):
    root = bundle_factory(dtype=dtype)
    inputs = {"x": np.array(values, dtype=dtype)}
    original = inputs["x"].copy()
    backend = NumpyAffineBackend()
    backend.load(load_bundle(root))
    try:
        direct = backend.infer(inputs)
        with InferenceSkill(root) as skill:
            result = skill.infer(inputs)
            assert skill.state == SessionState.READY
            assert skill.call_count == skill.success_count == 1
            assert not skill.errors
            np.testing.assert_array_equal(result.outputs["y"], np.array(expected, dtype=dtype))
            np.testing.assert_array_equal(result.outputs["y"], direct["y"])
            assert result.outputs["y"].dtype == np.dtype(dtype)
            assert result.model_version == "test-1"
            assert result.call_id
            assert math.isfinite(result.elapsed_seconds) and result.elapsed_seconds >= 0
        assert skill.state == SessionState.CLOSED
        np.testing.assert_array_equal(inputs["x"], original)
    finally:
        backend.close()


@pytest.mark.parametrize("inputs", [
    {}, {"wrong": np.array([0., 0.])}, {"x": np.array([0., 0.]), "extra": np.array([0.])},
    {"x": np.array([0.])}, {"x": np.zeros((1, 2))}, {"x": np.zeros(2, dtype=np.float32)},
    {"x": np.array([0, 0])}, {"x": [0., 0.]}, {"x": np.array([np.nan, 0.])},
    {"x": np.array([np.inf, 0.])}, {"x": np.array([-np.inf, 0.])},
    {"x": np.array([-1.01, 0.])}, {"x": np.array([0., 1.01])},
])
def test_rejected_inputs_count_without_backend_call(bundle_factory, inputs):
    root = bundle_factory(mutate=lambda m: m["inputs"][0].update(minimum=-1, maximum=1))
    backend = FaultBackend()
    with InferenceSkill(root, registry=injected_registry(backend)) as skill:
        with pytest.raises(ValidationError):
            skill.infer(inputs)
        assert skill.state == SessionState.READY
        assert skill.call_count == 1 and skill.success_count == 0
        assert len(skill.errors) == 1 and skill.errors[0]["call_id"]
        assert backend.infer_count == 0
        result = skill.infer({"x": np.array([-1., 1.])})
        np.testing.assert_array_equal(result.outputs["y"], [-0.5, 2.])
        assert skill.call_count == 2 and skill.success_count == 1
        assert result.call_id != skill.errors[0]["call_id"]


@pytest.mark.parametrize("output", [
    {}, {"wrong": np.array([0., 0.])}, {"y": np.array([0.])},
    {"y": np.zeros(2, dtype=np.float32)}, {"y": np.array([np.nan, 0.])},
    {"y": np.array([0., np.inf])}, {"y": [0., 0.]}, {"y": np.array([11., 0.])},
])
def test_invalid_backend_outputs_are_not_successes(bundle_factory, output):
    root = bundle_factory(mutate=lambda m: m["outputs"][0].update(minimum=-10, maximum=10))
    with InferenceSkill(root, registry=injected_registry(FaultBackend(output=output))) as skill:
        with pytest.raises(ValidationError):
            skill.infer({"x": np.array([0., 0.])})
        assert skill.state == SessionState.ERROR
        assert skill.call_count == 1 and skill.success_count == 0
        assert len(skill.errors) == 1
        with pytest.raises(StateError):
            skill.infer({"x": np.array([0., 0.])})


@pytest.mark.parametrize("failure", ["load", "infer", "close"])
def test_backend_exceptions_and_cleanup(bundle_factory, failure):
    backend = FaultBackend(fail=failure)
    skill = InferenceSkill(bundle_factory(), registry=injected_registry(backend))
    try:
        with pytest.raises(BackendError) as caught:
            if failure == "load":
                skill.open()
            else:
                skill.open()
                if failure == "infer":
                    skill.infer({"x": np.array([0., 0.])})
                else:
                    skill.close()
        assert isinstance(caught.value.__cause__, RuntimeError)
        assert skill.state == (SessionState.CLOSED if failure == "close" else SessionState.ERROR)
        assert skill.success_count == 0 and len(skill.errors) == 1
    finally:
        skill.close()
        skill.close()
    assert backend.close_count == 1
    assert skill.state == SessionState.CLOSED


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_partially_loaded_backend_is_rolled_back_without_hiding_load_error(
        bundle_factory, cleanup_fails):
    class PartialBackend(FaultBackend):
        def load(self, bundle):
            super().load(bundle)
            self.allocated = True
            raise RuntimeError("failure after allocation")

        def close(self):
            super().close()
            self.allocated = False
            if cleanup_fails:
                raise RuntimeError("cleanup also failed")

    backend = PartialBackend()
    skill = InferenceSkill(bundle_factory(), registry=injected_registry(backend))
    with pytest.raises(BackendError) as caught:
        skill.open()
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == "failure after allocation"
    assert backend.close_count == 1 and backend.allocated is False
    assert skill.backend is None and skill.state == SessionState.ERROR
    skill.close()
    assert skill.state == SessionState.CLOSED and backend.close_count == 1


def test_session_lifecycle_order(bundle_factory):
    skill = InferenceSkill(bundle_factory())
    assert skill.state == SessionState.NEW
    with pytest.raises(StateError):
        skill.infer({"x": np.zeros(2)})
    assert skill.open() is skill
    with pytest.raises(StateError):
        skill.open()
    skill.close()
    skill.close()
    for operation in (skill.open, lambda: skill.infer({"x": np.zeros(2)})):
        with pytest.raises(StateError):
            operation()
    assert skill.state == SessionState.CLOSED


def test_direct_backend_order_and_repeated_close(bundle_factory):
    backend = NumpyAffineBackend()
    with pytest.raises(StateError):
        backend.infer({"x": np.zeros(2)})
    checked = load_bundle(bundle_factory())
    backend.load(checked)
    with pytest.raises(StateError):
        backend.load(checked)
    backend.close()
    backend.close()
    with pytest.raises(StateError):
        backend.infer({"x": np.zeros(2)})


def test_context_cleans_up_and_preserves_primary_error(bundle_factory):
    backend = FaultBackend(fail="close")
    skill = InferenceSkill(bundle_factory(), registry=injected_registry(backend))
    with pytest.raises(ValueError, match="primary"):
        with skill:
            raise ValueError("primary")
    assert skill.state == SessionState.CLOSED and backend.close_count == 1


def test_failed_context_load_releases_backend(bundle_factory):
    backend = FaultBackend(fail="load")
    skill = InferenceSkill(bundle_factory(), registry=injected_registry(backend))
    with pytest.raises(BackendError):
        with skill:
            pytest.fail("Failed load must not enter context")
    assert skill.state == SessionState.ERROR and backend.close_count == 1
    skill.close()
    assert skill.state == SessionState.CLOSED and backend.close_count == 1


def test_nested_context_rejected_without_closing_active_session(bundle_factory):
    with InferenceSkill(bundle_factory()) as skill:
        with pytest.raises(StateError):
            with skill:
                pytest.fail("Repeated entry must be rejected")
        assert skill.state == SessionState.READY
        np.testing.assert_array_equal(skill.infer({"x": np.array([2., 4.])}).outputs["y"], [7., 8.])


@pytest.mark.parametrize("parameters", [
    {"weight": np.eye(2)},
    {"weight": np.eye(2), "bias": np.zeros(2), "extra": np.zeros(1)},
    {"weight": np.ones((3, 2)), "bias": np.zeros(2)},
    {"weight": np.eye(2), "bias": np.zeros(3)},
    {"weight": np.eye(2, dtype=np.float32), "bias": np.zeros(2)},
    {"weight": np.array([[np.nan, 0.], [0., 1.]]), "bias": np.zeros(2)},
    {"weight": np.eye(2), "bias": np.array([np.inf, 0.])},
    {"weight": np.array([[1, 0], [0, 1]], dtype=object), "bias": np.zeros(2)},
])
def test_real_malformed_parameters_fail_without_output(bundle_factory, parameters):
    skill = InferenceSkill(bundle_factory(parameters=parameters))
    with pytest.raises(SkillGenerationError):
        skill.open()
    assert skill.state == SessionState.ERROR
    assert skill.success_count == skill.call_count == 0
    assert len(skill.errors) == 1
    skill.close()


def test_explicit_custom_backend_registration(bundle_factory):
    created = []
    validated = []

    class OffsetBackend:
        stateless = True
        interface_version = "1"
        version = "custom-1"
        device = "cpu"
        inference_mode = "deterministic"

        @staticmethod
        def validate_bundle(bundle):
            assert bundle.manifest.backend_id == "test-offset"
            validated.append(bundle.manifest.version)

        def __init__(self):
            created.append(self)
            self.loaded = False

        def load(self, bundle):
            self.loaded = True

        def infer(self, inputs):
            assert self.loaded
            return {"y": inputs["x"] + np.array([3., -4.])}

        def close(self):
            self.loaded = False

    root = bundle_factory(mutate=lambda m: m["backend"].update(id="test-offset"))
    registry = BackendRegistry()
    registry.register("test-offset", OffsetBackend)
    registry.validate(load_bundle(root))
    assert validated == ["test-1"] and not created
    with InferenceSkill(root, registry=registry) as skill:
        result = skill.infer({"x": np.array([2., 4.])})
        np.testing.assert_array_equal(result.outputs["y"], [5., 0.])
        assert len(created) == 1
    assert not created[0].loaded


def test_sessions_isolated_and_versions_stable(bundle_factory):
    old = bundle_factory(version="v1")
    new = bundle_factory(version="v2", scale=2)
    with InferenceSkill(old) as first, InferenceSkill(old) as second, InferenceSkill(new) as third:
        assert first.backend is not second.backend
        inputs = {"x": np.array([2., 4.])}
        a, b, c = (session.infer(inputs) for session in (first, second, third))
        assert len({a.call_id, b.call_id, c.call_id}) == 3
        np.testing.assert_array_equal(a.outputs["y"], [7., 8.])
        np.testing.assert_array_equal(b.outputs["y"], [7., 8.])
        np.testing.assert_array_equal(c.outputs["y"], [13., 18.])
        assert (a.model_version, b.model_version, c.model_version) == ("v1", "v1", "v2")
        a.outputs["y"][0] = 12345
        with pytest.raises(ValidationError):
            first.infer({"x": np.zeros(1)})
        assert second.call_count == 1 and not second.errors
        first.close()
        again = second.infer(inputs)
        np.testing.assert_array_equal(again.outputs["y"], [7., 8.])
        assert again.model_version == "v1" and again.call_id != b.call_id
        assert third.state == SessionState.READY


def test_one_session_failure_does_not_affect_another(bundle_factory):
    root = bundle_factory()
    failed = InferenceSkill(root, registry=injected_registry(FaultBackend(fail="infer")))
    healthy = InferenceSkill(root)
    try:
        failed.open()
        healthy.open()
        with pytest.raises(BackendError):
            failed.infer({"x": np.array([2., 4.])})
        assert failed.state == SessionState.ERROR
        np.testing.assert_array_equal(
            healthy.infer({"x": np.array([2., 4.])}).outputs["y"], [7., 8.])
        assert healthy.state == SessionState.READY and healthy.success_count == 1
    finally:
        failed.close()
        healthy.close()


def test_runtime_owns_inputs_and_returned_outputs(bundle_factory):
    class ReusingBackend(NumpyAffineBackend):
        def __init__(self):
            super().__init__()
            self.buffer = np.zeros(2)

        def infer(self, inputs):
            inputs["x"][:] = -999
            self.buffer[:] = super().infer({"x": np.array([2., 4.])})["y"]
            return {"y": self.buffer}

    source = np.array([2., 4.])
    backend = ReusingBackend()
    with InferenceSkill(bundle_factory(), registry=injected_registry(backend)) as skill:
        first = skill.infer({"x": source})
        saved = first.outputs["y"].copy()
        backend.buffer[:] = 12345
        second = skill.infer({"x": source})
        np.testing.assert_array_equal(source, [2., 4.])
        np.testing.assert_array_equal(first.outputs["y"], saved)
        assert not np.shares_memory(first.outputs["y"], second.outputs["y"])


def test_error_history_is_bounded_but_total_is_preserved(bundle_factory):
    with InferenceSkill(bundle_factory(), max_error_history=3) as skill:
        for _ in range(8):
            with pytest.raises(ValidationError):
                skill.infer({"x": np.zeros(1)})
        assert len(skill.errors) == 3
        assert skill.total_error_count == 8


def test_throwing_monitor_is_observational_only(bundle_factory):
    class ThrowingMonitor:
        def record(self, *args, **kwargs):
            raise RuntimeError("monitor must not escape")

    skill = InferenceSkill(bundle_factory(), monitor=ThrowingMonitor())
    skill.open()
    result = skill.infer({"x": np.array([2., 4.])})
    np.testing.assert_array_equal(result.outputs["y"], [7., 8.])
    with pytest.raises(ValidationError):
        skill.infer({"x": np.zeros(1)})
    skill.close()
    assert skill.state == SessionState.CLOSED


def test_result_contains_stable_model_and_session_identity(bundle_factory):
    root = bundle_factory()
    with InferenceSkill(root) as skill:
        first = skill.infer({"x": np.array([2., 4.])})
        second = skill.infer({"x": np.array([2., 4.])})
    assert first.model_id == "independent-affine"
    assert first.model_version == "test-1"
    assert first.backend_id == "numpy-affine"
    assert first.session_id == second.session_id == skill.session_id
    assert first.model_fingerprint == second.model_fingerprint
    assert len(first.model_fingerprint) == 64


@pytest.mark.parametrize("mode", ["enabled", "disabled", "write-failure"])
def test_monitor_does_not_change_results_or_hide_errors(bundle_factory, tmp_path, mode, capsys):
    log = tmp_path if mode == "write-failure" else tmp_path / "events.jsonl"
    monitor = Monitor(enabled=mode != "disabled", jsonl=log)
    with InferenceSkill(bundle_factory(), monitor=monitor) as skill:
        result = skill.infer({"x": np.array([2., 4.])})
        np.testing.assert_array_equal(result.outputs["y"], [7., 8.])
        with pytest.raises(ValidationError):
            skill.infer({"x": np.array([np.nan, 0.])})
        assert skill.call_count == 2 and skill.success_count == 1 and len(skill.errors) == 1
    if mode == "disabled":
        assert not log.exists()
    elif mode == "write-failure":
        assert "MONITOR_WRITE_FAILED" in capsys.readouterr().err
    else:
        records = [json.loads(line) for line in log.read_text().splitlines()]
        events = [record["event"] for record in records]
        assert "loaded" in events and "error" in events and events[-1] == "closed"
        assert events.count("inference_completed") == 1
        assert all("inputs" not in record and "outputs" not in record for record in records)
        assert all(record["session_id"] == skill.session_id for record in records)
        assert all(datetime.fromisoformat(record["timestamp"]).utcoffset() is not None for record in records)
        assert all(record["run_id"] == monitor.run_id for record in records)
        completed = next(record for record in records if record["event"] == "inference_completed")
        assert completed["call_id"] == result.call_id
        assert completed["model_fingerprint"] == result.model_fingerprint
        assert completed["backend_version"] == "1.0" and completed["application_version"] == "0.1.0"
        assert completed["backend_device"] == "cpu"
        assert completed["inference_mode"] == "deterministic"
        assert completed["inference_seconds"] >= 0


def test_monitor_failure_does_not_fabricate_backend_success(bundle_factory, tmp_path):
    backend = FaultBackend(fail="infer")
    with InferenceSkill(bundle_factory(), registry=injected_registry(backend),
                        monitor=Monitor(jsonl=tmp_path)) as skill:
        with pytest.raises(BackendError):
            skill.infer({"x": np.array([2., 4.])})
        assert skill.success_count == 0 and skill.state == SessionState.ERROR


def test_installed_package_isolated_import_and_run(bundle_factory, tmp_path):
    root = bundle_factory()
    cwd = tmp_path / "external-working-directory"
    cwd.mkdir()
    project_root = Path(__file__).resolve().parents[2]
    assert not cwd.resolve().is_relative_to(project_root)
    code = """
import json, sys
import numpy as np
import skill_generation
from pathlib import Path
assert 'site-packages' in skill_generation.__file__
assert Path.cwd() == Path(sys.argv[2])
assert not any(name == 'torch' for name in sys.modules)
from skill_generation.skills import InferenceSkill
with InferenceSkill(sys.argv[1]) as skill:
    result = skill.infer({'x': np.array([2., 4.])})
    print(json.dumps({'output': result.outputs['y'].tolist(),
                      'version': result.model_version,
                      'isolated': sys.flags.isolated}))
"""
    process = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(root), str(cwd)],
                             cwd=cwd, capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout) == {"output": [7., 8.], "version": "test-1", "isolated": 1}
    demo = subprocess.run([sys.executable, "-I", "-B", "-m", "skill_generation.cli",
                           "demo", "--work-dir", str(tmp_path)],
                          cwd=cwd, capture_output=True, text=True, timeout=30)
    assert demo.returncode == 0, demo.stderr
    data = json.loads(demo.stdout)
    assert data["outputs"] == {"y": [11.5, 15.5]}
    assert data["model_version"] == "1.0.0"
    assert data["call_id"] and data["elapsed_seconds"] >= 0


@pytest.mark.parametrize("command", ["inspect-bundle", "validate-bundle"])
def test_cli_static_commands_and_failure_exit(bundle_factory, tmp_path, command):
    root = bundle_factory()
    process = subprocess.run([sys.executable, "-I", "-B", "-m", "skill_generation.cli", command, str(root)],
                             cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    assert "Static checks only" in json.loads(process.stdout)["notice"]
    (root / "artifacts/parameters.npz").unlink()
    failure = subprocess.run([sys.executable, "-I", "-B", "-m", "skill_generation.cli", command, str(root)],
                             cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert failure.returncode == 2
    assert not failure.stdout
    assert "field=" in failure.stderr and "expected=" in failure.stderr and "actual=" in failure.stderr


def test_base_import_does_not_load_optional_or_external_backends(tmp_path):
    code = """
import sys, skill_generation
assert 'numpy' not in sys.modules
assert 'torch' not in sys.modules
assert 'skill_generation.backends.numpy_affine' not in sys.modules
assert not any(n.startswith(('jmy_ppo', 'competition_envs')) for n in sys.modules)
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
