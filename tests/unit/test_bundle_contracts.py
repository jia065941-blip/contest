import numpy as np
import pytest

from skill_generation.artifacts import CheckedBundle, load_bundle
from skill_generation.backends.numpy_affine import NumpyAffineBackend
from skill_generation.contracts import TensorSpec
from skill_generation.errors import (ArtifactError, BackendError,
                                     DependencyCompatibilityError, ManifestError,
                                     ValidationError)
from skill_generation.registry import BackendRegistry, create_default_registry
from skill_generation.validator import validate_tensors


@pytest.mark.parametrize("manifest_file", [False, True])
def test_load_and_static_validate_do_not_construct_backend(bundle_factory, manifest_file, monkeypatch):
    def forbidden_load(*args, **kwargs):
        pytest.fail("Static inspection must not load parameter arrays")
    monkeypatch.setattr(np, "load", forbidden_load)
    root = bundle_factory()
    bundle = load_bundle(root / "manifest.json" if manifest_file else root)
    assert isinstance(bundle, CheckedBundle)
    assert bundle.manifest.version == "test-1"
    assert bundle.artifacts["parameters"] == root / "artifacts/parameters.npz"
    registry = BackendRegistry()
    def forbidden():
        pytest.fail("Static validation must not instantiate a backend")
    registry.register("numpy-affine", forbidden, validator=NumpyAffineBackend.validate_bundle)
    registry.validate(bundle)


@pytest.mark.parametrize("field", [
    "format_version", "model_id", "version", "backend", "artifacts", "inputs",
    "outputs", "contract_version", "preprocessing", "model_format", "dependencies", "source",
])
def test_missing_manifest_field(bundle_factory, field):
    with pytest.raises(ManifestError):
        load_bundle(bundle_factory(mutate=lambda m: m.pop(field)))


@pytest.mark.parametrize("target", ["manifest.json", "artifacts/parameters.npz"])
def test_missing_file(bundle_factory, target):
    root = bundle_factory()
    (root / target).unlink()
    with pytest.raises(ArtifactError):
        load_bundle(root)


def test_digest_mismatch_and_recheck_before_load(bundle_factory):
    root = bundle_factory()
    checked = load_bundle(root)
    artifact = root / "artifacts/parameters.npz"
    artifact.write_bytes(artifact.read_bytes() + b"tampered")
    with pytest.raises(ArtifactError):
        load_bundle(root)
    with pytest.raises(ArtifactError):
        NumpyAffineBackend().load(checked)


@pytest.mark.parametrize("path", ["../outside.npz", "/outside.npz", "artifacts/../../outside.npz",
                                  "artifacts/./parameters.npz", "artifacts\\parameters.npz"])
def test_unsafe_artifact_paths(bundle_factory, path):
    root = bundle_factory(mutate=lambda m: m["artifacts"][0].update(path=path))
    with pytest.raises(ArtifactError):
        load_bundle(root)


@pytest.mark.parametrize("target", ["artifact", "directory", "manifest"])
def test_external_symlinks(bundle_factory, tmp_path, target):
    root = bundle_factory()
    if target == "directory":
        source = root / "artifacts"
        destination = tmp_path / "external-artifacts"
    else:
        source = root / ("manifest.json" if target == "manifest" else "artifacts/parameters.npz")
        destination = tmp_path / "external-file"
    source.rename(destination)
    source.symlink_to(destination, target_is_directory=target == "directory")
    with pytest.raises(ArtifactError):
        load_bundle(root)


@pytest.mark.parametrize("mutation", [
    lambda m: m["backend"].update(id="unregistered"),
    lambda m: m["preprocessing"].update(required=True, artifact="parameters", version="unknown"),
    lambda m: m.update(model_format="unknown"),
    lambda m: m.update(contract_version="unknown"),
    lambda m: m.update(dependencies={"numpy": "0.0.0"}),
])
def test_unsupported_static_contract(bundle_factory, mutation):
    bundle = load_bundle(bundle_factory(mutate=mutation))
    with pytest.raises(BackendError):
        create_default_registry().validate(bundle)


def test_dependency_mismatch_has_actionable_diagnostic(bundle_factory):
    bundle = load_bundle(bundle_factory(
        mutate=lambda manifest: manifest.update(dependencies={"numpy": "0.0.0"})))
    with pytest.raises(DependencyCompatibilityError) as caught:
        create_default_registry().validate(bundle)
    message = str(caught.value)
    assert "Package: numpy" in message
    assert "Required: 0.0.0" in message
    assert f"Installed: {np.__version__}" in message
    assert "Model bundle validation failed." in message


@pytest.mark.parametrize("shape", [[None], [-1], ["batch"], [True], [2.0]])
def test_dynamic_or_invalid_dimensions(bundle_factory, shape):
    with pytest.raises(ManifestError):
        load_bundle(bundle_factory(mutate=lambda m: m["inputs"][0].update(shape=shape)))


@pytest.mark.parametrize("kwargs", [
    {"dtype": "object"}, {"finite": "yes"}, {"minimum": float("nan")},
    {"maximum": float("inf")}, {"minimum": 2, "maximum": 1},
])
def test_tensor_spec_invalid_options(kwargs):
    options = dict(name="x", dtype="float64", shape=(2,))
    options.update(kwargs)
    with pytest.raises(ManifestError):
        TensorSpec(**options)


def test_tensor_range_endpoints_and_finite_policy():
    spec = TensorSpec("x", "float64", (2,), minimum=-1, maximum=1)
    validate_tensors({"x": np.array([-1., 1.])}, (spec,))
    for value in (np.nextafter(-1., -np.inf), np.nextafter(1., np.inf)):
        with pytest.raises(ValidationError):
            validate_tensors({"x": np.array([0., value])}, (spec,))
    validate_tensors({"x": np.array([np.nan, np.inf])},
                     (TensorSpec("x", "float64", (2,), finite=False),))


def test_registry_duplicate_unknown_and_fresh_instances():
    registry = create_default_registry()
    with pytest.raises(BackendError):
        registry.register("numpy-affine", NumpyAffineBackend)
    with pytest.raises(BackendError):
        registry.create("missing")
    first, second = registry.create("numpy-affine"), registry.create("numpy-affine")
    assert first is not second
    assert first.stateless is True
    assert first.interface_version == "1"
    assert first.version == "1.0"


def test_registry_rejects_shared_backend_across_registries():
    backend = NumpyAffineBackend()
    first, second = BackendRegistry(), BackendRegistry()
    for registry in (first, second):
        registry.register("shared", lambda: backend)
    assert first.create("shared") is backend
    with pytest.raises(BackendError):
        first.create("shared")
    with pytest.raises(BackendError):
        second.create("shared")


@pytest.mark.parametrize("attribute,value", [("stateless", False), ("interface_version", "2"),
                                             ("version", None), ("device", None),
                                             ("inference_mode", None), ("infer", None)])
def test_registry_rejects_incompatible_backend(attribute, value):
    backend = NumpyAffineBackend()
    setattr(backend, attribute, value)
    registry = BackendRegistry()
    registry.register("bad", lambda: backend)
    with pytest.raises(BackendError):
        registry.create("bad")


@pytest.mark.parametrize("text", ['{', '{"version":1,"version":2}', '[]'])
def test_malformed_manifest(bundle_factory, text):
    root = bundle_factory()
    (root / "manifest.json").write_text(text, encoding="utf-8")
    with pytest.raises(ManifestError):
        load_bundle(root)


@pytest.mark.parametrize("preprocessing", [
    {"required": True}, {"required": True, "artifact": "parameters"},
    {"required": True, "artifact": "absent", "version": "1"},
])
def test_required_preprocessing_metadata_is_mandatory(bundle_factory, preprocessing):
    with pytest.raises(ManifestError):
        load_bundle(bundle_factory(mutate=lambda m: m.update(preprocessing=preprocessing)))


@pytest.mark.parametrize("field,value,bound,dtype", [
    ("maximum", 0.1, 0.1, "float32"),
    ("minimum", -0.1, -0.1, "float32"),
    ("maximum", 2**63 + 1, 2**63, "uint64"),
    ("minimum", 2**63, 2**63 + 1, "uint64"),
])
def test_range_boundaries_do_not_round_to_array_dtype(field, value, bound, dtype):
    spec = TensorSpec("x", dtype, (1,), **{field: bound})
    with pytest.raises(ValidationError):
        validate_tensors({"x": np.array([value], dtype=dtype)}, (spec,))


def test_backend_ownership_requirement_has_explicit_error():
    class SlottedBackend:
        __slots__ = ()
        stateless = True
        interface_version = "1"
        version = "test-1"
        device = "cpu"
        inference_mode = "deterministic"
        def load(self, bundle): pass
        def infer(self, inputs): return inputs
        def close(self): pass
    registry = BackendRegistry()
    registry.register("slotted", SlottedBackend)
    with pytest.raises(BackendError) as caught:
        registry.create("slotted")
    assert caught.value.code == "backend_not_weak_referenceable"
    assert isinstance(caught.value.__cause__, TypeError)


@pytest.mark.parametrize("mutation", [
    lambda m: m.update(format_version=2),
    lambda m: m.update(format_version=True),
    lambda m: m["backend"].update(interface_version="2"),
    lambda m: m["inputs"].append(dict(m["inputs"][0])),
])
def test_unsupported_versions_and_duplicate_names(bundle_factory, mutation):
    with pytest.raises(ManifestError):
        load_bundle(bundle_factory(mutate=mutation))


def test_error_messages_are_structural_not_full_inputs():
    private = np.array([12345.6789, 98765.4321])
    with pytest.raises(ValidationError) as caught:
        validate_tensors({"x": private}, (TensorSpec("x", "float64", (3,)),))
    message = str(caught.value)
    assert "shape_mismatch" in message and "input.x" in message
    assert "expected=(3,)" in message and "actual=(2,)" in message
    assert "12345" not in message and "98765" not in message


def test_bundle_fingerprint_changes_with_manifest_or_artifact(bundle_factory):
    first = load_bundle(bundle_factory(version="v1"))
    second = load_bundle(bundle_factory(version="v2"))
    third = load_bundle(bundle_factory(version="v1", scale=2))
    assert len({first.fingerprint, second.fingerprint, third.fingerprint}) == 3
    assert all(len(value) == 64 for value in
               (first.fingerprint, second.fingerprint, third.fingerprint))
