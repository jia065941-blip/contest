# Model Bundle Contract

Bundle layout: `models/<model_id>/<version>/manifest.json`, `artifacts/<file>`,
provider `README.md`, and optional explicitly supported preprocessing metadata.
This release requires JSON manifest format version `1` and concrete nonnegative
integer tensor dimensions (affine dimensions must be strictly positive).

## Manifest Fields

| Field | Meaning |
| --- | --- |
| `format_version` | Integer `1`; no implicit version upgrade |
| `model_id`, `version` | Explicit model identity and immutable release version |
| `backend.id`, `backend.interface_version` | Registered backend ID and interface version |
| `artifacts` | List of unique `{name, path, sha256}` declarations |
| `inputs`, `outputs` | Lists of `{name, dtype, shape, finite, minimum?, maximum?}` |
| `contract_version` | Version of input/output meanings, not merely their dimensions |
| `preprocessing` | `{required: false}` for affine; if true, require `artifact` (declared artifact name) and `version`; unsupported transformations rejected |
| `model_format` | Exact backend-supported structural/serialization identifier |
| `dependencies` | Package name to exact actually validated installed version |
| `source` | Nonempty source statement; explicitly mark unverifiable facts |

The framework dependency range and bundle dependency contract are different.
`pyproject.toml` describes versions on which the framework may be installed;
each bundle records the exact version validated for that artifact. A mismatch
raises `DependencyCompatibilityError` with the package, required version and
installed version. `constraints.txt` pins the fully tested release environment.

The executable generator `skill_generation.demo.create_demo_bundle` supplies a
complete example with a real SHA-256 and installed NumPy version, rather than
publishing a fake hash. Its affine contract is `affine-v1`, format
`numpy-affine-npz-v1`, backend `numpy-affine`, interface `1`. Input `x` is shape
`[n]`, output `y` shape `[m]`, both matching float32 or float64. Artifact named
`parameters` is an NPZ archive containing `weight` shape `[n,m]` and `bias`
shape `[m]`. All parameter values must be finite. Arithmetic is exactly
`x @ weight + bias`; validators never cast, clip, reorder, fill or transform.

## Validation And Trust

Malformed/missing fields, unsupported versions/shapes, nonexistent files,
hash mismatches, absolute paths, traversal and symlinks escaping the package
are errors. The manifest is not executable configuration. There is no arbitrary
module path, plugin discovery, network download or dependency installation.
Static inspection reads JSON and hashes files, but does not load numeric arrays.
Registry validation checks the trusted backend's format/semantic contract and
runtime dependencies. Loading separately checks numeric parameters with
`numpy.load(..., allow_pickle=False)`; object deserialization is forbidden.
`load_bundle()` computes a SHA-256 fingerprint over the exact manifest bytes and
every declared artifact identity and digest. Metadata, formatting or artifact
changes therefore change identity even if `model_id` and `version` are reused.
This detects changed content; it is not a publisher signature.

Input/output mappings must contain exactly the declared names and NumPy arrays
with exact shape and dtype. Finite and explicitly declared inclusive range
checks apply. Errors carry a code, field, expected value and actual structure,
not full input data. Invalid input leaves the session READY; invalid output is a
backend fault and leaves it ERROR. Backend load/infer failures preserve exception
chains and never synthesize a result.

Only trusted application code can register a backend. `register(id, factory,
validator=static_check)` attaches an explicit static contract checker; when
omitted, `factory.validate_bundle` is used if present. Static checks never invoke
the factory. A backend must declare itself stateless, provide interface and
implementation versions, explicit `device` and `inference_mode`, and implement
`load(bundle)`, `infer(named_arrays)` and idempotent `close()`. Every session gets an
exclusive instance in this release; hidden recurrent state and shared model
ownership are not implemented. Bundle files are assumed immutable while being
checked/loaded. Hash validation is not a sandbox against an attacker concurrently
rewriting files or a proof of publisher authenticity; filesystem permissions and
trusted handoff remain required.

Factories must return fresh, weak-referenceable backend instances. Reusing a
live instance, even via another registry, is rejected so closing one session
cannot release another session's backend. Trusted factories must also avoid
unmanaged shared resources internally. Backends are executable trusted
application code, not a sandbox boundary. Each supplied backend is isolated.
File validation currently uses Linux/POSIX no-follow directory descriptors;
non-POSIX portability has not been implemented or verified. All symlinks and
multiply hard-linked artifact/manifest files are rejected, a stricter rule than
only rejecting links that escape the bundle.

## Updates

Parameter-only updates may use a new directory when structure, semantics,
preprocessing and backend contract remain identical. A new session explicitly
selects it; current sessions use parameters already loaded in memory. Structural,
semantic or preprocessing changes require a new contract version and may require
a new trusted backend. Identical dimensions alone never prove compatibility.
Dependency changes require a newly recorded and revalidated release. External
environment changes never authorize rewriting bundles or faking values.

The compatibility boundary is:

- Parameter-only release: unchanged model format, tensor semantics,
  preprocessing, backend interface and dependencies; add a new version directory
  and hashes, with no Runtime change.
- Structure or inference-contract release: change `model_format` and/or
  `contract_version`, update or add the trusted backend, and revalidate fixtures.
- Runtime-dependency release: record the new exact dependency version and test
  it before publishing; compatibility is never assumed automatically.

"Only replace parameters later" is conditional on unchanged contracts, not a
guarantee for arbitrary new models. No latest-version selection, hot swapping,
automatic fallback, silent downgrade or background decision logic is provided.
