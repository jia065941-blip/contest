# skill_generation

Independent, sequential numerical inference infrastructure. It loads explicitly
selected local bundles, checks named arrays without modifying them, runs a
trusted registered backend and records lifecycle metadata. The supplied backend
performs `y = x @ weight + bias` using NumPy. Concrete independently installable
Skills live under top-level `skills/`; `skills/local_avoid_v13` supplies the V13
PPO Actor and official-environment adapter without adding PyTorch to this shared
framework package.

## Install And Run

Requires Python 3.11+ on Linux/POSIX (the hardened file reader uses POSIX file
descriptors). From the project root:

```bash
python -m venv .venv
mkdir -p .tmp
TMPDIR="$PWD/.tmp" .venv/bin/python -m pip install --no-cache-dir -c constraints.txt '.[test]'
.venv/bin/python -m skill_generation.cli demo --work-dir .
.venv/bin/python -m pytest
```

Do not install into another project's environment. The current dedicated `.venv`
is already installed and tested; see `docs/ACCEPTANCE.md` for actual commands.
No external repository belongs on `PYTHONPATH`. NumPy is the named-array contract
dependency as well as the bundled numerical backend dependency. `pyproject.toml`
declares the framework compatibility range, `constraints.txt` reproduces the
fully tested environment, and each bundle declares the exact dependency version
validated for that model. Other backend
dependencies are not imported or installed by the base package; future backends
must declare their own optional dependency groups and be explicitly registered.

The demo creates an ephemeral deterministic bundle, validates and loads it,
returns `{"y": [11.5, 15.5]}`, and removes the fixture. It never starts a simulator.
The retained harmless bundle `models/harmless-affine/1.0.0` was generated and
verified locally with NumPy 2.5.4. It can be inspected with the commands below.
To generate, inspect, validate and run another fixture on your installed NumPy,
use this complete disposable sequence (the helper never overwrites):

```bash
rm -rf .tmp/readme-example
.venv/bin/python -c "from pathlib import Path; from skill_generation.demo import create_demo_bundle; create_demo_bundle(Path('.tmp/readme-example/harmless-affine/1.0.1'), version='1.0.1')"
.venv/bin/python -m skill_generation.cli inspect-bundle .tmp/readme-example/harmless-affine/1.0.1
.venv/bin/python -m skill_generation.cli validate-bundle .tmp/readme-example/harmless-affine/1.0.1
.venv/bin/python -c "import numpy as np; from skill_generation.skills import InferenceSkill; s=InferenceSkill('.tmp/readme-example/harmless-affine/1.0.1'); s.open(); print(s.infer({'x': np.array([2., 3.])}).outputs['y']); s.close()"
```

Inspection parses the manifest and verifies files/hashes without loading model
parameters. Validation additionally checks a known registered backend contract
and declared runtime dependencies. Neither command proves inference correctness
or task success. Unknown backends never fall back to another backend.

## Python API

```python
from pathlib import Path
import numpy as np
from skill_generation.monitor import Monitor
from skill_generation.skills import InferenceSkill

monitor = Monitor(jsonl=Path("logs/inference.jsonl"))
with InferenceSkill("models/harmless-affine/1.0.0", monitor=monitor) as skill:
    result = skill.infer({"x": np.array([2.0, 3.0], dtype=np.float64)})
    print(result.outputs["y"], result.model_version, result.call_id)
```

The log parent must exist. A log-write failure is reported to stderr and does
not substitute outputs or suppress backend errors. `Monitor(enabled=False)`
disables records, not validation. `configs/runtime.example.toml` is explanatory
application configuration, not an automatically loaded CLI configuration file.

Each session has its own backend, ID, counters and bounded error history
(`max_error_history=100`; `total_error_count` is never truncated). `open()` moves
NEW to READY; invalid input rejects only that call; load/backend/output faults
move to ERROR; `close()` releases that session and is repeatable. Calls in NEW,
ERROR or CLOSED are rejected. `call_count` counts attempts made while READY,
including rejected input; `success_count` counts only validated results.
Runtime gives the backend owned input copies and returns owned output copies, so
caller arrays cannot be modified and a reused backend output buffer cannot change
an earlier result. Hidden recurrent state is unsupported. There is deliberately no model resource
sharing, automatic selection, downgrade, hot replacement, clipping, padding,
casting, output replacement or implicit preprocessing.

## Delivery

- `docs/LOCAL_AUDIT.md`: inspected facts, boundaries and unknowns.
- `docs/MODEL_BUNDLE.md`: manifest and compatibility rules.
- `docs/MODEL_HANDOFF.md`: provider materials and manual placement.
- `docs/ACCEPTANCE.md`: actual execution and limitations.

The shared root package deliberately contains no domain observations, action
mapping, control rules, training, optimization or simulator dependencies. The
separate `skills/local_avoid_v13` package integrates the author's V13 Actor and
a process-local adapter for unmodified `competition_envs`; see that package's
README and `docs/MODEL_HANDOFF.md`.

Each session currently loads an independent backend. For the bundled 554-byte
parameter artifact, two simultaneous sessions used distinct backend objects and
the measured process peak was about 33.5 MiB on the recorded environment. This
is acceptable for the current model, so shared weight caching remains a future
model-specific optimization rather than an unverified framework feature.
