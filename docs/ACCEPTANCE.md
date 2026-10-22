# Acceptance Record

Executed on 2026-10-15 on Linux. The current shared-framework result is **134 passed,
0 failed, 0 skipped**, in 0.76 seconds. These are generic infrastructure tests using
harmless arithmetic, not task-model or environment acceptance.

## Actual Environment

| Item | Observed value |
| --- | --- |
| Interpreter | `/home/amax/ry/competition/skill_generation/.venv/bin/python` |
| Python | CPython 3.13.9, GCC 11.2.0, Anaconda-built base interpreter |
| Prefix | `/home/amax/ry/competition/skill_generation/.venv` |
| Base prefix | `/home/amax/miniconda3` |
| Package | skill-generation 0.1.0 |
| NumPy | 2.5.4 |
| pytest / pluggy | 9.1.1 / 1.6.0 |
| pip / setuptools / wheel | 25.2 / 84.0.0 / 0.48.0 |
| Imported package | `.venv/lib/python3.13/site-packages/skill_generation/__init__.py` |

Installation is a normal built-wheel installation, **not editable**, and tests
run against the installed package using Python isolated mode. No external
repository is added to a Python search path. Test subprocesses also assert this
and check that optional/external modules are not imported.

## Actual Commands And Outcomes

Unless stated otherwise, working directory was
`/home/amax/ry/competition/skill_generation`.

1. `python -m venv .venv` created the dedicated project environment. The
   reproducible install command is `TMPDIR="$PWD/.tmp" .venv/bin/python -m pip
   install --no-cache-dir -c constraints.txt '.[test]'`; the verified pins are
   NumPy 2.5.4, pytest 9.1.1, setuptools 84.0.0 and wheel 0.48.0.
2. `TMPDIR="$PWD/.tmp" .venv/bin/python -m pip install --no-build-isolation
   --no-deps --force-reinstall .` exited **0**. The installed package is a normal
   wheel installation, not editable.
3. `.venv/bin/python -I -m pytest` exited **0** with **134 passed, 0 failed,
   0 skipped**. Pytest's default system `tmp_path` was intentionally used so the
   installed-package working-directory test runs outside the source tree.
4. `.venv/bin/python -m pip check` exited **0** with `No broken requirements
   found.`
5. The complete README sequence generated version `1.0.1`, inspected that same
   path, validated it and inferred `[11.5, 15.5]`; every command exited **0**.
6. `.venv/bin/python -m pip wheel --no-build-isolation --no-deps --wheel-dir
   /tmp/kilo/skill-generation-final-wheel .` exited **0**. The final wheel
   `skill_generation-0.1.0-py3-none-any.whl` had SHA-256
   `3fbb4467b13e70c9d4cd437697107c2072f40b414aa25fe7d49df10a703f3f5e`.
7. A fresh `/tmp/kilo/skill-generation-final-venv` installed only that wheel and
   the NumPy 2.5.4 wheel. From a separate `/tmp/skill-generation-final-run-*`
   working directory, `python -I -m skill_generation.cli demo` exited **0** and
   returned `[11.5, 15.5]`, model/session/backend IDs and a 64-hex fingerprint.
8. `python -m zipfile -l` on the wheel showed only package modules and standard
   `.dist-info`; no build directory, cache, test, log, model or temporary file
   was included.
9. `sha256sum` independently verified the read-only V13 file as
   `632662d47fb35edfdb10d680a359cced03fab75af09cff666631a97f0ccf87f3`
   (2,643,498 bytes). It was deserialized only through the digest-pinned
   `weights_only=True` reader; `weights_only=False` was not used by this package.

## Coverage Evidence

| Requirement | Exercised behavior |
| --- | --- |
| Real parameter loading | Independent NPZ fixture; float32/float64; hand-calculated positive, zero and negative cases |
| End-to-end equivalence | Direct backend versus skill plus validator and monitor, unchanged input arrays |
| Bundle failures | Required fields, malformed/duplicate-key JSON, missing files, changed hashes, unsafe paths, external symlinks |
| Unsupported contracts | Unknown backend, preprocessing metadata, versions, formats, semantic contracts, dynamic dimensions, dependency mismatch |
| Strict inputs | Exact names/count, shape, dtype, array type, NaN/Inf, inclusive ranges, no filling/casting |
| Strict outputs | Invalid names/shape/dtype/type/finiteness/ranges rejected, ERROR state, no success count |
| Numeric artifact safety | Missing/extra parameters, wrong shapes/dtypes, nonfinite/object arrays rejected; pickle disabled |
| Lifecycle and exceptions | Load/infer/close faults, exception chains, invalid order, repeated close, context cleanup, nested-entry regression |
| Failed-load rollback | Failure before allocation and after allocation, cleanup failure preserving the primary exception, idempotent later close |
| Session ownership | Independent IDs/counters/errors, independent parameters, one close leaves other sessions functional, singleton factories rejected across registries |
| Array ownership | Backend mutation cannot affect caller input; reused backend output buffers cannot alter prior results |
| Explicit versions | v1/v2 selected independently, old session retains v1 and old numerical results |
| Monitoring | On/off/write-failure/throwing monitor produce the same result; failures do not hide backend errors; identity metadata verified |
| Bounded diagnostics | Configurable recent-error deque is capped while `total_error_count` retains all failures |
| Content identity | Manifest and artifact changes produce distinct stable SHA-256 bundle fingerprints |
| Installation/isolation | Installed package imported from site-packages outside root with `-I -B`; no external repo search paths; base import loads no NumPy/PyTorch/backend |
| CLI | Static commands execute without array loading; success exit 0, diagnostic failures exit 2; root-external demo executes real inference |
| Extensibility | Explicit custom trusted backend and static validator, no factory call during static validation |
| Precision regression | Range checks preserve float32 value versus manifest boundary and uint64 precision |

The fixture generator is independent of production demo code and lives in
`tests/conftest.py`; generated data uses pytest temporary directories. Unit
tests live in `tests/unit`, integration tests in `tests/integration`; no external
project was renamed, moved, deleted, imported or executed for isolation.

## Limits And Unexecuted Work

### V13 Acceptance

- Independent package suite: `20 passed, 0 failed, 0 skipped` in 0.86 seconds.
- Original Actor comparison: 258 observations, maximum absolute logit error `0`,
  all argmax actions equal.
- Original encoder comparison: four sequential no/one/two/three-threat cases,
  maximum absolute feature error `0`.
- Original mission gate comparison: 1,000 generated cases, all masks and recovery
  states equal.
- Official smoke tests: E01/E02/E03 each ran one engine step and produced valid
  summaries labeled `local_avoid_v13`; Runtime completed 25/26/23 inference calls
  with zero errors. Original-author mode also ran E01 and was labeled
  `local_avoid_v13_original`.
- Final wheels were installed together into a fresh `/tmp` Python 3.11 venv with
  system scientific packages. Actor inference from an unrelated external working
  directory returned `[-1.0860736, 1.6324025, -0.99082255]` and the expected
  bundle fingerprint. Wheel SHA-256 values were
  `de6163213c31b723b9aa8147e28216baf823afec2423e242546cb39cdf7694b8`
  for the framework and
  `f407e657a220535aadf08e0220d75f8019655a58f66f8832e55bc131d6101e63`
  for V13.
- The installed `local-avoid-v13` entry point automatically re-executed through
  the private glibc 2.38 loader and completed an E01 official smoke test.
- Final20 preset resolution is tested for E01, E02 and E03. A legacy `easy/E01`
  path is rejected when it claims `--deployment-plan preset`. Both Skill and
  original-author modes completed a Final20 E01 one-step smoke test with the same
  fixed deployment: 126 red platforms, 19 launches, 114 blue entities and 29
  objective-weight points. Both explicitly used the official R9 commander and
  `mission_v12` gate.
- The user-completed full paired runs used seed 701 and the full official limits
  of 1200/1500/1600 steps. For E01, E02 and E03, original and Skill summaries
  were identical except for the policy label: both scored 0, destroyed no
  objectives, lost no red platforms and launched every red platform. This proves
  adapter equivalence for these runs but also shows V13 produced no task reward
  under this official scenario/upper-layer combination. These were legacy
  `easy/E01-E03` cases without a fixed deployment plan, not the V13 Final20
  evaluation preset, and must not be used as its performance result.

- V13 is integrated through the separate `skills/local_avoid_v13` package. It
  has 20 passing tests on Python 3.11.17, NumPy 2.4.6 and PyTorch 2.6.0+cpu.
- `turn_left`, `turn_right` and `stop_turning` are separate installable packages.
  Their shared exact-duration lifecycle adds seven framework tests, and their
  three package tests all pass. Invalid/non-integer durations and invalid state
  transitions are rejected explicitly.
- Conversion and original-mode comparison use digest-pinned `weights_only=True`;
  `weights_only=False` was never used.
- PPO training, recurrent state, model sharing, hot reload,
  automatic model selection, alternate inference backends, databases and network
  services are not delivered in this first version.
- CPython 3.13.9 with NumPy 2.5.4 was tested for the shared framework. CPython
  3.11.17 with NumPy 2.4.6 and PyTorch 2.6.0+cpu was tested for V13. Python 3.12
  remains unverified. Non-Linux platforms were not tested.
- External working-tree cleanliness and model availability remain unknown. Git
  commands were omitted as allowed; metadata-only inspection is documented.
- The first pip command's internal temporary-file locations were not verified.
  Later installation/test/demo temporary destinations were explicitly project-local.

Conclusion: the generic infrastructure and separately packaged V13 Skill are
installed and verified. The wrapper supplies `skill` and `original` modes without
modifying `competition_envs`, enabling paired E01-E03 runs with identical official
scenarios, seed, R9 commander and engine. One-step new-chain tests passed for all
three scenarios and original mode passed E01. Full 1200/1500/1600-step score runs
  remain explicit evaluation commands rather than unit tests.

### PlanSpec-v2 Upper Layer

- The authorized Plan Anchor release remains a read-only source dependency. Its
  verifier passed: 25 protocol files, 71 snapshot Python files, 240 plans, 1,200
  score rows and 1,200 formula runs; published A00 mean is `92.18081954253796`.
- `local-avoid-v13-plan` runs the release's deterministic PlanSpec-v2, Unified
  Agent, scenario snapshot and historical explicit-launch engine. It replaces
  only the V13 numerical implementation (`original` or `skill`).
- E01/A00/B0/seed 701 completed in both modes at step 841. Both scored
  `12.643678160919539`, destroyed the same targets `2570/2573/2575`, discovered
  all four hidden ships and emitted 126 launches.
- Plan ID was
  `098c872358b7dd4378d55822ee5fb651911e5481045b42662e12714e4a916f4a`.
  Deployment matched 126/126 entities; launch step and target ledger matched
  126/126; semantic retarget violations were empty.
- The bundled historical TrainingEnv has no native lifecycle telemetry API, so
  `engine_confirmed_launches` is recorded as `null`, not fabricated. Command-level
  plan consistency and official score remain available.
