# Local Audit

Audit date: 2026-10-15. Explicit file inspection is confined to
`/home/amax/ry/competition`; all explicitly selected write destinations are
confined to `skill_generation`. See the installer temporary-directory caveat below.

## Confirmed Facts

- Target `/home/amax/ry/competition/skill_generation` existed and was empty,
  including no hidden entries returned by the directory reader. There were no
  pre-existing target files to replace or preserve.
- Parent entries: `competition_envs`, `glibc-2.38`, `jmy_ppo`,
  `personal_train_ppo`, `personal_train`, `Results`, `skill_generation`.
  Unrelated directories are not implementation inputs.
- Official repository: `/home/amax/ry/competition/competition_envs`.
  Read-only `.git/HEAD` points to `refs/heads/pku`; that loose ref contains
  `235017020658a356a07f00aaaf161275eb0f2c4f`.
- Author repository: `/home/amax/ry/competition/jmy_ppo/contest` (not its parent).
  `.git/HEAD` points to `refs/heads/main`; the loose ref contains
  `c2f756613376d5b6f3f5b38eb4c08551c2749f78`.
- Both inspected requirements files declare NumPy `2.4.6` and various
  environment dependencies. These declarations are not evidence of installed
  versions, model compatibility, release provenance, or executable interfaces.
- Initial interpreter: `/home/amax/miniconda3/bin/python`, CPython 3.13.9,
  prefix and base prefix `/home/amax/miniconda3`. Package metadata reported
  NumPy 2.5.2, pip 25.2, setuptools 80.9.0, wheel 0.45.1, and no pytest.
- No `pyvenv.cfg` was found by the initial project-tree glob. This is not proof
  that other environment managers have no environments.
- A dedicated isolated environment was created with `python -m venv
  /home/amax/ry/competition/skill_generation/.venv`. No system packages or
  external project environments are modified.

## Boundaries And Unknowns

Git commands were omitted as permitted by the user. External working-tree
cleanliness, tags, remote provenance and uncommitted changes are not confirmed;
the commit refs above do not certify clean working trees. No files in those
repositories were changed. Only root listings,
  requirements metadata and Git ref metadata were read during the first phase.
  The second phase read the author's V13 model documentation, loader, runtime,
  encoder, action mapping and E01-E03 entry points, plus matching official
  environment entry points. Integration later executed those read-only sources
  through an explicit adapter; only Actor/RMS numerical values were converted
  into the allowed project directory.

The initial target contained no package configuration or user modifications.
All implementation, docs, generated demo models, temporary test data, logs,
build artifacts and installation environment belong under `skill_generation`.
Test subprocesses use this project environment and exclude external projects
from their Python search paths. No model release is inferred from directory
names, timestamps, evaluation folders, `best` or `latest`.

Actual dependency installation and test outcomes are recorded in ACCEPTANCE.md.
The final interpreter is `.venv/bin/python` (3.13.9), with NumPy 2.5.4,
pytest 9.1.1, pip 25.2, setuptools 84.0.0 and wheel 0.48.0. The installed
skill-generation version is 0.1.0; import location was confirmed under this
environment's `site-packages`. No runtime/install blocker remained for the
  Linux numerical infrastructure after verification.

## V13 Integration Findings

- The author model is `local_avoid_ppo_v13_gate_aligned_s3_best.pt`; provider
  documentation states SHA-256
  `632662d47fb35edfdb10d680a359cced03fab75af09cff666631a97f0ccf87f3`.
- Deployment uses a 33-D float32 Actor observation, stored actor normalization,
  deterministic three-action selection, external mission gating and external
  engine action mapping. The checkpoint configuration, not source defaults,
  must define the eventual backend structure.
- The source loader uses `torch.load(..., weights_only=False)`; the new adapter
  never calls it and instead uses digest-pinned `weights_only=True` with a minimal
  NumPy type allowlist.
- `competition_envs` does not natively register the author's policy. The
  process-local wrapper provides both original and InferenceSkill modes without
  editing official files, and forces both modes to the same official scenarios.
- The bundled model is tiny and independent loading is acceptable: the parameter
  artifact is 554 bytes (4 KiB filesystem allocation); two open sessions used
  separate backend instances and the measured process peak was 34,304 KiB.
  Shared weights are deferred until a real backend demonstrates a need and safe
  immutable ownership semantics.

## V13 Integration Result

- The Actor-only NPZ is 434,518 bytes with SHA-256
  `90bd81310b581fd6fcbb792c917cb15a630944077807005a46bbf4c9d82884d4`;
  the bundle fingerprint is
  `b424c4d91e052794f71fcc8b2e741bcb6644fb59b005d8260590c78fb2954944`.
- 258 Actor inputs and four sequential raw-observation encoder cases matched the
  original implementation exactly (`max_abs_error=0`). One thousand mission
  gate cases produced identical masks and recovery state.
- E01, E02 and E03 completed one-step official-environment smoke tests with 25,
  26 and 23 valid InferenceSkill calls and zero Runtime errors. Original-author
  mode also completed E01 in the same official environment.
- The simulator extension requires glibc 2.38. The Skill-local launcher combines
  the existing private loader with the requested `competition` Conda Python;
  neither external environment is modified.
- The initial full user comparison used legacy `easy/E01-E03` and no fixed
  deployment, so its zero scores are not V13 Final20 performance evidence. The
  adapter now accepts `--deployment-plan preset`, resolves the file beside the
  selected Final20 scenario, forces R9 plus `mission_v12`, and rejects a missing
  preset. Final20 E01 original/Skill smoke summaries matched.

## Plan Anchor Integration

- `/home/amax/ry/competition/plan-anchor-final20-v2` was used read-only with
  team authorization. No file in that checkout was changed or copied wholesale.
- The actual upper layer is the release snapshot's PlanSpec-v2,
  `DeterministicPlanPolicy`, `UnifiedAttackMissileAgent` and `InfoCenter`, not the
  repository root's older R9 class.
- The snapshot omits envengine, so the adapter composes snapshot policy/core code
  with the checkout's historical explicit-launch envengine. A process-local
  lifecycle bridge supplies phased reporting, ordered decisions, satellite
  settlement and honest telemetry fallback without editing the external code.
- Release verification passed for all 240 plans and 1,200 published result rows.
  E01 A00 produced identical original/Skill results and a positive official score
  of `12.643678160919539`, resolving the stock auto-launch zero-score mismatch for
  the historical PlanSpec evaluation protocol.

Only `/home/amax/ry/competition/skill_generation` was modified. The target is
not a Git repository, so Git status/commit operations were unavailable and not
required. The shared framework was tested on CPython 3.13.9. The V13 package and
official adapter were tested on CPython 3.11.17 from the requested Conda
environment. Python 3.12 was unavailable and remains unverified.

## Installer Temporary-Directory Caveat

The initial project-venv dependency installation used `--no-cache-dir`, but did
not explicitly set `TMPDIR`. Pip's internal transient-file destinations for
that command were not verified, so strict confinement of those transient files
cannot be claimed. No outside paths were inspected or cleaned afterward.
Subsequent package installations used project-local temporary paths. Pytest's
default external `tmp_path` is used for portability tests. No global package
installation was performed.
