# Generic Model Handoff

This is a generic provider checklist, not a source-file migration list. The
author's V13 local-avoid PPO was inspected read-only for an integration boundary.
The handoff copies only numerical Actor/RMS data into a verified bundle; both
external source trees remain unmodified.

## Provider Materials

| Category | Required information | Embedded or separate |
| --- | --- | --- |
| Model artifacts | Exact explicitly designated files, parameter data, format | Parameters can be in one numeric archive; each file remains listed and hashed |
| Structure/format | Architecture or documented format identifier compatible with a trusted backend | May be embedded in a self-describing artifact; still declare format in manifest |
| I/O contract | Names, dtype, concrete shapes, semantics, finite/range requirements, contract version | Metadata may be embedded, but manifest must state validated public contract |
| Preprocessing | Whether needed, exact transformations, artifact and semantic version | May be embedded only if backend explicitly supports and validates it; otherwise separate declared file |
| Dependencies | Actually validated package names and exact versions | Manifest plus reproducible installation guidance, not inferred from checkpoint |
| Provenance/hash | Source description, known and unknown origin facts, per-file SHA-256 | Manifest; separate release/signature evidence if authenticity is required |

Hashes prove agreement with the supplied manifest, not publisher identity,
safety or accuracy. A single file can contain several categories, but does not
remove the need for an explicit contract and supported trusted loader. Never
guess architecture from an arbitrary checkpoint or disable safe loading.

## Manual Placement

1. Obtain provider-designated artifacts and the categories above. Record unknown
   facts honestly; do not infer a release from `best`, `latest` or file time.
2. Select a new explicit `models/<model_id>/<version>/` directory. Do not overwrite
   active or published versions. Keep filenames relative and contained.
3. Place numeric artifacts under `artifacts/`, write the manifest with their
   actual SHA-256 values, and include a README explaining format and semantics.
4. Install only reviewed dependencies in the project's dedicated environment.
   Add any new backend through trusted application code, never a bundle import
   path or bundle script. Stateful/recurrent backends are unsupported here.
5. Run `inspect-bundle` and `validate-bundle`. Unsupported preprocessing,
   interface versions, formats or contracts must fail, never become identity.
6. Separately run authorized inference fixtures against independently specified
   expected values. Static validation alone does not do this.
7. Explicitly select the version for each new session. Existing sessions retain
   their loaded parameters. Domain-task acceptance is separate and not included.

Do not execute unknown package scripts, download model code, copy entire source
repositories into the package, or add external repositories to Python paths.

## Backend Interface Contract

A reusable trusted backend exposes class metadata `stateless = True`,
`interface_version`, implementation `version`, `device` and `inference_mode`.
Its factory returns a fresh weak-referenceable instance. `load(CheckedBundle)`
validates and owns resources, `infer(Mapping[str, ndarray])` returns exactly the
declared named arrays, and `close()` is safe to repeat. A separate static
`validate_bundle` rejects unknown formats, contracts, preprocessing and
dependencies without constructing the backend or loading parameters. Runtime
owns boundary copies, but backend code must still treat inputs as read-only and
must not expose mutable internal weights.

## Audited V13 Boundary

The read-only source model is
`local_avoid_ppo_v13_gate_aligned_s3_best.pt`, documented by its provider with
SHA-256 `632662d47fb35edfdb10d680a359cced03fab75af09cff666631a97f0ccf87f3`.
Its deployment Actor accepts a 33-element float32 observation, normalizes it with
stored actor RMS, runs an MLP whose checkpoint configuration is authoritative,
and selects one of three actions deterministically. The 43-D Critic input is not
needed for deployment action selection. Mission encoding, `mission_v12` gating
and mapping actions to `-20G/0/+20G` belong in an environment adapter, not in the
numerical backend.

The original checkpoint loader uses `torch.load(..., weights_only=False)`. The
`skills/local_avoid_v13` package replaces that boundary with an exact checkpoint
SHA-256 check followed by `weights_only=True` and a minimal NumPy type allowlist.
Its conversion tool exports only Actor tensors, actor RMS and JSON configuration
to a pickle-free NPZ/JSON bundle. The original checkpoint is never loaded by
Runtime and optimizer/Critic/training state is not shipped.

The official environment adapter temporarily maps the external V13 policy onto
the official `ppo` hook and patches only symbols already imported by
`core/main.py`. No `competition_envs` file is edited. A separate `original` mode
loads the author's runtime source read-only with the same safe checkpoint payload,
enabling paired tests in the identical official environment.

For PlanSpec-v2 evaluation, the separate `local-avoid-v13-plan` entry uses the
authorized Plan Anchor release read-only. It intentionally runs the release's
historical explicit-launch engine because deterministic waves, held reserves and
planned launch targets are incompatible with stock auto-launch semantics. The
summary labels the true plan and V13 implementation; this mode must not be
reported as a stock current-`competition_envs` run.
