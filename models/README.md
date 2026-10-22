# Local Model Bundles

Place explicitly provided, immutable versions in `<model_id>/<version>/` with
`manifest.json`, declared `artifacts/` files, and a provider README. Supply
`preprocessing.json` only when required and supported by the selected backend;
this release's affine backend supports no preprocessing.

No task models are shipped or selected. Evaluation directories are not assumed
to be release bundles. See `../docs/MODEL_HANDOFF.md`. Never replace an active
version directory. The demo and test helper create harmless numerical fixtures.
