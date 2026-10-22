"""Static inspection and a harmless end-to-end numeric demonstration."""

import argparse
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from .errors import SkillGenerationError


def _json_value(value):
    if is_dataclass(value):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect-bundle", "validate-bundle"):
        commands.add_parser(name).add_argument("bundle_dir", type=Path)
    demo = commands.add_parser("demo")
    demo.add_argument("--work-dir", type=Path, default=Path.cwd(),
                      help="Existing directory for temporary harmless fixture")
    args = parser.parse_args(argv)
    try:
        if args.command in ("inspect-bundle", "validate-bundle"):
            from .artifacts import load_bundle
            bundle = load_bundle(args.bundle_dir)
            if args.command == "validate-bundle":
                from .registry import create_default_registry
                create_default_registry().validate(bundle)
            print(json.dumps({"manifest": _json_value(bundle.manifest),
                              "model_fingerprint": bundle.fingerprint,
                              "files": {key: {"path": str(path), "bytes": path.stat().st_size}
                                        for key, path in bundle.artifacts.items()},
                              "notice": "Static checks only; not inference or task validation."},
                             indent=2))
        else:
            import numpy as np
            from .demo import create_demo_bundle
            from .skills import InferenceSkill
            with TemporaryDirectory(prefix="skill-demo-", dir=args.work_dir) as directory:
                bundle_dir = create_demo_bundle(Path(directory) / "1.0.0")
                with InferenceSkill(bundle_dir) as skill:
                    result = skill.infer({"x": np.array([2., 3.], dtype=np.float64)})
                    print(json.dumps({"outputs": {k: v.tolist() for k, v in result.outputs.items()},
                                      "call_id": result.call_id, "session_id": result.session_id,
                                      "model_id": result.model_id,
                                      "model_version": result.model_version,
                                      "model_fingerprint": result.model_fingerprint,
                                      "backend_id": result.backend_id,
                                      "elapsed_seconds": result.elapsed_seconds,
                                      "notice": "Harmless arithmetic only; no simulation or task model."}))
        return 0
    except SkillGenerationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"CLI_IO_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
