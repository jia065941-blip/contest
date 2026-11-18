"""Normalize scene files, validate them, and refresh the catalogue checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from envengine.environment.scene_migration import migrate_scene
from .scenario_checker import validate_scene


SCENARIO_ROOT = Path(__file__).resolve().parent


def generate_scenarios(paths=None):
    """Write the common pos/vel format while preserving IDs, health and topology."""
    selected = list(paths) if paths is not None else sorted(
        path for path in SCENARIO_ROOT.glob("*.json") if path.name != "manifest.json")
    generated = []
    for source in selected:
        path = Path(source)
        data = migrate_scene(json.loads(path.read_text(encoding="utf-8")))
        validate_scene(data)
        content = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        path.write_text(content, encoding="utf-8", newline="\n")
        generated.append(path)
    manifest_path = SCENARIO_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(schema_version=2, version="20261103", geometry_revision="local_xyz_m_20261103",
                    coordinate_system={"x": "east", "y": "north", "z": "height", "unit": "metre"})
    for case in manifest["cases"]:
        path = SCENARIO_ROOT / case["scenario"]
        case["files_sha256"][case["scenario"]] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8", newline="\n")
    return tuple(generated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Scene JSON files; defaults to the whole catalogue")
    parser.add_argument("--redraw", action="store_true", help="Refresh the twenty catalogue diagrams")
    args = parser.parse_args()
    generated = generate_scenarios(args.paths or None)
    print(f"Validated and wrote {len(generated)} scenes")
    if args.redraw:
        from .cases.plot_maps import main as redraw
        import sys
        sys.argv = [sys.argv[0]]
        redraw()


if __name__ == "__main__":
    main()
