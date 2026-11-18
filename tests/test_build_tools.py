import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import build_tools.build_layout as build_layout

from build_tools.build_layout import (
    BUILD_ROOT, DIST_ROOT, KEEP_SOURCE_DIRS, MANIFEST_NAME, ROOT, artifact_path, clean_build_directory,
    collect_sources, require_runtime, retained_engine_file, runtime_identity,
)


class BuildToolsTests(unittest.TestCase):
    def test_current_runtime_and_extension_name(self):
        require_runtime()
        self.assertEqual(runtime_identity()["python"], "cpython-311")
        self.assertTrue(artifact_path("envengine/__init__.py").startswith("envengine/__init__."))

    def test_compile_real_sources_and_exclude_native_wrappers(self):
        sources = collect_sources()
        names = dict(sources)
        self.assertEqual(len(sources), len(names))
        self.assertEqual(names["envengine.action_validation"], "envengine/action_validation.py")
        for _, source in sources:
            self.assertTrue((ROOT / source).is_file())
            self.assertFalse(any(source.startswith(directory + "/") for directory in KEEP_SOURCE_DIRS))

    def test_keep_native_dependencies_and_exclude_engine_source(self):
        for relative in (
            "envengine/sdk/Util/_UtilsPy.pyd",
            "envengine/simulator/models/ACMMissileModel/Utils.dll",
            "envengine/render/map.png",
        ):
            self.assertTrue((ROOT / relative).is_file())
            self.assertTrue(retained_engine_file(ROOT / relative, relative))
        for relative in ("envengine/action_validation.py", "envengine/environment/training_env.py", "envengine/sdk/Util/UtilsPy.py", "envengine/sdk/Util/UtilsPy.pyi"):
            self.assertFalse(retained_engine_file(ROOT / relative, relative))

    def test_rebuilding_removes_existing_output(self):
        BUILD_ROOT.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="build-clean-check-", dir=BUILD_ROOT) as temporary:
            directory = Path(temporary)
            (directory / "old-record.txt").write_text("previous build", encoding="utf-8")
            clean_build_directory(directory)
            self.assertEqual(list(directory.iterdir()), [])

    def test_cleanup_rejects_source_directories(self):
        with self.assertRaises(ValueError):
            clean_build_directory(ROOT / "envengine")
        self.assertTrue((ROOT / "envengine/action_validation.py").is_file())

    def test_distribution_copies_current_entry_points_and_configuration(self):
        specification = importlib.util.spec_from_file_location(
            "_test_finalize_dist", ROOT / "build_tools/finalize_dist.py",
        )
        finalize = importlib.util.module_from_spec(specification)
        with patch.dict(sys.modules, {"build_layout": build_layout}):
            specification.loader.exec_module(finalize)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            build = source / "build"
            destination = build / "distribution"
            names = ("main.py", "simulator.py", "previewer.py", "config.yaml",
                     "requirements.txt", "README.md", "libUtils.so")
            for name in names:
                shutil.copy2(ROOT / name, source / name)
            for package in finalize.RUNTIME_PACKAGES:
                (source / package).mkdir()
                (source / package / "__init__.py").write_text("", encoding="utf-8")
            asset = source / "user_agents/example/resource.json"
            asset.parent.mkdir()
            asset.write_text('{"example": true}', encoding="utf-8")
            manifest = {"schema_version": 1, "runtime": runtime_identity(), "modules": {}}
            with (
                patch.object(build_layout, "ROOT", source),
                patch.object(build_layout, "BUILD_ROOT", build),
                patch.object(finalize, "ROOT", source),
                patch.object(finalize, "DIST_ROOT", destination),
                patch.object(finalize, "verified_build", return_value=manifest),
                patch.object(finalize, "require_runtime"),
                patch.object(finalize, "require_native_dependencies"),
                patch("builtins.print"),
            ):
                finalize.main()
            copied = json.loads((destination / MANIFEST_NAME).read_text(encoding="utf-8"))
            for name in names:
                self.assertEqual((destination / name).read_bytes(), (source / name).read_bytes())
                self.assertEqual(copied["files"][name], build_layout.sha256(source / name))
            self.assertEqual((destination / asset.relative_to(source)).read_bytes(), asset.read_bytes())


@unittest.skipUnless(
    (DIST_ROOT / MANIFEST_NAME).is_file()
    and json.loads((DIST_ROOT / MANIFEST_NAME).read_text(encoding="utf-8"))["runtime"] == runtime_identity(),
    "需要先构建与当前平台匹配的完整产物",
)
class BuiltDistributionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="distribution-check-", dir=BUILD_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.copy = Path(self.temporary.name) / "runtime"
        shutil.copytree(DIST_ROOT, self.copy)
        self.manifest = json.loads((self.copy / MANIFEST_NAME).read_text(encoding="utf-8"))

    def validate(self):
        code = (
            "import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); "
            "from run_compiled import validate_distribution; validate_distribution(Path(sys.argv[2]))"
        )
        return subprocess.run(
            [sys.executable, "-B", "-c", code, str(ROOT / "build_tools"), str(self.copy)],
            capture_output=True, text=True, encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

    def test_actual_distribution_is_complete(self):
        result = self.validate()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_extension_is_rejected_with_source_still_present(self):
        entry = self.manifest["modules"]["envengine.action_validation"]
        (self.copy / entry["artifact"]).unlink()
        self.assertTrue((ROOT / entry["source"]).is_file())
        result = self.validate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FileNotFoundError", result.stderr)

    def test_modified_map_is_rejected(self):
        path = self.copy / "scenarios/E01.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        path.write_bytes(path.read_bytes() + b"\n")
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), original)
        result = self.validate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ValueError", result.stderr)


if __name__ == "__main__":
    unittest.main()
