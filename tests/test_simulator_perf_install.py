"""Offline installer checks: temporary fixtures only, no simulator execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / 'tools/simulator_perf/official_launch20'
spec = importlib.util.spec_from_file_location('perf_install', KIT / 'install.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sdk = self.root / 'sdk'
        self.package = self.root / 'package'
        self.sdk.mkdir(); (self.package / 'overlay').mkdir(parents=True)
        self.files = {}
        for name in ('first.py', 'second.py'):
            (self.sdk / name).write_bytes(b'original')
            (self.package / 'overlay' / name).write_bytes(b'optimized')
            self.files[name] = {'before': hashlib.sha256(b'original').hexdigest(),
                                'after': hashlib.sha256(b'optimized').hexdigest()}
        self.save_manifest()
        self.patch = patch.object(installer, 'HERE', self.package)
        self.patch.start(); self.addCleanup(self.patch.stop)

    def save_manifest(self):
        (self.package / 'manifest.json').write_text(json.dumps(
            {'base_sdk_commit': 'fixture', 'files': self.files}))

    def test_check_does_not_write(self):
        result = installer.install(self.sdk, check=True)
        self.assertEqual(len(result['files_to_update']), 2)
        self.assertEqual((self.sdk / 'first.py').read_bytes(), b'original')

    def test_install_and_repeated_install(self):
        installer.install(self.sdk)
        self.assertEqual((self.sdk / 'first.py').read_bytes(), b'optimized')
        self.assertEqual(installer.install(self.sdk)['files_to_update'], [])

    def test_bad_destination_rejected_before_any_write(self):
        (self.sdk / 'second.py').write_bytes(b'local edit')
        with self.assertRaises(RuntimeError): installer.install(self.sdk)
        self.assertEqual((self.sdk / 'first.py').read_bytes(), b'original')

    def test_corrupt_source_rejected_before_any_write(self):
        (self.package / 'overlay/second.py').write_bytes(b'corrupt')
        with self.assertRaises(RuntimeError): installer.install(self.sdk)
        self.assertEqual((self.sdk / 'first.py').read_bytes(), b'original')

    def test_missing_new_file_allowed(self):
        (self.sdk / 'second.py').unlink()
        self.files['second.py']['before'] = None
        self.save_manifest()
        installer.install(self.sdk)
        self.assertEqual((self.sdk / 'second.py').read_bytes(), b'optimized')

    def test_shipped_sources_match_manifest(self):
        manifest = json.loads((KIT / 'manifest.json').read_text())
        for relative, hashes in manifest['files'].items():
            self.assertEqual(installer.digest(KIT / 'overlay' / relative), hashes['after'])
        for relative, wanted in manifest['runtime_files'].items():
            self.assertEqual(installer.digest(KIT / relative), wanted)


if __name__ == '__main__':
    unittest.main()
