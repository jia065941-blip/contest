"""Bundled map identifiers and explicit custom JSON paths resolve locally."""

from contextlib import chdir
from pathlib import Path
import tempfile
import unittest

from scenarios import resolve_scenario_path


CATALOGUE = Path(__file__).resolve().parents[1] / "scenarios"


class ScenarioPathTests(unittest.TestCase):
    def test_bundled_ids_and_paths_resolve_from_an_unrelated_working_directory(self):
        with tempfile.TemporaryDirectory() as directory, chdir(directory):
            self.assertEqual(resolve_scenario_path(), CATALOGUE / "E01.json")
            for identifier in ("E01", "M01", "H01"):
                expected = CATALOGUE / f"{identifier}.json"
                for source in (identifier, f"{identifier}.json", f"scenarios/{identifier}.json",
                               f"./scenarios/{identifier}.json", expected):
                    with self.subTest(source=source):
                        self.assertEqual(resolve_scenario_path(source), expected)

    def test_explicit_custom_files_resolve_from_the_working_directory(self):
        with tempfile.TemporaryDirectory() as directory, chdir(directory):
            custom = Path(directory) / "maps/custom.json"
            custom.parent.mkdir()
            custom.write_text("{}", encoding="utf-8")
            local_e01 = Path(directory) / "E01.json"
            local_e01.write_text("{}", encoding="utf-8")
            for source, expected in (("maps/custom.json", custom), (custom, custom),
                                     ("E01.json", local_e01), ("E01", CATALOGUE / "E01.json")):
                with self.subTest(source=source):
                    self.assertEqual(resolve_scenario_path(source), expected)

    def test_unknown_ids_and_missing_json_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory, chdir(directory):
            for source in ("E99", "maps/missing.json"):
                with self.subTest(source=source), self.assertRaises(ValueError):
                    resolve_scenario_path(source)


if __name__ == "__main__":
    unittest.main()
