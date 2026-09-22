"""Checks for independently controllable blue/red/simulation seeds."""

from __future__ import annotations

import unittest

from run import build_parser, resolve_run_seeds


class RunSeedControlTests(unittest.TestCase):
    def parse(self, *arguments: str):
        return build_parser().parse_args(("run", *arguments))

    def test_legacy_seed_remains_shared_fallback(self) -> None:
        args = self.parse("--seed", "17")

        self.assertEqual(resolve_run_seeds(args), (17, 17, 17))

    def test_specific_seeds_are_independent(self) -> None:
        args = self.parse(
            "--seed", "17",
            "--blue-seed", "101",
            "--red-seed", "202",
            "--simulation-seed", "303",
        )

        self.assertEqual(resolve_run_seeds(args), (101, 202, 303))

    def test_partial_override_keeps_shared_fallbacks(self) -> None:
        args = self.parse("--seed", "17", "--blue-seed", "101")

        self.assertEqual(resolve_run_seeds(args), (101, 17, 17))


if __name__ == "__main__":
    unittest.main()
