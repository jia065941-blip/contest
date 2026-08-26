"""Build reproducible evaluation command matrices without duplicating engine logic."""

from __future__ import annotations

import itertools
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationCase:
    blue_policy: str
    scenario: str
    seed: int


def enumerate_cases(blue_policies: list[str], scenarios: list[str], seeds: list[int]) -> list[EvaluationCase]:
    return [EvaluationCase(*values) for values in itertools.product(blue_policies, scenarios, seeds)]
