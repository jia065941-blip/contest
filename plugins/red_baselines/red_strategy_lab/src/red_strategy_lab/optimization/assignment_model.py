"""Dependency-free branch-and-bound solver for static assignment baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from ..domain import Platform, Target


@dataclass(frozen=True)
class AssignmentSolution:
    pairs: tuple[tuple[int, int, float], ...]
    objective: float
    exact: bool


class AssignmentModel:
    """Maximize pair score subject to one assignment per platform and target capacity."""

    def __init__(self, target_capacity: int | Mapping[int, int], candidate_limit: int = 6, exact_platform_limit: int = 12) -> None:
        self.target_capacity = target_capacity
        self.candidate_limit = candidate_limit
        self.exact_platform_limit = exact_platform_limit

    def _capacity_for(self, target_id: int) -> int:
        if isinstance(self.target_capacity, Mapping):
            return int(self.target_capacity.get(target_id, 0))
        return int(self.target_capacity)

    def solve(
        self,
        platforms: Sequence[Platform],
        targets: Sequence[Target],
        score: Callable[[Platform, Target], float],
    ) -> AssignmentSolution:
        ordered_platforms = tuple(sorted(platforms, key=lambda item: item.entity_id))
        ordered_targets = tuple(sorted(targets, key=lambda item: item.entity_id))
        choices: list[list[tuple[Target, float]]] = []
        for platform in ordered_platforms:
            ranked = sorted(((target, score(platform, target)) for target in ordered_targets), key=lambda item: (-item[1], item[0].entity_id))
            choices.append([(target, value) for target, value in ranked[: self.candidate_limit] if value > 0])
        if len(ordered_platforms) > self.exact_platform_limit:
            return self._greedy(ordered_platforms, choices)
        return self._branch_and_bound(ordered_platforms, choices)

    def _greedy(self, platforms: Sequence[Platform], choices: Sequence[Sequence[tuple[Target, float]]]) -> AssignmentSolution:
        counts: dict[int, int] = {}
        pairs: list[tuple[int, int, float]] = []
        for platform, options in zip(platforms, choices):
            selected = next((item for item in options if counts.get(item[0].entity_id, 0) < self._capacity_for(item[0].entity_id)), None)
            if selected is None:
                continue
            target, value = selected
            counts[target.entity_id] = counts.get(target.entity_id, 0) + 1
            pairs.append((platform.entity_id, target.entity_id, value))
        return AssignmentSolution(tuple(pairs), sum(item[2] for item in pairs), exact=False)

    def _branch_and_bound(self, platforms: Sequence[Platform], choices: Sequence[Sequence[tuple[Target, float]]]) -> AssignmentSolution:
        upper_tail = [0.0] * (len(platforms) + 1)
        for index in range(len(platforms) - 1, -1, -1):
            upper_tail[index] = upper_tail[index + 1] + (choices[index][0][1] if choices[index] else 0.0)
        best_score = 0.0
        best_pairs: tuple[tuple[int, int, float], ...] = ()
        counts: dict[int, int] = {}
        selected: list[tuple[int, int, float]] = []

        def visit(index: int, score_total: float) -> None:
            nonlocal best_score, best_pairs
            if score_total + upper_tail[index] < best_score - 1e-12:
                return
            if index == len(platforms):
                candidate = tuple(selected)
                if score_total > best_score + 1e-12 or (abs(score_total - best_score) <= 1e-12 and candidate < best_pairs):
                    best_score, best_pairs = score_total, candidate
                return
            visit(index + 1, score_total)
            platform = platforms[index]
            for target, value in choices[index]:
                if counts.get(target.entity_id, 0) >= self._capacity_for(target.entity_id):
                    continue
                counts[target.entity_id] = counts.get(target.entity_id, 0) + 1
                selected.append((platform.entity_id, target.entity_id, value))
                visit(index + 1, score_total + value)
                selected.pop()
                counts[target.entity_id] -= 1

        visit(0, 0.0)
        return AssignmentSolution(best_pairs, best_score, exact=True)
