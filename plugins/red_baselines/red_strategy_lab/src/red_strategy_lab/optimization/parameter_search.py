"""Deterministic parameter selection for B3."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Callable, Iterable, Mapping


@dataclass(frozen=True)
class SearchResult:
    parameters: Mapping[str, float]
    mean_score: float
    cvar_loss: float
    objective: float


class ParameterSearch:
    def __init__(self, risk_weight: float = 0.0, tail_fraction: float = 0.2) -> None:
        if not 0 < tail_fraction <= 1:
            raise ValueError("tail_fraction must be in (0, 1]")
        self.risk_weight = risk_weight
        self.tail_fraction = tail_fraction

    def select(self, candidates: Iterable[Mapping[str, float]], evaluate: Callable[[Mapping[str, float]], Iterable[float]]) -> SearchResult:
        best: SearchResult | None = None
        for candidate in candidates:
            scores = tuple(float(score) for score in evaluate(candidate))
            if not scores:
                raise ValueError("each candidate needs at least one score")
            mean_score = sum(scores) / len(scores)
            worst_count = ceil(len(scores) * self.tail_fraction)
            cvar_loss = sum(sorted((-score for score in scores), reverse=True)[:worst_count]) / worst_count
            result = SearchResult(dict(candidate), mean_score, cvar_loss, mean_score - self.risk_weight * cvar_loss)
            if best is None or result.objective > best.objective or (result.objective == best.objective and tuple(sorted(result.parameters.items())) < tuple(sorted(best.parameters.items()))):
                best = result
        if best is None:
            raise ValueError("at least one candidate is required")
        return best

