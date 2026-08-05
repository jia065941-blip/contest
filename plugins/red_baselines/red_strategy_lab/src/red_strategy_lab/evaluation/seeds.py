from __future__ import annotations

from typing import Iterable


def split_seeds(seeds: Iterable[int], train_fraction: float = 0.5) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ordered = tuple(sorted(set(seeds)))
    if len(ordered) < 2:
        raise ValueError("at least two distinct seeds are required")
    pivot = max(1, min(len(ordered) - 1, round(len(ordered) * train_fraction)))
    return ordered[:pivot], ordered[pivot:]

