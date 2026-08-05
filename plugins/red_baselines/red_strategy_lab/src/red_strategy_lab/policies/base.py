from __future__ import annotations

from abc import ABC, abstractmethod

from ..domain import Decision, Observation


class BaselinePolicy(ABC):
    @abstractmethod
    def decide(self, observation: Observation) -> Decision:
        """Return a deterministic decision for one observation."""

