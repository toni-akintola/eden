"""
Base class for mechanism design problems in the econ-bench.

All verification problems (Vickrey, VCG public good, serial dictatorship, etc.)
conform to this interface so they can be run by a common benchmark runner.
"""

from abc import ABC, abstractmethod

from typing import Any, Callable, Dict, Literal

import numpy as np

tiers: Literal = ["tier1", "tier2", "tier3"]
class MechanismProperties:
    """Test suite for mechanism properties (IC, IR, efficiency, etc.)"""

    @staticmethod
    def is_individually_rational(
        valuations: np.ndarray,
        allocation_rule: Callable[[np.ndarray], Any],
        payment_rule: Callable[[np.ndarray, Any], float],
        no_allocation_value: Any = -1,
    ) -> bool:
        """Winner's utility >= 0 (for single-item: value - payment >= 0)."""
        bids = valuations.copy()
        winner = allocation_rule(bids)
        if winner == no_allocation_value:
            return True
        payment = payment_rule(bids, winner)
        return valuations[winner] >= payment - 1e-9

    @staticmethod
    def is_efficient(
        valuations: np.ndarray,
        allocation_rule: Callable[[np.ndarray], Any],
        no_allocation_value: Any = -1,
    ) -> bool:
        """Does item go to highest valuer? (Assuming truthful bids.)"""
        bids = valuations.copy()
        winner = allocation_rule(bids)
        if winner == no_allocation_value:
            return np.max(valuations) <= 0
        return winner == np.argmax(valuations)


class MechanismDesignProblem(ABC):
    """Base class for all mechanism design problems."""

    @abstractmethod
    def sample_instance(self) -> Dict[str, Any]:
        """Sample a problem instance (valuations, preferences, etc.)."""
        pass

    @abstractmethod
    def evaluate_mechanism(self, mechanism: Any, n_trials: int) -> Dict[str, float]:
        """Evaluate a mechanism and return metrics."""
        pass

    @abstractmethod
    def get_ground_truth(self) -> Callable:
        """Return the known optimal mechanism (if exists)."""
        pass

    @abstractmethod
    def verify_properties(self, mechanism: Any) -> Dict[str, bool]:
        """Check theoretical properties (IC, IR, efficiency, etc.)."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for the problem."""
        pass

    @property
    @abstractmethod
    def difficulty(self) -> Literal:
        """'tier1', 'tier2', or 'tier3'."""
        pass
