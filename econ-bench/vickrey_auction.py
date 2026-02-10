"""
Vickrey (Second-Price) Auction - Tier 1 Validation Problem

Ground Truth:
- Allocation: Give item to highest bidder
- Payment: Winner pays second-highest bid
- Properties: Truthful (dominant strategy), efficient, individually rational

This is the canonical mechanism design result. Any mechanism design system
should be able to rediscover this.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional

import numpy as np

# Allow running as script from repo root or from econ-bench
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mechanism_base import MechanismDesignProblem, MechanismProperties


@dataclass
class AuctionResult:
    """Result of running an auction mechanism."""

    winner_idx: int  # Which bidder won (-1 if no winner)
    payment: float
    revenue: float
    efficiency: float
    truthful: bool


# Ground truth implementations (used by get_ground_truth and __main__)
def vickrey_allocation(bids: np.ndarray) -> int:
    """Allocate to highest bidder."""
    return int(np.argmax(bids))


def vickrey_payment(bids: np.ndarray, winner: int) -> float:
    """Winner pays second-highest bid."""
    if len(bids) < 2:
        return 0.0
    sorted_bids = np.sort(bids)
    return float(sorted_bids[-2])


# Alternative mechanisms for comparison
def first_price_allocation(bids: np.ndarray) -> int:
    """Same allocation as Vickrey."""
    return int(np.argmax(bids))


def first_price_payment(bids: np.ndarray, winner: int) -> float:
    """Winner pays their own bid (NOT truthful)."""
    return float(bids[winner])


def random_allocation(bids: np.ndarray) -> int:
    """Allocate randomly (inefficient)."""
    return int(np.random.randint(len(bids)))


def random_payment(bids: np.ndarray, winner: int) -> float:
    """Random payment."""
    return float(np.random.uniform(0, np.max(bids)))


# Type for value distribution
ValueDistribution = Literal["uniform", "normal"]


class VickreyAuctionProblem(MechanismDesignProblem):
    """
    Single-item auction problem.

    A mechanism is a dict with:
    - 'allocation_rule': (bids) -> winner_idx
    - 'payment_rule': (bids, winner_idx) -> payment

    Evaluated on: revenue, efficiency, truthfulness, individual rationality.
    """

    name = "single_item_auction"
    difficulty = "tier1"
    NO_ALLOCATION = -1

    def __init__(
        self,
        n_bidders: int = 5,
        value_distribution: ValueDistribution = "uniform",
    ):
        self.n_bidders = n_bidders
        self.value_distribution = value_distribution

    def sample_valuations(self, n_samples: int = 1) -> np.ndarray:
        """Sample bidder valuations from distribution."""
        if self.value_distribution == "uniform":
            return np.random.uniform(0, 100, size=(n_samples, self.n_bidders))
        elif self.value_distribution == "normal":
            return np.abs(
                np.random.normal(50, 20, size=(n_samples, self.n_bidders))
            )
        else:
            raise ValueError(f"Unknown distribution: {self.value_distribution}")

    def sample_instance(self) -> Dict[str, Any]:
        """Sample a problem instance (valuations)."""
        return {"valuations": self.sample_valuations(1)[0]}

    def get_ground_truth(self) -> Dict[str, Any]:
        """Return the known optimal mechanism."""
        return {
            "allocation_rule": vickrey_allocation,
            "payment_rule": vickrey_payment,
            "properties": {
                "truthful": True,
                "efficient": True,
                "individually_rational": True,
                "budget_balanced": False,
            },
        }

    def _mechanism_rules(self, mechanism: Any):
        """Extract allocation_rule and payment_rule from mechanism dict."""
        if isinstance(mechanism, dict):
            return mechanism["allocation_rule"], mechanism["payment_rule"]
        raise TypeError("mechanism must be dict with allocation_rule and payment_rule")

    def _compute_utility(
        self,
        bids: np.ndarray,
        test_bidder: int,
        true_value: float,
        allocation_rule: Callable[[np.ndarray], int],
        payment_rule: Callable[[np.ndarray, int], float],
    ) -> float:
        """Utility of test_bidder when bids are used (true value for payoff)."""
        winner = allocation_rule(bids)
        if winner == test_bidder:
            payment = payment_rule(bids, winner)
            return true_value - payment
        return 0.0

    def _test_truthfulness(
        self,
        allocation_rule: Callable[[np.ndarray], int],
        payment_rule: Callable[[np.ndarray, int], float],
        n_tests: int = 100,
        deviation_strategy: Literal["grid", "random"] = "grid",
        n_deviations_per_test: int = 20,
    ) -> Dict[str, Any]:
        """
        Test if truthful bidding is optimal.

        Returns:
            is_truthful, max_profitable_deviation, fraction_profitable, counterexample
        """
        max_gain = 0.0
        profitable_count = 0
        total_tests = 0
        counterexample: Optional[Dict[str, Any]] = None
        tol = 1e-6

        for _ in range(n_tests):
            valuations = self.sample_valuations(1)[0]
            test_bidder = np.random.randint(self.n_bidders)
            true_value = valuations[test_bidder]
            bids = valuations.copy()

            truthful_utility = self._compute_utility(
                bids, test_bidder, true_value, allocation_rule, payment_rule
            )

            if deviation_strategy == "grid":
                test_bids = np.linspace(0, 100, n_deviations_per_test)
            else:
                test_bids = np.random.uniform(0, 100, n_deviations_per_test)

            for test_bid in test_bids:
                total_tests += 1
                deviated_bids = bids.copy()
                deviated_bids[test_bidder] = float(test_bid)

                deviated_utility = self._compute_utility(
                    deviated_bids,
                    test_bidder,
                    true_value,
                    allocation_rule,
                    payment_rule,
                )

                gain = deviated_utility - truthful_utility
                if gain > tol:
                    profitable_count += 1
                    if gain > max_gain:
                        max_gain = gain
                        counterexample = {
                            "valuations": valuations.copy(),
                            "test_bidder": test_bidder,
                            "true_bid": float(true_value),
                            "profitable_bid": float(test_bid),
                            "gain": float(gain),
                        }

        return {
            "is_truthful": max_gain < tol,
            "max_profitable_deviation": float(max_gain),
            "fraction_profitable": (
                profitable_count / total_tests if total_tests > 0 else 0.0
            ),
            "counterexample": counterexample,
        }

    def evaluate_mechanism(
        self,
        mechanism: Any,
        n_trials: int = 1000,
        *,
        deviation_strategy: Literal["grid", "random"] = "grid",
    ) -> Dict[str, float]:
        """
        Evaluate a mechanism across many sampled valuations.

        mechanism: dict with 'allocation_rule' and 'payment_rule'.

        Returns metrics including revenue, efficiency, truthfulness details,
        individual rationality, and robustness.
        """
        allocation_rule, payment_rule = self._mechanism_rules(mechanism)

        results: Dict[str, list] = {
            "revenue": [],
            "allocated_value": [],
            "max_value": [],
            "payments": [],
            "allocations": [],
        }

        for _ in range(n_trials):
            valuations = self.sample_valuations(1)[0]
            bids = valuations.copy()
            max_val = float(np.max(valuations))
            results["max_value"].append(max_val)

            winner = allocation_rule(bids)

            if winner == self.NO_ALLOCATION:
                results["allocated_value"].append(0.0)
                results["revenue"].append(0.0)
                results["payments"].append(0.0)
                results["allocations"].append(self.NO_ALLOCATION)
            else:
                payment = payment_rule(bids, winner)
                results["allocated_value"].append(float(valuations[winner]))
                results["revenue"].append(payment)
                results["payments"].append(payment)
                results["allocations"].append(winner)

        truthfulness = self._test_truthfulness(
            allocation_rule, payment_rule, deviation_strategy=deviation_strategy
        )

        winner_utilities = [
            results["allocated_value"][i] - results["payments"][i]
            for i in range(len(results["payments"]))
            if results["allocations"][i] != self.NO_ALLOCATION
        ]
        mean_winner_utility = (
            np.mean(winner_utilities) if winner_utilities else 0.0
        )

        mean_allocated = np.mean(results["allocated_value"])
        mean_max = np.mean(results["max_value"])
        efficient_count = sum(
            av == mv
            for av, mv in zip(
                results["allocated_value"], results["max_value"]
            )
        )

        return {
            "mean_revenue": float(np.mean(results["revenue"])),
            "mean_social_welfare": mean_allocated,
            "efficiency_ratio": float(
                mean_allocated / mean_max if mean_max > 0 else 0.0
            ),
            "efficient_allocation_rate": float(
                efficient_count / n_trials if n_trials > 0 else 0.0
            ),
            "is_truthful": truthfulness["is_truthful"],
            "max_profitable_deviation": truthfulness["max_profitable_deviation"],
            "fraction_profitable": truthfulness["fraction_profitable"],
            "mean_winner_utility": float(mean_winner_utility),
            "revenue_std": float(np.std(results["revenue"])),
            "efficiency_std": float(np.std(results["allocated_value"])),
        }

    def verify_properties(self, mechanism: Any) -> Dict[str, bool]:
        """Check theoretical properties (truthful, efficient, IR)."""
        allocation_rule, payment_rule = self._mechanism_rules(mechanism)
        metrics = self.evaluate_mechanism(mechanism, n_trials=500)

        # Spot-check IR on a few instances
        ir_ok = True
        for _ in range(50):
            inst = self.sample_instance()
            if not MechanismProperties.is_individually_rational(
                inst["valuations"],
                allocation_rule,
                payment_rule,
                no_allocation_value=self.NO_ALLOCATION,
            ):
                ir_ok = False
                break

        return {
            "truthful": metrics["is_truthful"],
            "efficient": metrics["efficiency_ratio"] > 0.99,
            "individually_rational": metrics["mean_winner_utility"] >= -0.01
            and ir_ok,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("VICKREY AUCTION - GROUND TRUTH VALIDATION")
    print("=" * 60)

    problem = VickreyAuctionProblem(n_bidders=5)
    ground_truth = problem.get_ground_truth()
    vickrey_mechanism = {
        "allocation_rule": ground_truth["allocation_rule"],
        "payment_rule": ground_truth["payment_rule"],
    }

    print("\n1. Testing Vickrey (Second-Price) Mechanism:")
    print("-" * 60)
    vickrey_results = problem.evaluate_mechanism(vickrey_mechanism, n_trials=1000)
    print(f"Mean Revenue:        ${vickrey_results['mean_revenue']:.2f}")
    print(f"Mean Social Welfare:  ${vickrey_results['mean_social_welfare']:.2f}")
    print(f"Efficiency Ratio:     {vickrey_results['efficiency_ratio']:.3f}")
    print(f"Efficient Alloc Rate: {vickrey_results['efficient_allocation_rate']:.3f}")
    print(
        f"Truthful:            {vickrey_results['is_truthful']} ✓"
        if vickrey_results["is_truthful"]
        else f"Truthful:            {vickrey_results['is_truthful']} ✗"
    )
    print(f"Mean Winner Utility:  ${vickrey_results['mean_winner_utility']:.2f}")

    print("\n2. Testing First-Price Mechanism (for comparison):")
    print("-" * 60)
    first_price_mechanism = {
        "allocation_rule": first_price_allocation,
        "payment_rule": first_price_payment,
    }
    first_price_results = problem.evaluate_mechanism(
        first_price_mechanism, n_trials=1000
    )
    print(f"Mean Revenue:     ${first_price_results['mean_revenue']:.2f}")
    print(f"Mean Social Welfare: ${first_price_results['mean_social_welfare']:.2f}")
    print(
        f"Truthful:         {first_price_results['is_truthful']} (Expected: False)"
    )

    print("\n3. Testing Random Mechanism (baseline):")
    print("-" * 60)
    random_mechanism = {
        "allocation_rule": random_allocation,
        "payment_rule": random_payment,
    }
    random_results = problem.evaluate_mechanism(random_mechanism, n_trials=1000)
    print(f"Mean Revenue:     ${random_results['mean_revenue']:.2f}")
    print(f"Mean Social Welfare: ${random_results['mean_social_welfare']:.2f}")
    print(f"Efficiency Ratio: {random_results['efficiency_ratio']:.3f}")

    print("\n4. Property verification (Vickrey):")
    print("-" * 60)
    props = problem.verify_properties(vickrey_mechanism)
    for k, v in props.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"✓ Vickrey is truthful:  {vickrey_results['is_truthful']}")
    print(
        f"✓ Vickrey is efficient: {vickrey_results['efficiency_ratio'] > 0.99}"
    )
    print(f"✓ First-price NOT truthful: {not first_price_results['is_truthful']}")
    print(
        f"✓ Random is less efficient than Vickrey: {random_results['efficiency_ratio'] < 0.8}"
    )
    print("\nGround truth validated! ✓")
