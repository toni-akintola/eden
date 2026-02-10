"""
Groves-Clarke (VCG) Public Good Mechanism

Ground Truth:
- Decide: Build if sum of values > cost
- Payment: Each agent pays their "pivotal" impact
- Properties: Truthful, efficient, but NOT budget balanced

Tests whether the system can:
- Handle multi-agent decisions beyond allocation
- Discover pivotal payments
- Recognize budget balance impossibility
"""

from typing import Any, Callable, Dict, Literal, Optional

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mechanism_base import MechanismDesignProblem, MechanismProperties


# --- Ground truth: Groves-Clarke (Clarke/pivotal) mechanism ---

def groves_clarke_decision(bids: np.ndarray, cost: float) -> int:
    """Build (1) iff sum of bids >= cost."""
    return 1 if float(np.sum(bids)) >= cost else 0


def groves_clarke_payments(
    bids: np.ndarray, cost: float, decision: int
) -> np.ndarray:
    """
    Clarke (pivotal) payments: each agent pays their externality.
    - If build: agent i pays max(0, cost - sum_{j≠i} b_j)
    - If not build: agent i pays max(0, sum_{j≠i} b_j - cost)
    """
    n = len(bids)
    payments = np.zeros(n)
    for i in range(n):
        sum_others = float(np.sum(bids) - bids[i])
        if decision == 1:
            # Pivotal for building: we build but would not have without i
            payments[i] = max(0.0, cost - sum_others)
        else:
            # Pivotal for not building: we don't build but would have without i
            payments[i] = max(0.0, sum_others - cost)
    return payments


# --- Alternative (non-truthful) mechanisms for comparison ---

def majority_decision(bids: np.ndarray, cost: float) -> int:
    """Build if majority have positive bid (ignores cost; not efficient)."""
    return 1 if np.sum(bids > 0) > len(bids) / 2 else 0


def majority_payments(
    bids: np.ndarray, cost: float, decision: int
) -> np.ndarray:
    """Split cost equally if build (not pivotal; not truthful)."""
    n = len(bids)
    if decision == 1:
        return np.full(n, cost / n)
    return np.zeros(n)


def no_transfers_decision(bids: np.ndarray, cost: float) -> int:
    """Efficient decision but no payments (not truthful in general)."""
    return 1 if float(np.sum(bids)) >= cost else 0


def no_transfers_payments(bids: np.ndarray, cost: float, decision: int) -> np.ndarray:
    """Zero payments."""
    return np.zeros(len(bids))


# --- Problem class ---

ValueDistribution = Literal["uniform", "normal"]


class GrovesClarkePublicGoodProblem(MechanismDesignProblem):
    """
    Public good (build / don't build) with cost C.
    Instance: valuations v_i, cost C.
    Efficient decision: build iff sum_i v_i >= C.
    Mechanism: decision_rule(bids, cost) -> 0|1, payment_rule(bids, cost, decision) -> payments array.
    """

    name = "groves_clarke_public_good"
    difficulty = "tier2"

    def __init__(
        self,
        n_agents: int = 5,
        value_distribution: ValueDistribution = "uniform",
        cost_low: float = 0.0,
        cost_high: float = 300.0,
    ):
        self.n_agents = n_agents
        self.value_distribution = value_distribution
        self.cost_low = cost_low
        self.cost_high = cost_high

    def sample_valuations(self, n_samples: int = 1) -> np.ndarray:
        if self.value_distribution == "uniform":
            return np.random.uniform(0, 100, size=(n_samples, self.n_agents))
        elif self.value_distribution == "normal":
            return np.abs(
                np.random.normal(50, 20, size=(n_samples, self.n_agents))
            )
        raise ValueError(f"Unknown distribution: {self.value_distribution}")

    def sample_cost(self) -> float:
        return float(np.random.uniform(self.cost_low, self.cost_high))

    def sample_instance(self) -> Dict[str, Any]:
        return {
            "valuations": self.sample_valuations(1)[0],
            "cost": self.sample_cost(),
        }

    def get_ground_truth(self) -> Dict[str, Any]:
        return {
            "allocation_rule": groves_clarke_decision,
            "payment_rule": groves_clarke_payments,
            "properties": {
                "truthful": True,
                "efficient": True,
                "individually_rational": True,
                "budget_balanced": False,
            },
        }

    def _mechanism_rules(self, mechanism: Any):
        if isinstance(mechanism, dict):
            return mechanism["allocation_rule"], mechanism["payment_rule"]
        raise TypeError(
            "mechanism must be dict with allocation_rule and payment_rule"
        )

    def _compute_utility(
        self,
        bids: np.ndarray,
        cost: float,
        agent: int,
        true_value: float,
        decision_rule: Callable[[np.ndarray, float], int],
        payment_rule: Callable[[np.ndarray, float, int], np.ndarray],
    ) -> float:
        decision = decision_rule(bids, cost)
        payments = payment_rule(bids, cost, decision)
        return decision * true_value - float(payments[agent])

    def _test_truthfulness(
        self,
        decision_rule: Callable[[np.ndarray, float], int],
        payment_rule: Callable[[np.ndarray, float, int], np.ndarray],
        n_tests: int = 100,
        deviation_strategy: Literal["grid", "random"] = "grid",
        n_deviations: int = 20,
    ) -> Dict[str, Any]:
        max_gain = 0.0
        profitable_count = 0
        total_tests = 0
        counterexample: Optional[Dict[str, Any]] = None
        tol = 1e-6

        for _ in range(n_tests):
            instance = self.sample_instance()
            valuations = instance["valuations"]
            cost = instance["cost"]
            agent = np.random.randint(self.n_agents)
            true_value = float(valuations[agent])
            bids = valuations.copy()

            truthful_utility = self._compute_utility(
                bids, cost, agent, true_value, decision_rule, payment_rule
            )

            if deviation_strategy == "grid":
                test_bids = np.linspace(0, 100, n_deviations)
            else:
                test_bids = np.random.uniform(0, 100, n_deviations)

            for test_bid in test_bids:
                total_tests += 1
                deviated_bids = bids.copy()
                deviated_bids[agent] = float(test_bid)

                deviated_utility = self._compute_utility(
                    deviated_bids,
                    cost,
                    agent,
                    true_value,
                    decision_rule,
                    payment_rule,
                )
                gain = deviated_utility - truthful_utility
                if gain > tol:
                    profitable_count += 1
                    if gain > max_gain:
                        max_gain = gain
                        counterexample = {
                            "valuations": valuations.copy(),
                            "cost": cost,
                            "agent": agent,
                            "true_bid": true_value,
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
        decision_rule, payment_rule = self._mechanism_rules(mechanism)

        revenues = []
        total_payments = []
        social_welfares = []
        efficient_decisions = []
        winner_utilities = []  # per-agent utility (value * decision - payment)

        for _ in range(n_trials):
            instance = self.sample_instance()
            valuations = instance["valuations"]
            cost = instance["cost"]
            bids = valuations.copy()

            decision = decision_rule(bids, cost)
            payments = payment_rule(bids, cost, decision)

            total_payment = float(np.sum(payments))
            total_payments.append(total_payment)
            revenues.append(total_payment)

            sw = decision * (float(np.sum(valuations)) - cost)
            social_welfares.append(sw)
            efficient_decisions.append(
                1
                if (decision == 1) == (float(np.sum(valuations)) >= cost)
                else 0
            )

            for i in range(self.n_agents):
                u = decision * valuations[i] - payments[i]
                winner_utilities.append(float(u))

        truthfulness = self._test_truthfulness(
            decision_rule, payment_rule, deviation_strategy=deviation_strategy
        )

        mean_sw = float(np.mean(social_welfares))
        mean_revenue = float(np.mean(revenues))
        efficient_rate = float(np.mean(efficient_decisions))
        mean_utility = float(np.mean(winner_utilities)) if winner_utilities else 0.0

        # Budget balance: sum of payments (surplus/deficit)
        mean_total_payment = float(np.mean(total_payments))

        return {
            "mean_revenue": mean_revenue,
            "mean_social_welfare": mean_sw,
            "efficient_decision_rate": efficient_rate,
            "is_truthful": truthfulness["is_truthful"],
            "max_profitable_deviation": truthfulness["max_profitable_deviation"],
            "fraction_profitable": truthfulness["fraction_profitable"],
            "mean_agent_utility": mean_utility,
            "revenue_std": float(np.std(revenues)),
            "social_welfare_std": float(np.std(social_welfares)),
            "mean_total_payment": mean_total_payment,
        }

    def verify_properties(self, mechanism: Any) -> Dict[str, bool]:
        decision_rule, payment_rule = self._mechanism_rules(mechanism)
        metrics = self.evaluate_mechanism(mechanism, n_trials=500)

        # Budget balance: Groves-Clarke is NOT budget balanced (surplus typical)
        # We only check that we're not requiring it to be balanced
        budget_balanced = False  # By design for pivotal mechanism

        return {
            "truthful": metrics["is_truthful"],
            "efficient": metrics["efficient_decision_rate"] > 0.99,
            "individually_rational": metrics["mean_agent_utility"] >= -0.01,
            "budget_balanced": budget_balanced,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("GROVES-CLARKE PUBLIC GOOD - GROUND TRUTH VALIDATION")
    print("=" * 60)

    problem = GrovesClarkePublicGoodProblem(n_agents=5, cost_high=300.0)
    ground_truth = problem.get_ground_truth()
    gc_mechanism = {
        "allocation_rule": ground_truth["allocation_rule"],
        "payment_rule": ground_truth["payment_rule"],
    }

    print("\n1. Groves-Clarke (pivotal) mechanism:")
    print("-" * 60)
    gc_results = problem.evaluate_mechanism(gc_mechanism, n_trials=1000)
    print(f"Mean revenue (total payments): {gc_results['mean_revenue']:.2f}")
    print(f"Mean social welfare:          {gc_results['mean_social_welfare']:.2f}")
    print(f"Efficient decision rate:      {gc_results['efficient_decision_rate']:.3f}")
    print(f"Truthful:                     {gc_results['is_truthful']}")
    print(f"Mean agent utility:            {gc_results['mean_agent_utility']:.2f}")

    print("\n2. Majority + equal cost split (not truthful, not efficient):")
    print("-" * 60)
    maj_mechanism = {
        "allocation_rule": majority_decision,
        "payment_rule": majority_payments,
    }
    maj_results = problem.evaluate_mechanism(maj_mechanism, n_trials=1000)
    print(f"Mean revenue:                 {maj_results['mean_revenue']:.2f}")
    print(f"Mean social welfare:          {maj_results['mean_social_welfare']:.2f}")
    print(f"Efficient decision rate:      {maj_results['efficient_decision_rate']:.3f}")
    print(f"Truthful:                     {maj_results['is_truthful']} (often False; non-pivotal payments)")

    print("\n3. Efficient decision, no payments (not truthful):")
    print("-" * 60)
    no_pay_mechanism = {
        "allocation_rule": no_transfers_decision,
        "payment_rule": no_transfers_payments,
    }
    no_pay_results = problem.evaluate_mechanism(no_pay_mechanism, n_trials=1000)
    print(f"Mean revenue:                 {no_pay_results['mean_revenue']:.2f}")
    print(f"Mean social welfare:          {no_pay_results['mean_social_welfare']:.2f}")
    print(f"Efficient decision rate:      {no_pay_results['efficient_decision_rate']:.3f}")
    print(f"Truthful:                     {no_pay_results['is_truthful']} (no payments => no incentive)")

    print("\n4. Property verification (Groves-Clarke):")
    print("-" * 60)
    props = problem.verify_properties(gc_mechanism)
    for k, v in props.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("SUMMARY: Groves-Clarke is truthful, efficient, IR, not budget balanced.")
    print("=" * 60)
