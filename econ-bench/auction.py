"""
auction.py

Minimal 2-bidder, 2-item auction environment and evaluator.

A mechanism is defined by:
  - alloc(v1, v2) -> (a1, a2)
      v1, v2 : np.ndarray of shape (2,)  -- reported type vectors for bidder 1 and 2
      a1, a2 : np.ndarray of shape (2,)  -- allocation probabilities for each bidder
                                            a1[j] = prob bidder 1 gets item j
                                            feasibility: a1[j] + a2[j] <= 1 for each j
  - pay(v1, v2) -> (p1, p2)
      p1, p2 : float -- payment from each bidder (must respect budget if constrained)

Evaluation mirrors RegretNet's two reported metrics:
  1. Revenue  : E[p1 + p2]
  2. Regret   : E[max_{v'_i} u_i(v'_i, v_{-i}) - u_i(v_i, v_{-i})]  (ex post, per bidder)

Welfare (sum of allocated values) is also computed for reference but is NOT the primary
fitness signal — revenue minus penalized regret is, matching RegretNet's objective.

Instance parameters (all optional kwargs to AuctionInstance):
  - n_items       : int   (default 2)
  - n_bidders     : int   (default 2)
  - v_ranges      : list of (lo, hi) per item, e.g. [(0,1),(0,1)]
  - correlation   : float rho in [-1, 1]; 0 = independent (default)
  - budgets       : list of per-bidder budget caps, or None = unconstrained (default)

Usage:
  inst = AuctionInstance()
  profiles = inst.sample(n=10_000)
  result = evaluate(mechanism, profiles, n_misreports=500, penalty=1.0)
  print(result)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
TypeVector = np.ndarray  # shape (2,)  -- one bidder's valuations
TypeProfile = Tuple[TypeVector, TypeVector]  # (v1, v2)
AllocPair = Tuple[TypeVector, TypeVector]  # (a1, a2)
PayPair = Tuple[float, float]  # (p1, p2)

Mechanism = Tuple[
    Callable[[TypeVector, TypeVector], AllocPair],
    Callable[[TypeVector, TypeVector], PayPair],
]


# ---------------------------------------------------------------------------
# Instance definition
# ---------------------------------------------------------------------------
@dataclass
class AuctionInstance:
    """
    Defines the distributional parameters of a 2-bidder, 2-item auction instance.

    Parameters
    ----------
    n_items     : number of items (default 2)
    n_bidders   : number of bidders (default 2)
    v_ranges    : value support per item, list of (lo, hi)
    correlation : Pearson rho between a bidder's two item values (same for both bidders)
    budgets     : per-bidder payment caps; None = unconstrained
    """

    n_items: int = 2
    n_bidders: int = 2
    v_ranges: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 1.0), (0.0, 1.0)]
    )
    correlation: float = 0.0
    budgets: Optional[List[float]] = None  # None = unconstrained

    def __post_init__(self):
        assert (
            len(self.v_ranges) == self.n_items
        ), "v_ranges must have one entry per item"
        if self.budgets is not None:
            assert (
                len(self.budgets) == self.n_bidders
            ), "budgets must have one entry per bidder"

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Sample n type profiles.

        Returns
        -------
        profiles : np.ndarray of shape (n, n_bidders, n_items)
            profiles[l, i, j] = bidder i's value for item j in profile l
        """
        if rng is None:
            rng = np.random.default_rng()

        profiles = np.zeros((n, self.n_bidders, self.n_items))

        for i in range(self.n_bidders):
            if self.correlation == 0.0 or self.n_items == 1:
                for j, (lo, hi) in enumerate(self.v_ranges):
                    profiles[:, i, j] = rng.uniform(lo, hi, size=n)
            else:
                # Draw correlated uniforms via Gaussian copula
                rho = np.clip(self.correlation, -1 + 1e-6, 1 - 1e-6)
                cov = np.array([[1.0, rho], [rho, 1.0]])
                z = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=n)
                from scipy.stats import norm

                u = norm.cdf(z)  # shape (n, 2), uniform marginals on [0,1]
                for j, (lo, hi) in enumerate(self.v_ranges):
                    profiles[:, i, j] = lo + u[:, j] * (hi - lo)

        return profiles


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def bidder_utility(
    alloc: TypeVector,
    payment: float,
    true_value: TypeVector,
) -> float:
    """
    Quasi-linear utility: u_i = v_i · a_i - p_i
    (dot product of true values and allocation probabilities, minus payment)
    """
    return float(np.dot(true_value, alloc)) - payment


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass
class EvalResult:
    revenue: float  # E[p1 + p2]
    regret: float  # mean ex post regret across bidders and profiles
    welfare: float  # E[sum_i v_i · a_i]  (reference only)
    fitness: float  # revenue - penalty * regret  (RegretNet-style objective)

    def __repr__(self):
        return (
            f"EvalResult(\n"
            f"  revenue = {self.revenue:.4f}\n"
            f"  regret  = {self.regret:.6f}\n"
            f"  welfare = {self.welfare:.4f}\n"
            f"  fitness = {self.fitness:.4f}\n"
            f")"
        )


def evaluate(
    mechanism: Mechanism,
    profiles: np.ndarray,
    n_misreports: int = 500,
    penalty: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    instance: Optional[AuctionInstance] = None,
) -> EvalResult:
    """
    Evaluate a mechanism on a batch of type profiles.

    Parameters
    ----------
    mechanism    : (alloc_fn, pay_fn) tuple
    profiles     : np.ndarray shape (n, n_bidders, n_items)
    n_misreports : number of random misreport candidates per bidder per profile
                   (replaces gradient ascent; larger = more accurate regret estimate)
    penalty      : lambda in  fitness = revenue - penalty * regret
    rng          : optional numpy Generator for reproducibility
    instance     : AuctionInstance, used to respect budget caps when checking misreports
                   (pass None to skip budget enforcement)

    Returns
    -------
    EvalResult
    """
    if rng is None:
        rng = np.random.default_rng()

    alloc_fn, pay_fn = mechanism
    n, n_bidders, n_items = profiles.shape

    revenues = np.zeros(n)
    welfares = np.zeros(n)
    regrets = np.zeros((n, n_bidders))

    for l in range(n):
        v = profiles[l]  # shape (n_bidders, n_items)
        v1, v2 = v[0], v[1]

        # --- truthful allocation and payment ---
        a1_t, a2_t = alloc_fn(v1, v2)
        p1_t, p2_t = pay_fn(v1, v2)

        # enforce budget caps on truthful payments
        if instance is not None and instance.budgets is not None:
            p1_t = min(p1_t, instance.budgets[0])
            p2_t = min(p2_t, instance.budgets[1])

        revenues[l] = p1_t + p2_t
        welfares[l] = np.dot(v1, a1_t) + np.dot(v2, a2_t)

        # --- regret for bidder 1 ---
        u1_truth = bidder_utility(a1_t, p1_t, v1)
        best_gain_1 = 0.0
        # sample misreports for bidder 1 uniformly from the same support as v1
        # (if instance not provided, assume [0,1]^n_items)
        lo = np.zeros(n_items)
        hi = np.ones(n_items)
        if instance is not None:
            lo = np.array([r[0] for r in instance.v_ranges])
            hi = np.array([r[1] for r in instance.v_ranges])

        misreports_1 = rng.uniform(lo, hi, size=(n_misreports, n_items))
        for mr in misreports_1:
            a1_mr, _ = alloc_fn(mr, v2)
            p1_mr, _ = pay_fn(mr, v2)
            if instance is not None and instance.budgets is not None:
                p1_mr = min(p1_mr, instance.budgets[0])
            u1_mr = bidder_utility(a1_mr, p1_mr, v1)  # true value, misreported bid
            best_gain_1 = max(best_gain_1, u1_mr - u1_truth)
        regrets[l, 0] = best_gain_1

        # --- regret for bidder 2 ---
        u2_truth = bidder_utility(a2_t, p2_t, v2)
        best_gain_2 = 0.0
        misreports_2 = rng.uniform(lo, hi, size=(n_misreports, n_items))
        for mr in misreports_2:
            _, a2_mr = alloc_fn(v1, mr)
            _, p2_mr = pay_fn(v1, mr)
            if instance is not None and instance.budgets is not None:
                p2_mr = min(p2_mr, instance.budgets[1])
            u2_mr = bidder_utility(a2_mr, p2_mr, v2)
            best_gain_2 = max(best_gain_2, u2_mr - u2_truth)
        regrets[l, 1] = best_gain_2

    mean_revenue = float(np.mean(revenues))
    mean_regret = float(np.mean(regrets))
    mean_welfare = float(np.mean(welfares))
    fitness = mean_revenue - penalty * mean_regret

    return EvalResult(
        revenue=mean_revenue,
        regret=mean_regret,
        welfare=mean_welfare,
        fitness=fitness,
    )


# ---------------------------------------------------------------------------
# Example mechanisms (baselines)
# ---------------------------------------------------------------------------


def second_price_bundle(v1: TypeVector, v2: TypeVector) -> Tuple[AllocPair, PayPair]:
    """
    Sell the grand bundle (both items together) via second-price auction.
    Bidder with higher total value wins both items and pays the other's total.
    """
    b1 = float(np.sum(v1))
    b2 = float(np.sum(v2))
    a1 = np.ones(2) if b1 >= b2 else np.zeros(2)
    a2 = np.ones(2) - a1
    p1 = b2 if b1 >= b2 else 0.0
    p2 = b1 if b2 > b1 else 0.0
    return (a1, a2), (p1, p2)


def item_by_item_second_price(
    v1: TypeVector, v2: TypeVector
) -> Tuple[AllocPair, PayPair]:
    """
    Run a separate second-price auction for each item independently.
    """
    a1 = np.zeros(2)
    a2 = np.zeros(2)
    p1 = 0.0
    p2 = 0.0
    for j in range(2):
        if v1[j] >= v2[j]:
            a1[j] = 1.0
            p1 += v2[j]
        else:
            a2[j] = 1.0
            p2 += v1[j]
    return (a1, a2), (p1, p2)


def _wrap_baseline(fn):
    """Wrap a baseline into (alloc_fn, pay_fn) format."""

    def alloc_fn(v1, v2):
        (a1, a2), _ = fn(v1, v2)
        return a1, a2

    def pay_fn(v1, v2):
        _, (p1, p2) = fn(v1, v2)
        return p1, p2

    return alloc_fn, pay_fn


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== 2-bidder, 2-item auction evaluator ===\n")

    inst = AuctionInstance(
        v_ranges=[(0.0, 1.0), (0.0, 1.0)],
        correlation=0.0,
        budgets=None,
    )

    rng = np.random.default_rng(42)
    profiles = inst.sample(n=2_000, rng=rng)

    print("Instance:", inst)
    print(f"Sampled {len(profiles)} profiles.\n")

    bundle_mech = _wrap_baseline(second_price_bundle)
    item_mech = _wrap_baseline(item_by_item_second_price)

    print("--- Grand-bundle second-price ---")
    print(
        evaluate(
            bundle_mech,
            profiles,
            n_misreports=200,
            penalty=1.0,
            rng=np.random.default_rng(0),
            instance=inst,
        )
    )

    print("\n--- Item-by-item second-price ---")
    print(
        evaluate(
            item_mech,
            profiles,
            n_misreports=200,
            penalty=1.0,
            rng=np.random.default_rng(0),
            instance=inst,
        )
    )
