# EDEN Auction Benchmark

A benchmark of analytically intractable mechanism design problems for evaluating
LLM-driven evolutionary search (EDEN) against neural approaches (RegretNet).

---

## Background

### EDEN

EDEN (*Economic Design Engine*) is an LLM-driven evolutionary search process for
mechanism design. An evolutionary population of candidate mechanisms ("organisms")
is maintained across generations. Each organism is a complete mechanism
specification — represented as interpretable symbolic functions (Python lambdas) —
and is evaluated by running a simulation and computing a welfare score. GPT-4/5 acts
as the mutation operator: given a parent organism and high-performing "inspiration"
organisms, the LLM generates a child with modified rules. Selection pressure, elitism,
crossover, and adaptive mutation strength guide the search toward high-fitness regions.

EDEN was validated on the **Che–Tercieux queue design problem**, where it achieved
mean best fitness of 14.00 ± 0.02 across three independent runs (theoretical optimum:
W* = 14.0), independently rediscovering the NO INFORMATION policy predicted by theory.

### RegretNet

RegretNet (Dütting et al., 2019) is the leading neural-network-based approach to
automated auction design. It represents allocation and payment rules as neural
networks and trains them to maximize revenue subject to approximate dominant-strategy
incentive compatibility (DSIC), enforced via an augmented Lagrangian penalty on
ex post regret. RegretNet has been applied to multi-bidder, multi-item settings where
no closed-form optimal mechanism is known, producing the best available computational
conjectures about optimal auction structure.

**Key limitation:** RegretNet's solutions are neural network weights — not
human-readable. The mechanisms it discovers cannot be directly inspected or reasoned
about by economists.

---

## Motivation for This Benchmark

EDEN's validated contribution — rediscovering known results via evolutionary search —
establishes *that the method works*. The more interesting claim is that EDEN can produce
useful conjectures on problems where theory has not delivered. A benchmark of genuinely
hard, analytically intractable problems would shift EDEN's contribution from
"rediscovering known results" to "generating novel, interpretable conjectures."

The ideal benchmark problem satisfies three criteria:

1. **Analytically intractable** — no known closed-form optimal mechanism.
2. **Low training leakage risk** — the solution space is parametric and combinatorial
   enough that the LLM cannot retrieve the answer from pretraining, even if it has
   seen related work.
3. **Easily evaluable** — a computable scalar objective function (fitness) exists.

The **2-bidder, 2-item auction with budget-constrained bidders** satisfies all three
criteria and is the focus of this benchmark.

---

## Problem: Multi-Item Auction with Budget Constraints

### Why this problem?

The single-item optimal auction is fully characterized by Myerson (1981). The
multi-item case — even with just 2 bidders and 2 items — remains **open**: no general
characterization of the revenue- or welfare-optimal mechanism is known. RegretNet has
been applied to the unconstrained 2×2 case, producing strong baselines. Adding
**budget constraints** (each bidder has a hard cap on what they can pay) pushes the
problem further into open territory:

- Budget constraints break the standard Myerson-style analytical toolkit by coupling
  the payment rule and allocation rule in ways that don't arise in the unconstrained
  case.
- Che and Gale (1998) showed that with budgets, randomized mechanisms can strictly
  dominate deterministic ones — a counterintuitive finding not well understood in the
  multi-item setting.
- **RegretNet has not been applied to budget-constrained settings.** Any systematic
  computational exploration is novel.

### EDEN's comparative advantage

RegretNet produces neural network weights. EDEN produces interpretable symbolic
functions. In a setting with no known theoretical benchmark, interpretability matters:
a mechanism you can read and reason about is a scientific contribution in itself. If
EDEN discovers an allocation rule with a clear structural property — e.g., "always
allocate the bundle to the bidder with the highest value-to-budget ratio when budgets
are binding" — that is a more useful conjecture than a black-box network achieving
slightly higher revenue.

---

## Problem Specification

### Setting

- **n_bidders = 2**, **n_items = 2**
- Each bidder i has a **type vector** `v_i = (v_i1, v_i2)` where `v_ij` is their
  private value for item j, drawn i.i.d. from a distribution parameterized by the
  instance.
- **Additive valuations**: bidder i's value for a bundle S is `Σ_{j∈S} v_ij`.
- Optional **budget cap** `B_i`: bidder i's payment cannot exceed `B_i`.

### Instance parameters

| Parameter | Description | Baseline value |
|---|---|---|
| `v_ranges` | Support `[lo, hi]` per item | `[(0,1), (0,1)]` |
| `correlation` | Pearson ρ between a bidder's two item values | `0.0` |
| `budgets` | Per-bidder payment caps | `None` (unconstrained) |

Varying these parameters generates a family of distinct instances. Note that uniform
rescaling of a single item's support does not change the problem structure — what
matters is the **ratio** of scales across items (asymmetry) and the correlation
structure within a bidder's type.

### Mechanism

A mechanism is a pair of functions:

```
alloc(v1, v2) -> (a1, a2)
```
- `v1, v2`: type vectors for bidder 1 and 2 (shape `(2,)`)
- `a1, a2`: allocation probability vectors (shape `(2,)`)
- Feasibility constraint: `a1[j] + a2[j] <= 1` for each item j

```
pay(v1, v2) -> (p1, p2)
```
- `p1, p2`: payments from each bidder (floats)
- Budget constraint: `p_i <= B_i` if budgets are set

---

## Evaluation (mirroring RegretNet)

### Quantities computed

Given a mechanism `(alloc_fn, pay_fn)` and a sample of L type profiles:

**Revenue** — average total payment:
```
rev = (1/L) Σ_l [p1(v^l) + p2(v^l)]
```

**Ex post regret** — average maximum gain from misreporting, per bidder:
```
rgt_i(v) = max_{v'_i} [ u_i(alloc(v'_i, v_{-i}), pay(v'_i, v_{-i}), v_i)
                       - u_i(alloc(v_i,  v_{-i}), pay(v_i,  v_{-i}), v_i) ]
rgt = (1 / n·L) Σ_i Σ_l rgt_i(v^l)
```
where utility is quasi-linear: `u_i = v_i · a_i - p_i`.

RegretNet computes this inner max via gradient ascent (2000 steps, 1000 random
initializations). We use **random sampling over misreport candidates** instead, which
is exact up to the number of samples and avoids local-optima issues inherent in
gradient ascent over a non-differentiable symbolic function.

**Social welfare** — average total allocated value (reported for reference):
```
welfare = (1/L) Σ_l [v1^l · a1(v^l) + v2^l · a2(v^l)]
```

**Fitness** — RegretNet-style objective:
```
fitness = revenue - λ · regret
```
where λ is a fixed penalty weight (default: 1.0). This is a simplified version of
RegretNet's augmented Lagrangian, with the Lagrange multiplier fixed rather than
adaptive, making the revenue-regret tradeoff explicit and interpretable.

### Why revenue over welfare?

RegretNet optimizes revenue, so we use revenue as the primary objective to make the
comparison apples-to-apples. Welfare is reported as a secondary diagnostic. The two
diverge sharply under budget constraints — a useful feature, since tracking both
reveals whether evolved mechanisms achieve efficiency gains at the cost of revenue or
vice versa.

---

## Baseline Mechanisms

Two analytical baselines are implemented in `auction.py`:

**Grand-bundle second-price**: sell both items together to the highest total bidder at
the second-highest total bid. DSIC by construction (regret = 0). Higher revenue than
item-by-item in expectation; lower welfare.

**Item-by-item second-price**: run a separate second-price auction for each item
independently. Also DSIC. Higher welfare (more efficient allocation); lower revenue.

Sample output on 2,000 profiles (Uniform[0,1]², unconstrained, seed 42):

```
--- Grand-bundle second-price ---
revenue = 0.7577   regret = 0.000000   welfare = 1.2256   fitness = 0.7577

--- Item-by-item second-price ---
revenue = 0.6541   regret = 0.000000   welfare = 1.3292   fitness = 0.6541
```

These match theoretical expectations: bundling raises revenue at the cost of welfare.

---

## Repository Structure

```
auction.py       Core auction environment and evaluator
README.md        This file
```

Planned additions:
- `mutator.py`   LLM-based mechanism mutation (EDEN evolutionary loop)
- `evolve.py`    Main evolutionary search loop
- `instances.py` Parameterized instance suite for benchmarking

---

## Comparison Plan

1. **Unconstrained baseline**: run RegretNet (via `dimonenka/optimaler` PyTorch
   implementation) on `additive_2x2_uniform` to obtain revenue and regret figures.
   Run EDEN on the same instance. Compare fitness and inspect evolved functional forms.

2. **Budget-constrained sweep**: vary budget cap B from "never binds" (B = 2.0) to
   "always binds" (B = 0.3) across a grid of instances. RegretNet has no results here.
   EDEN produces the first systematic characterization of how near-optimal mechanisms
   change as budgets tighten.

3. **Asymmetric instances**: vary item-value asymmetry (e.g., `v_ranges = [(0,1),(0,3)]`)
   and correlation ρ to test robustness. Compare structural properties of evolved
   mechanisms across instance families.

---

## References

- Che, Y.-K. and Gale, I. (1998). Standard Auctions with Financially Constrained Bidders.
- Che, Y.-K. and Tercieux, O. (2023). Optimal Queue Design.
- Dütting, P., Feng, Z., Narasimhan, H., Parkes, D. C., and Ravindranath, S. S. (2019).
  Optimal Auctions through Deep Learning.
- Manelli, A. and Vincent, D. (2007). Multidimensional Mechanism Design: Revenue
  Maximization and the Multiple-Good Monopoly.
- Myerson, R. (1981). Optimal Auction Design.
- Novikov, A. et al. (2025). AlphaEvolve: A Coding Agent for Scientific and
  Algorithmic Discovery.