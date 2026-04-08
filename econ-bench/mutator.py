import logging
import os
import re
import random
import numpy as np
from typing import List, Optional, Dict, Any
from langfuse import observe
from langfuse.openai import openai
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Organism
# ---------------------------------------------------------------------------


@dataclass
class Organism:
    """
    A candidate auction mechanism, defined by two lambda source strings.

    Contract:
        alloc(v1, v2) -> (a1, a2)
            v1, v2 : np.ndarray shape (2,)  -- FULL type vectors for both bidders
            a1, a2 : np.ndarray shape (2,)  -- allocation probabilities per item
            feasibility: a1[j] + a2[j] <= 1 for j in {0, 1}

        pay(v1, v2) -> (p1, p2)
            p1, p2 : float  -- non-negative payments

    Both lambdas receive the full (v1, v2) input so they can condition on
    cross-item and cross-bidder information — this is essential for good mechanisms.
    """

    alloc_src: str
    pay_src: str
    fitness: float = -np.inf
    revenue: float = 0.0
    regret: float = 0.0
    welfare: float = 0.0
    generation: int = 0
    parent_id: Optional[str] = None
    id: str = field(default_factory=lambda: f"{random.randint(0, 999999):06d}")
    mutation_reasoning: str = ""

    def to_callables(self):
        """Compile source strings into callable functions."""
        alloc_fn = eval(self.alloc_src, {"np": np})
        pay_fn = eval(self.pay_src, {"np": np})
        return alloc_fn, pay_fn

    def summary(self) -> str:
        return (
            f"[{self.id}] gen={self.generation}  "
            f"fitness={self.fitness:.4f}  rev={self.revenue:.4f}  rgt={self.regret:.6f}\n"
            f"  alloc: {self.alloc_src}\n"
            f"  pay:   {self.pay_src}"
        )


# ---------------------------------------------------------------------------
# Seed organisms — used to initialise generation 0
# ---------------------------------------------------------------------------

SEED_ORGANISMS = [
    Organism(
        alloc_src=(
            "lambda v1, v2: ("
            "np.array([1.0, 1.0]) if np.sum(v1) >= np.sum(v2) else np.array([0.0, 0.0]), "
            "np.array([0.0, 0.0]) if np.sum(v1) >= np.sum(v2) else np.array([1.0, 1.0])"
            ")"
        ),
        pay_src=(
            "lambda v1, v2: ("
            "float(np.sum(v2)) if np.sum(v1) >= np.sum(v2) else 0.0, "
            "float(np.sum(v1)) if np.sum(v2) > np.sum(v1) else 0.0"
            ")"
        ),
        mutation_reasoning="Grand-bundle second-price auction (seed baseline)",
    ),
    Organism(
        alloc_src=(
            "lambda v1, v2: ("
            "np.array([float(v1[0] >= v2[0]), float(v1[1] >= v2[1])]), "
            "np.array([float(v2[0] > v1[0]),  float(v2[1] > v1[1])])"
            ")"
        ),
        pay_src=(
            "lambda v1, v2: ("
            "float(v2[0] * float(v1[0] >= v2[0])) + float(v2[1] * float(v1[1] >= v2[1])), "
            "float(v1[0] * float(v2[0] > v1[0]))  + float(v1[1] * float(v2[1] > v1[1]))"
            ")"
        ),
        mutation_reasoning="Item-by-item second-price auction (seed baseline)",
    ),
]


# ---------------------------------------------------------------------------
# MutationResponse — typed output format for the LLM
# ---------------------------------------------------------------------------


@dataclass
class MutationResponse:
    alloc_src: str  # lambda v1, v2: ...
    pay_src: str  # lambda v1, v2: ...
    mutation_reasoning: str  # brief explanation of what was changed and why


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------


def crossover(parent1: Organism, parent2: Organism) -> Organism:
    """
    Uniform crossover between two parent organisms.

    Each component (alloc_src, pay_src) is independently drawn from one
    of the two parents at random.
    """
    alloc_source = random.choice([parent1, parent2])
    pay_source = random.choice([parent1, parent2])

    reasoning = (
        f"Crossover: alloc from [{alloc_source.id}], " f"pay from [{pay_source.id}]"
    )

    child_gen = max(parent1.generation, parent2.generation) + 1
    logger.info(
        "crossover parents=%s,%s -> alloc_from=%s pay_from=%s child_gen=%s",
        parent1.id,
        parent2.id,
        alloc_source.id,
        pay_source.id,
        child_gen,
    )

    return Organism(
        alloc_src=alloc_source.alloc_src,
        pay_src=pay_source.pay_src,
        generation=child_gen,
        parent_id=parent1.id,
        mutation_reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Adaptive mutation controller — identical logic to Che-Tercieux version
# ---------------------------------------------------------------------------


class AdaptiveMutationController:
    """
    Controls mutation strength based on evolutionary progress.
    Increases exploration when stuck, decreases when improving.
    """

    def __init__(
        self,
        initial_temperature: float = 1.0,
        min_temperature: float = 0.3,
        max_temperature: float = 3.0,
        cooldown_rate: float = 0.95,
        heatup_rate: float = 1.1,
        stagnation_threshold: int = 5,
    ):
        self.temperature = initial_temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.cooldown_rate = cooldown_rate
        self.heatup_rate = heatup_rate
        self.stagnation_threshold = stagnation_threshold

        self.best_fitness_history: List[float] = []
        self.steps_without_improvement: int = 0
        self.last_best_fitness: Optional[float] = None

    def update(self, current_best_fitness: float) -> None:
        """Update temperature based on whether fitness improved."""
        self.best_fitness_history.append(current_best_fitness)

        if self.last_best_fitness is None:
            self.last_best_fitness = current_best_fitness
            return

        if current_best_fitness > self.last_best_fitness + 0.01:
            self.steps_without_improvement = 0
            old_temp = self.temperature
            self.temperature = max(
                self.min_temperature, self.temperature * self.cooldown_rate
            )
            logger.debug(
                "fitness improved: best %.4g -> %.4g; temp %.4g -> %.4g",
                self.last_best_fitness,
                current_best_fitness,
                old_temp,
                self.temperature,
            )
            self.last_best_fitness = current_best_fitness
        else:
            self.steps_without_improvement += 1
            if self.steps_without_improvement >= self.stagnation_threshold:
                old_temp = self.temperature
                self.temperature = min(
                    self.max_temperature, self.temperature * self.heatup_rate
                )
                logger.debug(
                    "stagnation heatup: steps_no_improve=%d temp %.4g -> %.4g",
                    self.steps_without_improvement,
                    old_temp,
                    self.temperature,
                )

    def get_mutation_strength(self) -> str:
        if self.temperature < 0.6:
            return "small"
        elif self.temperature < 1.2:
            return "medium"
        elif self.temperature < 2.0:
            return "large"
        else:
            return "radical"

    def should_do_random_restart(self) -> bool:
        base_chance = 0.05
        stagnation_bonus = min(0.2, self.steps_without_improvement * 0.02)
        return random.random() < (base_chance + stagnation_bonus)

    def get_crossover_probability(self) -> float:
        return min(0.6, 0.3 * self.temperature)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert mechanism designer working on a 2-bidder, 2-item auction.

SETTING:
- Two bidders, each with private values for two items drawn from Uniform[0, 1].
- Bidder i's type is a vector v_i = [v_i1, v_i2] where v_ij is their value for item j.
- Valuations are additive: bidder i values a bundle at the sum of values for items they receive.
- A mechanism has two rules:
    alloc(v1, v2) -> (a1, a2)
        a1, a2 are np.ndarray of shape (2,) giving allocation probabilities per item.
        Feasibility: a1[j] + a2[j] <= 1 for each item j.
    pay(v1, v2) -> (p1, p2)
        p1, p2 are non-negative floats — what each bidder pays.

OBJECTIVE:
Maximise fitness = revenue - 1.0 * regret, where:
  revenue = E[p1 + p2]
  regret  = mean over bidders and profiles of max_{v'_i} [u_i(v'_i, v_{-i}) - u_i(v_i, v_{-i})]
  u_i     = dot(v_i, a_i) - p_i   (quasi-linear utility at TRUE values)

High revenue is good. Near-zero regret means no bidder can gain by misreporting.

LAMBDA CONSTRAINTS:
- Both lambdas must be valid single-line Python expressions.
- numpy is available as np. No other imports.
- No def, return, or multi-line constructs.
- alloc must return a tuple of two np.ndarray of shape (2,).
- pay must return a tuple of two floats.
- Both lambdas receive the FULL (v1, v2) input — use cross-item and cross-bidder info.

OUTPUT FORMAT (respond with ONLY these two lines, nothing else):
ALLOC: lambda v1, v2: <expression>
PAY: lambda v1, v2: <expression>
REASONING: <one sentence explaining the key change and economic intuition>
"""


def build_mutation_prompt(
    parent: Organism,
    inspirations: List[Organism],
    mutation_strength: str,
) -> str:
    lines = []

    lines.append("PARENT ORGANISM (your starting point — mutate this):")
    lines.append(f"  fitness  = {parent.fitness:.4f}")
    lines.append(f"  revenue  = {parent.revenue:.4f}")
    lines.append(f"  regret   = {parent.regret:.6f}")
    lines.append(f"  welfare  = {parent.welfare:.4f}")
    lines.append(f"  alloc:   {parent.alloc_src}")
    lines.append(f"  pay:     {parent.pay_src}")
    if parent.mutation_reasoning:
        lines.append(f"  previous reasoning: {parent.mutation_reasoning}")
    lines.append("")

    if inspirations:
        lines.append(
            "ELITE INSPIRATION ORGANISMS (draw ideas from these high-fitness mechanisms):"
        )
        for i, org in enumerate(inspirations):
            lines.append(
                f"  [{i+1}] fitness={org.fitness:.4f}  "
                f"rev={org.revenue:.4f}  rgt={org.regret:.6f}"
            )
            lines.append(f"       alloc: {org.alloc_src}")
            lines.append(f"       pay:   {org.pay_src}")
        lines.append("")

    lines.append(_get_strength_guidance(mutation_strength))
    lines.append("")
    lines.append("Produce the mutated organism now.")

    return "\n".join(lines)


def _get_strength_guidance(strength: str) -> str:
    if strength == "small":
        return (
            "MUTATION STRENGTH: SMALL (fine-tuning mode)\n"
            "- Adjust ONE constant, threshold, or scaling factor by a small amount.\n"
            "- Do NOT restructure the allocation or payment logic.\n"
            "- Example: change a threshold from 0.5 to 0.45, or scale payments by 0.95."
        )
    elif strength == "medium":
        return (
            "MUTATION STRENGTH: MEDIUM (balanced exploration)\n"
            "- Adjust 1-2 parameters or introduce a modest structural change.\n"
            "- May modify how items are allocated relative to each other.\n"
            "- Example: add a cross-item term to the allocation rule, or adjust payment formula."
        )
    elif strength == "large":
        return (
            "MUTATION STRENGTH: LARGE (exploration mode)\n"
            "- Make a significant structural change — try a new allocation principle.\n"
            "- Consider threshold-based bundling, value-ratio conditions, or mixed strategies.\n"
            "- Example: switch from sum-based to ratio-based allocation, or add randomisation."
        )
    else:  # radical
        return (
            "MUTATION STRENGTH: RADICAL (breakthrough mode)\n"
            "- Completely rethink the mechanism. Ignore the parent structure.\n"
            "- Try unconventional approaches: asymmetric treatment, probabilistic rules,\n"
            "  multi-threshold allocation, or payment rules that depend on value differences.\n"
            "- Must change BOTH alloc and pay significantly."
        )


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def _parse_response(text: str) -> MutationResponse:
    """Parse ALLOC / PAY / REASONING lines from LLM output."""
    alloc_match = re.search(
        r"ALLOC:\s*(lambda\s+v1\s*,\s*v2\s*:.+?)(?=\nPAY:|\Z)", text, re.DOTALL
    )
    pay_match = re.search(
        r"PAY:\s*(lambda\s+v1\s*,\s*v2\s*:.+?)(?=\nREASONING:|\Z)", text, re.DOTALL
    )
    reasoning_match = re.search(r"REASONING:\s*(.+)", text)

    if not alloc_match or not pay_match:
        raise ValueError(f"Could not parse ALLOC/PAY from LLM response:\n{text}")

    return MutationResponse(
        alloc_src=alloc_match.group(1).strip().replace("\n", " "),
        pay_src=pay_match.group(1).strip().replace("\n", " "),
        mutation_reasoning=reasoning_match.group(1).strip() if reasoning_match else "",
    )


# ---------------------------------------------------------------------------
# Validation — compile and spot-check before accepting
# ---------------------------------------------------------------------------


def _validate_organism(alloc_src: str, pay_src: str) -> bool:
    """
    Return True if the lambdas compile and produce valid outputs on a
    small set of test profiles. Does not raise — just returns False on failure.
    """
    try:
        alloc_fn = eval(alloc_src, {"np": np})
        pay_fn = eval(pay_src, {"np": np})
    except Exception:
        return False

    test_profiles = [
        (np.array([0.9, 0.1]), np.array([0.2, 0.8])),
        (np.array([0.5, 0.5]), np.array([0.5, 0.5])),
        (np.array([0.1, 0.9]), np.array([0.8, 0.2])),
        (np.array([0.0, 0.0]), np.array([1.0, 1.0])),
    ]

    for v1, v2 in test_profiles:
        try:
            a1, a2 = alloc_fn(v1, v2)
            p1, p2 = pay_fn(v1, v2)

            # shape checks
            assert np.array(a1).shape == (2,), "a1 must have shape (2,)"
            assert np.array(a2).shape == (2,), "a2 must have shape (2,)"

            # feasibility
            a1, a2 = np.array(a1, dtype=float), np.array(a2, dtype=float)
            assert np.all(a1 >= -1e-6) and np.all(a1 <= 1 + 1e-6), "a1 out of [0,1]"
            assert np.all(a2 >= -1e-6) and np.all(a2 <= 1 + 1e-6), "a2 out of [0,1]"
            assert np.all(a1 + a2 <= 1 + 1e-6), "allocation infeasible: a1+a2 > 1"

            # payments non-negative
            assert float(p1) >= -1e-6, "p1 negative"
            assert float(p2) >= -1e-6, "p2 negative"

        except Exception:
            return False

    return True


# ---------------------------------------------------------------------------
# Mutator
# ---------------------------------------------------------------------------


class Mutator:
    """LLM-based mutator for the auction EDEN loop."""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.adaptive_controller = AdaptiveMutationController()

    def update_adaptive_state(self, current_best_fitness: float) -> None:
        self.adaptive_controller.update(current_best_fitness)

    def get_mutation_info(self) -> Dict[str, Any]:
        return {
            "temperature": self.adaptive_controller.temperature,
            "mutation_strength": self.adaptive_controller.get_mutation_strength(),
            "crossover_probability": self.adaptive_controller.get_crossover_probability(),
            "steps_without_improvement": self.adaptive_controller.steps_without_improvement,
        }

    @observe()
    def mutate(
        self,
        parent: Organism,
        inspirations: List[Organism],
        mutation_strength: Optional[str] = None,
        max_retries: int = 3,
    ) -> Organism:
        """
        Mutate a parent organism to produce a child.

        Parameters
        ----------
        parent           : organism to mutate
        inspirations     : elite organisms to draw ideas from
        mutation_strength: override adaptive controller if provided
        max_retries      : number of LLM call retries on parse/validation failure

        Returns
        -------
        A new Organism with mutated alloc_src and pay_src.
        Falls back to the parent on persistent failure (with a flag in reasoning).
        """
        strength = mutation_strength or self.adaptive_controller.get_mutation_strength()
        user_prompt = build_mutation_prompt(parent, inspirations, strength)
        logger.info(
            "LLM mutate: parent=%s gen=%s strength=%s model=%s inspirations=%d max_retries=%d",
            parent.id,
            parent.generation,
            strength,
            self.model,
            len(inspirations),
            max_retries,
        )

        for attempt in range(max_retries):
            try:
                req_temp = 0.8 + 0.1 * attempt
                logger.debug(
                    "LLM request attempt %d/%d (request_temperature=%.2f)",
                    attempt + 1,
                    max_retries,
                    req_temp,
                )
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=req_temp,
                )

                raw = response.choices[0].message.content
                logger.debug(
                    "LLM raw response length=%d chars",
                    len(raw or ""),
                )
                result = _parse_response(raw)

                if not _validate_organism(result.alloc_src, result.pay_src):
                    raise ValueError(
                        "Validation failed — lambdas produced invalid outputs"
                    )

                child = Organism(
                    alloc_src=result.alloc_src,
                    pay_src=result.pay_src,
                    generation=parent.generation + 1,
                    parent_id=parent.id,
                    mutation_reasoning=result.mutation_reasoning,
                )
                logger.info(
                    "LLM mutate success: child=%s gen=%s (attempt %d/%d)",
                    child.id,
                    child.generation,
                    attempt + 1,
                    max_retries,
                )
                reason = result.mutation_reasoning or ""
                logger.debug("mutation reasoning: %s", reason[:500])
                return child

            except Exception as e:
                logger.warning(
                    "LLM mutate attempt %d/%d failed (parent=%s): %s",
                    attempt + 1,
                    max_retries,
                    parent.id,
                    e,
                )
                if attempt == max_retries - 1:
                    logger.error(
                        "LLM mutate exhausted retries; returning parent copy as child (parent=%s)",
                        parent.id,
                    )
                    return Organism(
                        alloc_src=parent.alloc_src,
                        pay_src=parent.pay_src,
                        generation=parent.generation + 1,
                        parent_id=parent.id,
                        mutation_reasoning=f"FALLBACK (mutation failed after {max_retries} attempts: {e})",
                    )

    def _get_strength_guidance(self, strength: str) -> str:
        return _get_strength_guidance(strength)
