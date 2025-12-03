import os
import random
from typing import List, Optional, Dict, Any
from langfuse import observe
from langfuse.openai import openai
from database import Organism, MutationRecord
from utils import build_mutation_prompt, MutationResponse, create_mutation_record
from evolve_types import SimulationResults


def crossover(parent1: Organism, parent2: Organism) -> Organism:
    """
    Perform uniform crossover between two parent organisms.

    Randomly selects each trait (entry rule, exit rule, queue discipline,
    information rule) from one of the two parents.

    Returns:
        A new child organism with mixed traits from both parents.
    """
    # Randomly select each component from one of the parents
    entry_source = random.choice([parent1, parent2])
    exit_source = random.choice([parent1, parent2])
    discipline_source = random.choice([parent1, parent2])
    info_source = random.choice([parent1, parent2])

    # Track what came from where for mutation record
    changes = []
    if entry_source != exit_source:
        changes.append("entry+exit_crossover")
    if discipline_source != entry_source:
        changes.append("discipline_crossover")
    if info_source != entry_source:
        changes.append("info_crossover")

    mutation_record = MutationRecord(
        entry_rule_changed=(entry_source == parent2),
        exit_rule_changed=(exit_source == parent2),
        queue_discipline_changed=(discipline_source == parent2),
        information_rule_changed=(info_source == parent2),
        mutation_reasoning=f"Crossover: entry from {entry_source.id}, exit from {exit_source.id}, "
        f"discipline from {discipline_source.id}, info from {info_source.id}",
        parent_fitness=parent1.fitness,
        parent_entry_rule=parent1.entry_rule_code,
        parent_exit_rule=parent1.exit_rule_code,
        parent_queue_discipline=parent1.queue_discipline,
        parent_information_rule=parent1.information_rule,
    )

    return Organism(
        entry_rule_code=entry_source.entry_rule_code,
        exit_rule_code=exit_source.exit_rule_code,
        queue_discipline=discipline_source.queue_discipline,
        information_rule=info_source.information_rule,
        generation=max(parent1.generation, parent2.generation) + 1,
        parent_id=parent1.id,  # Track primary parent
        mutation_record=mutation_record,
    )


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

        # Tracking
        self.best_fitness_history: List[float] = []
        self.steps_without_improvement = 0
        self.last_best_fitness: Optional[float] = None

    def update(self, current_best_fitness: float) -> None:
        """Update temperature based on whether we improved."""
        self.best_fitness_history.append(current_best_fitness)

        if self.last_best_fitness is None:
            self.last_best_fitness = current_best_fitness
            return

        if (
            current_best_fitness > self.last_best_fitness + 0.01
        ):  # Small epsilon for improvement
            # We improved! Cool down (more exploitation)
            self.steps_without_improvement = 0
            self.temperature = max(
                self.min_temperature, self.temperature * self.cooldown_rate
            )
            self.last_best_fitness = current_best_fitness
        else:
            # No improvement
            self.steps_without_improvement += 1

            if self.steps_without_improvement >= self.stagnation_threshold:
                # Heat up (more exploration)
                self.temperature = min(
                    self.max_temperature, self.temperature * self.heatup_rate
                )

    def get_mutation_strength(self) -> str:
        """Get mutation strength description based on temperature."""
        if self.temperature < 0.6:
            return "small"
        elif self.temperature < 1.2:
            return "medium"
        elif self.temperature < 2.0:
            return "large"
        else:
            return "radical"

    def should_do_random_restart(self) -> bool:
        """Check if we should inject a random organism."""
        # 5% base chance, increases with stagnation
        base_chance = 0.05
        stagnation_bonus = min(0.2, self.steps_without_improvement * 0.02)
        return random.random() < (base_chance + stagnation_bonus)

    def get_crossover_probability(self) -> float:
        """Get probability of using crossover vs mutation."""
        # Higher temperature = more crossover (exploration)
        # Base: 30% crossover, scales with temperature
        return min(0.6, 0.3 * self.temperature)


class Mutator:
    """LLM-based mutator for genetic programming with adaptive mutation."""

    def __init__(self, model: str = "gpt-5.1-2025-11-13"):
        self.model = model
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.adaptive_controller = AdaptiveMutationController()

    def update_adaptive_state(self, current_best_fitness: float) -> None:
        """Update the adaptive mutation controller with current best fitness."""
        self.adaptive_controller.update(current_best_fitness)

    def get_mutation_info(self) -> Dict[str, Any]:
        """Get current mutation state info for logging."""
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
        arrival_rate: float,
        service_rate: float,
        parent_simulation_results: Optional[SimulationResults] = None,
        lineage_history: Optional[List[Dict[str, Any]]] = None,
        successful_patterns: Optional[Dict[str, int]] = None,
        mutation_strength: Optional[str] = None,
    ) -> Organism:
        """
        Mutate a parent organism to create a child.

        Args:
            parent: The parent organism to mutate
            inspirations: High-performing organisms for inspiration
            arrival_rate: Fixed exogenous arrival rate (lambda)
            service_rate: Fixed exogenous service rate (mu)
            parent_simulation_results: Optional simulation results from parent evaluation
            lineage_history: Optional mutation history from ancestors
            successful_patterns: Optional dict of mutation patterns that tend to work
            mutation_strength: Optional override for mutation strength (small/medium/large/radical)

        Returns:
            A new child Organism with mutated entry/exit rules and mutation record
        """
        # Use adaptive strength if not overridden
        strength = mutation_strength or self.adaptive_controller.get_mutation_strength()

        prompt = build_mutation_prompt(
            parent,
            inspirations,
            arrival_rate,
            service_rate,
            parent_simulation_results,
            lineage_history,
            successful_patterns,
        )

        # Add mutation strength guidance to prompt
        strength_guidance = self._get_strength_guidance(strength)
        prompt = prompt + f"\n\n{strength_guidance}"

        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": "Mutate the parent organism to create an improved child.",
                },
            ],
            text_format=MutationResponse,
        )

        result = response.output_parsed

        # Create mutation record tracking what changed
        mutation_record = create_mutation_record(
            parent=parent,
            child_entry=result.entry_rule_code,
            child_exit=result.exit_rule_code,
            child_discipline=result.queue_discipline,
            child_info_rule=result.information_rule,
            mutation_reasoning=result.mutation_reasoning,
        )

        return Organism(
            entry_rule_code=result.entry_rule_code,
            exit_rule_code=result.exit_rule_code,
            queue_discipline=result.queue_discipline.upper(),
            information_rule=result.information_rule.upper(),
            generation=parent.generation + 1,
            parent_id=parent.id,
            mutation_record=mutation_record,
        )

    def _get_strength_guidance(self, strength: str) -> str:
        """Get mutation strength guidance for the LLM."""
        if strength == "small":
            return """MUTATION STRENGTH: SMALL (fine-tuning mode)
- Make minimal changes - adjust ONE parameter by a small amount
- Do NOT change the structure of rules
- Do NOT change queue discipline or information rule
- Example: change 0.8 to 0.75, or 5 to 4"""
        elif strength == "medium":
            return """MUTATION STRENGTH: MEDIUM (balanced exploration)
- Make moderate changes - adjust 1-2 parameters or conditions
- Can modify mathematical relationships slightly
- Can consider changing queue discipline or information rule if clearly beneficial
- Example: change threshold from 5 to 3, or add a small modifier"""
        elif strength == "large":
            return """MUTATION STRENGTH: LARGE (exploration mode)
- Make significant changes - try new approaches
- Can restructure rules, change rule types, or combine ideas
- Encouraged to try different queue discipline or information rule
- Example: change from threshold-based to gradient-based rule"""
        else:  # radical
            return """MUTATION STRENGTH: RADICAL (breakthrough mode)
- Make dramatic changes - completely rethink the approach
- Try unconventional combinations and rule structures
- MUST change at least 2 components significantly
- Ignore what worked before - search for new optima
- Example: completely new entry/exit strategy"""
