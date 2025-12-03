import os
from typing import List, Optional
from langfuse import observe
from langfuse.openai import openai
from database import Organism
from utils import build_mutation_prompt, MutationResponse
from evolve_types import SimulationResults


class Mutator:
    """LLM-based mutator for genetic programming."""

    def __init__(self, model: str = "gpt-5.1-2025-11-13"):
        self.model = model
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @observe()
    def mutate(
        self,
        parent: Organism,
        inspirations: List[Organism],
        arrival_rate: float,
        service_rate: float,
        parent_simulation_results: Optional[SimulationResults] = None,
    ) -> Organism:
        """
        Mutate a parent organism to create a child.

        Args:
            parent: The parent organism to mutate
            inspirations: High-performing organisms for inspiration
            arrival_rate: Fixed exogenous arrival rate (lambda)
            service_rate: Fixed exogenous service rate (mu)
            parent_simulation_results: Optional simulation results from parent evaluation

        Returns:
            A new child Organism with mutated entry/exit rules
        """
        prompt = build_mutation_prompt(
            parent, inspirations, arrival_rate, service_rate, parent_simulation_results
        )

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

        return Organism(
            entry_rule_code=result.entry_rule_code,
            exit_rule_code=result.exit_rule_code,
            queue_discipline=result.queue_discipline.upper(),
            generation=parent.generation + 1,
            parent_id=parent.id,
        )
