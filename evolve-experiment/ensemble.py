import os
from typing import List
from openai import OpenAI
from database import Organism
from utils import build_mutation_prompt, MutationResponse


class Mutator:
    """LLM-based mutator for genetic programming."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def mutate(self, parent: Organism, inspirations: List[Organism]) -> Organism:
        """
        Mutate a parent organism to create a child.

        Args:
            parent: The parent organism to mutate
            inspirations: High-performing organisms for inspiration

        Returns:
            A new child Organism with mutated code
        """
        prompt = build_mutation_prompt(parent, inspirations)

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
            arrival_rate_code=result.arrival_rate_code,
            service_rate_code=result.service_rate_code,
            entry_rule_code=result.entry_rule_code,
            exit_rule_code=result.exit_rule_code,
            generation=parent.generation + 1,
            parent_id=parent.id,
        )
