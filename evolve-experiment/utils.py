from typing import Callable, List
from pydantic import BaseModel, ConfigDict
from database import Organism


def parse_code_to_function(code_str: str) -> Callable:
    """Parse a Python lambda string into a callable function."""
    if not code_str:
        raise ValueError("Code string cannot be empty")

    clean_code = code_str.strip()
    if clean_code.startswith("```"):
        clean_code = clean_code.split("\n", 1)[-1]
    if clean_code.endswith("```"):
        clean_code = clean_code.rsplit("```", 1)[0]
    clean_code = clean_code.strip()

    func = eval(clean_code)
    if not callable(func):
        raise ValueError(f"Code does not evaluate to a callable: {clean_code}")
    return func


class MutationResponse(BaseModel):
    """LLM response schema for mutating an organism."""

    model_config = ConfigDict(extra="forbid")

    entry_rule_code: str
    exit_rule_code: str
    mutation_reasoning: str


def build_mutation_prompt(
    parent: Organism,
    inspirations: List[Organism],
    arrival_rate: float,
    service_rate: float,
) -> str:
    """Build the prompt for mutating a parent organism."""

    inspiration_text = ""
    if inspirations:
        inspiration_text = "\n\nHIGH-PERFORMING ORGANISMS FOR INSPIRATION:\n"
        for i, org in enumerate(inspirations, 1):
            inspiration_text += f"""
Organism {i} (fitness: {org.fitness:.4f}):
  entry_rule_code: {org.entry_rule_code}
  exit_rule_code: {org.exit_rule_code}
"""

    return f"""You are a genetic programming system that mutates queue control functions.

FIXED EXOGENOUS PARAMETERS (do not change):
  arrival_rate (lambda): {arrival_rate}
  service_rate (mu): {service_rate}

PARENT ORGANISM (generation {parent.generation}, fitness: {parent.fitness if parent.fitness else 'not evaluated'}):
  entry_rule_code: {parent.entry_rule_code}
  exit_rule_code: {parent.exit_rule_code}
{inspiration_text}
FUNCTION SIGNATURES (what you can mutate):
- entry_rule_code: lambda k: <float>  (k = queue length, returns entry probability [0,1])
- exit_rule_code: lambda k, l: (<float>, <float>)  (k = queue length, l = position, returns (exit_rate, exit_prob))

MUTATION GUIDELINES:
- Make small, targeted changes to improve fitness
- Can adjust constants, add/remove conditions, change mathematical relationships
- Learn from high-performing inspirations but don't copy exactly
- Ensure functions are valid Python lambda expressions
- Entry rule should return probability in [0, 1]
- Exit rule returns (0.0, 0.0) for no exits, or (rate, probability) for designer-induced exits

OBJECTIVE: Maximize welfare score W (higher is better) by optimizing entry/exit policies.

Output the complete mutated functions."""
