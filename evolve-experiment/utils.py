from typing import Callable, List
from pydantic import BaseModel, ConfigDict
from database import Organism


def parse_code_to_function(code_str: str) -> Callable:
    """Parse a Python lambda string into a callable function."""
    if not code_str:
        raise ValueError("Code string cannot be empty")

    clean_code = code_str.strip()
    # Strip markdown if present
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

    arrival_rate_code: str
    service_rate_code: str
    entry_rule_code: str
    exit_rule_code: str
    mutation_reasoning: str


def build_mutation_prompt(parent: Organism, inspirations: List[Organism]) -> str:
    """Build the prompt for mutating a parent organism."""

    # Format inspirations
    inspiration_text = ""
    if inspirations:
        inspiration_text = "\n\nHIGH-PERFORMING ORGANISMS FOR INSPIRATION:\n"
        for i, org in enumerate(inspirations, 1):
            inspiration_text += f"""
Organism {i} (fitness: {org.fitness:.4f}):
  arrival_rate_code: {org.arrival_rate_code}
  service_rate_code: {org.service_rate_code}
  entry_rule_code: {org.entry_rule_code}
  exit_rule_code: {org.exit_rule_code}
"""

    return f"""You are a genetic programming system that mutates queue control functions.

PARENT ORGANISM (generation {parent.generation}, fitness: {parent.fitness if parent.fitness else 'not evaluated'}):
  arrival_rate_code: {parent.arrival_rate_code}
  service_rate_code: {parent.service_rate_code}
  entry_rule_code: {parent.entry_rule_code}
  exit_rule_code: {parent.exit_rule_code}
{inspiration_text}
FUNCTION SIGNATURES:
- arrival_rate_code: lambda k: <float>  (k = queue length, returns arrival rate)
- service_rate_code: lambda k: <float>  (k = queue length, returns service rate)
- entry_rule_code: lambda k: <float>    (k = queue length, returns probability [0,1])
- exit_rule_code: lambda k, l: (<float>, <float>)  (k = queue length, l = position, returns (rate, prob))

MUTATION GUIDELINES:
- Make small, targeted changes to improve fitness
- Can adjust constants, add/remove conditions, change mathematical relationships
- Learn from high-performing inspirations but don't copy exactly
- Ensure functions are valid Python lambda expressions
- For exit_rule_code, return (0.0, 0.0) for no exits

OBJECTIVE: Maximize welfare score W (higher is better).

Output the complete mutated functions."""
