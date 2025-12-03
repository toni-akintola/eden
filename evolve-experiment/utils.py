from typing import Callable, List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field
from database import Organism, MutationRecord
from evolve_types import SimulationResults


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
    queue_discipline: str = Field(
        description="Queue discipline: FCFS, LIFO, or SIRO",
        pattern="^(FCFS|LIFO|SIRO)$",
    )
    information_rule: str = Field(
        description="Information rule: NO_INFORMATION, FULL_INFORMATION, or COARSE_INFORMATION",
        pattern="^(NO_INFORMATION|FULL_INFORMATION|COARSE_INFORMATION)$",
    )
    mutation_reasoning: str


def create_mutation_record(
    parent: Organism,
    child_entry: str,
    child_exit: str,
    child_discipline: str,
    child_info_rule: str,
    mutation_reasoning: str,
) -> MutationRecord:
    """Create a MutationRecord tracking what changed from parent to child."""
    return MutationRecord(
        entry_rule_changed=(parent.entry_rule_code != child_entry),
        exit_rule_changed=(parent.exit_rule_code != child_exit),
        queue_discipline_changed=(
            parent.queue_discipline.upper() != child_discipline.upper()
        ),
        information_rule_changed=(
            parent.information_rule.upper() != child_info_rule.upper()
        ),
        mutation_reasoning=mutation_reasoning,
        parent_fitness=parent.fitness,
        parent_entry_rule=parent.entry_rule_code,
        parent_exit_rule=parent.exit_rule_code,
        parent_queue_discipline=parent.queue_discipline,
        parent_information_rule=parent.information_rule,
    )


def build_mutation_prompt(
    parent: Organism,
    inspirations: List[Organism],
    arrival_rate: float,
    service_rate: float,
    parent_simulation_results: Optional[SimulationResults] = None,
    lineage_history: Optional[List[Dict[str, Any]]] = None,
    successful_patterns: Optional[Dict[str, int]] = None,
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
  queue_discipline: {org.queue_discipline}
  information_rule: {org.information_rule}
"""

    behavior_data = ""
    if parent_simulation_results:
        total_agents = (
            parent_simulation_results.num_served
            + parent_simulation_results.num_voluntary_abandonment
            + parent_simulation_results.num_designer_exit
        )
        abandonment_rate = (
            parent_simulation_results.num_voluntary_abandonment / total_agents
            if total_agents > 0
            else 0.0
        )

        behavior_data = f"""
PARENT SIMULATION RESULTS (agent behavior data):
  Expected queue length E[k]: {parent_simulation_results.expected_queue_length_E_k:.4f}
  Expected service flow E[μ_k]: {parent_simulation_results.expected_service_flow_E_mu_k:.4f}
  Total agents served: {parent_simulation_results.num_served}
  Voluntary abandonments: {parent_simulation_results.num_voluntary_abandonment} ({abandonment_rate*100:.2f}% of total)
  Designer-induced exits: {parent_simulation_results.num_designer_exit}
  Average wait time (served agents): {parent_simulation_results.avg_wait_time_served:.4f}
  Total simulation time: {parent_simulation_results.total_run_time:.2f}
"""

    # Build lineage/mutation history section
    lineage_text = ""
    if lineage_history and len(lineage_history) > 1:
        lineage_text = "\n\nMUTATION HISTORY (recent ancestors):\n"
        lineage_text += "This shows what mutations led to the current organism and whether they helped.\n"

        for i, ancestor in enumerate(lineage_history[1:], 1):  # Skip current (index 0)
            fitness_str = f"{ancestor['fitness']:.4f}" if ancestor["fitness"] else "N/A"
            lineage_text += f"\n  Ancestor {i} (gen {ancestor['generation']}, fitness: {fitness_str}):\n"
            lineage_text += f"    entry: {ancestor['entry_rule']}\n"
            lineage_text += f"    exit: {ancestor['exit_rule']}\n"
            lineage_text += f"    discipline: {ancestor['queue_discipline']}, info: {ancestor['information_rule']}\n"

            if "mutation" in ancestor:
                mut = ancestor["mutation"]
                delta_str = (
                    f"{mut['fitness_delta']:+.4f}" if mut["fitness_delta"] else "N/A"
                )
                success_str = "✓ IMPROVED" if mut["was_successful"] else "✗ worse/same"
                lineage_text += (
                    f"    → Mutation: {mut['changes']} ({success_str}, Δ={delta_str})\n"
                )
                if mut["reasoning"]:
                    # Truncate long reasoning
                    reasoning = (
                        mut["reasoning"][:200] + "..."
                        if len(mut["reasoning"]) > 200
                        else mut["reasoning"]
                    )
                    lineage_text += f"    → Reasoning: {reasoning}\n"

    # Build successful patterns section
    patterns_text = ""
    if successful_patterns:
        patterns_text = "\n\nSUCCESSFUL MUTATION PATTERNS (what tends to work):\n"
        sorted_patterns = sorted(successful_patterns.items(), key=lambda x: -x[1])
        for pattern, count in sorted_patterns[:5]:  # Top 5 patterns
            patterns_text += f"  - {pattern}: {count} successful mutations\n"

    return f"""You are a genetic programming system that mutates queue control functions.

FIXED EXOGENOUS PARAMETERS (do not change):
  arrival_rate (lambda): {arrival_rate}
  service_rate (mu): {service_rate}

PARENT ORGANISM (generation {parent.generation}, fitness: {parent.fitness if parent.fitness else 'not evaluated'}):
  entry_rule_code: {parent.entry_rule_code}
  exit_rule_code: {parent.exit_rule_code}
  queue_discipline: {parent.queue_discipline}
  information_rule: {parent.information_rule}
{behavior_data}{lineage_text}{patterns_text}{inspiration_text}
MUTABLE PARAMETERS:
- entry_rule_code: lambda k: <float>  (k = queue length, returns entry probability [0,1])
- exit_rule_code: lambda k, l: (<float>, <float>)  (k = queue length, l = position, returns (exit_rate, exit_prob))
- queue_discipline: One of "FCFS", "LIFO", or "SIRO"
- information_rule: One of "NO_INFORMATION", "FULL_INFORMATION", or "COARSE_INFORMATION"

QUEUE DISCIPLINE OPTIONS:
- FCFS: Agents are served in order of arrival (fair, predictable wait times)
- LIFO: Most recently arrived agents are served first (reduces wait for new arrivals, but unfair to early arrivals)
- SIRO: Agents are selected randomly for service (unpredictable, can reduce strategic behavior)

INFORMATION RULE OPTIONS (affects agent beliefs and voluntary abandonment):
- NO_INFORMATION: Agents don't observe queue length; they use expected queue length from the steady-state distribution. Beliefs are uncertain and updated via Bayesian inference.
- FULL_INFORMATION: Agents observe the exact queue length and their position. They know exactly how long they'll wait, leading to more accurate abandonment decisions.
- COARSE_INFORMATION: Agents observe a coarse signal (short/medium/long queue). Provides partial information - better than no information but less precise than full.

HOW INFORMATION AFFECTS AGENT BEHAVIOR:
- With NO_INFORMATION, agents may stay too long (don't know queue is long) or leave too early (pessimistic beliefs)
- With FULL_INFORMATION, agents make optimal abandonment decisions, which can reduce welfare if many leave
- With COARSE_INFORMATION, agents get partial guidance - a balance between the extremes

MUTATION GUIDELINES:
- Make small, targeted changes to improve fitness
- Can adjust constants, add/remove conditions, change mathematical relationships
- Can change queue discipline and information rule if it might improve performance
- Consider how information rule interacts with entry/exit policies
- Learn from high-performing inspirations but don't copy exactly
- Ensure functions are valid Python lambda expressions
- Entry rule should return probability in [0, 1]
- Exit rule returns (0.0, 0.0) for no exits, or (rate, probability) for designer-induced exits

OBJECTIVE: Maximize welfare score W (higher is better) by optimizing entry/exit policies, queue discipline, and information rule.

Output the complete mutated functions, queue discipline, and information rule."""
