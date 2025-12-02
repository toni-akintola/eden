from typing import Annotated, Callable, Union, Any
from openai import OpenAI
import os
from database import Database
import json
from pydantic import BaseModel, Field, ConfigDict
from evolve_types import (
    CheTercieuxQueueModel,
    PrimitiveProcess,
    EntryExitRule,
    QueueDiscipline,
    InformationRule,
)


def parse_code_to_function(code_str: str) -> Callable:
    """
    Parses a string containing a Python lambda expression into a callable function.

    WARNING: Uses eval(), so only use with trusted/generated code.

    Args:
        code_str: Python code string (preferably a lambda expression)

    Returns:
        Callable function

    Examples:
        >>> f = parse_code_to_function("lambda k: 2.0")
        >>> f(5)
        2.0
        >>> g = parse_code_to_function("lambda k: k * 0.5")
        >>> g(10)
        5.0
    """
    if not code_str:
        raise ValueError("Code string cannot be empty")

    # Clean up markdown code blocks if present
    clean_code = code_str.strip()
    if clean_code.startswith("```python"):
        clean_code = clean_code[9:]
    if clean_code.startswith("```"):
        clean_code = clean_code[3:]
    if clean_code.endswith("```"):
        clean_code = clean_code[:-3]
    clean_code = clean_code.strip()

    try:
        # Try eval first (for lambdas and simple expressions)
        func = eval(clean_code)
        if callable(func):
            return func
        else:
            raise ValueError(f"Code does not evaluate to a callable: {clean_code}")
    except SyntaxError:
        # If eval fails, try exec (for def statements)
        local_scope = {}
        exec(clean_code, {"__builtins__": __builtins__}, local_scope)
        # Return the first callable found
        for value in local_scope.values():
            if callable(value):
                return value
        raise ValueError(f"No callable function found in code: {clean_code}")
    except Exception as e:
        raise ValueError(f"Failed to parse code '{clean_code}': {e}")


class QueueConfigSchema(BaseModel):
    """Schema for queue design configuration returned by agents (optimization variables only)."""

    model_config = ConfigDict(extra="forbid")

    arrival_rate_code: str = Field(
        description="Python lambda expression: lambda k: <arrival_rate>"
    )
    service_rate_code: str = Field(
        description="Python lambda expression: lambda k: <service_rate>"
    )
    entry_rule_code: str = Field(
        description="Python lambda expression: lambda k: <entry_probability [0,1]>"
    )
    exit_rule_code: str = Field(
        description="Python lambda expression: lambda k, l: (<rate_y>, <prob_z>) or 'lambda k, l: (0.0, 0.0)' for no exits"
    )
    queue_discipline: str = Field(description="Queue discipline: FCFS, LIFO, or SIRO")
    information_rule: str = Field(
        description="Information rule: NO_INFORMATION_BEYOND_REC, FULL_INFORMATION, or COARSE_INFORMATION"
    )
    reasoning: str = Field(description="Reasoning for this configuration")


class ExploreResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis: str
    pattern_insights: str
    candidate_configs: list[QueueConfigSchema]
    confidence: str
    key_learnings: str


class RefineResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation: str
    optimal_strategy: str
    recommended_config: QueueConfigSchema
    reasoning: str
    expected_outcome: str
    alternative_config: QueueConfigSchema


class ActResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_config: QueueConfigSchema
    confidence_level: str
    reasoning: str
    expected_metrics: str


def get_model_names():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return [model.id for model in client.models.list()]


def build_explore_prompt(task: str, database: Database) -> str:
    attempts_data = []
    for attempt in database.get_attempts():
        attempts_data.append(
            {
                "hyperparameters": attempt.attempt,  # This will be a dict
                "feedback": attempt.observations,
                "efficiency_score": attempt.score,
            }
        )

    return f"""You are an expert queuing systems analyst optimizing the Che-Tercieux queue model parameters.

TASK: {task}

PAST CONFIGURATIONS TESTED:
{json.dumps(attempts_data, indent=2) if attempts_data else "No previous configurations tested yet."}

Your job is to EXPLORE and analyze the hyperparameter space. Based on past configurations:
1. Analyze which hyperparameter combinations performed well or poorly
2. Identify patterns in efficiency scores and feedback
3. Consider queuing theory principles and the Che-Tercieux model dynamics
4. Propose 2-3 candidate hyperparameter configurations with reasoning

QUEUE DESIGN PARAMETERS TO OPTIMIZE:
- arrival_rate_code: Python lambda expression (k -> arrival_rate)
- service_rate_code: Python lambda expression (k -> service_rate)
- entry_rule_code: Python lambda expression (k -> entry_probability [0,1])
- exit_rule_code: Python lambda expression ((k, l) -> (rate_y, prob_z))
- queue_discipline: string ("FCFS", "LIFO", or "SIRO")
- information_rule: string ("NO_INFORMATION_BEYOND_REC", "FULL_INFORMATION", or "COARSE_INFORMATION")

NOTE: Economic parameters (V_surplus, C_waiting_cost, R_provider_profit, alpha_weight) are FIXED by the problem context.

FUNCTION CODE EXAMPLES:
- Constant: "lambda k: 2.0"
- Linear: "lambda k: 2.0 + k * 0.1"
- Threshold: "lambda k: 3.0 if k < 5 else 1.5"
- Decay: "lambda k: max(0.1, 1.0 - k * 0.1)"
- Bounded: "lambda k: min(5.0, 2.0 + k * 0.5)"
- Exit (no exits): "lambda k, l: (0.0, 0.0)"
- Exit (threshold): "lambda k, l: (0.5, 0.9) if k > 10 else (0.0, 0.0)"

OBJECTIVE: Maximize designer's welfare score W (higher is better), which balances provider profit and agent utility.

Respond in JSON format:
{{
    "analysis": "Your analysis of what configurations worked well and why",
    "pattern_insights": "Key insights from the efficiency scores and feedback patterns",
    "candidate_configs": [
        {{
            "arrival_rate_code": "lambda k: ...",
            "service_rate_code": "lambda k: ...",
            "entry_rule_code": "lambda k: ...",
            "exit_rule_code": "lambda k, l: ...",
            "queue_discipline": "<FCFS|LIFO|SIRO>",
            "information_rule": "<NO_INFORMATION_BEYOND_REC|FULL_INFORMATION|COARSE_INFORMATION>",
            "reasoning": "why this configuration might work well"
        }},
        {{
            "arrival_rate_code": "lambda k: ...",
            "service_rate_code": "lambda k: ...",
            "entry_rule_code": "lambda k: ...",
            "exit_rule_code": "lambda k, l: ...",
            "queue_discipline": "<FCFS|LIFO|SIRO>",
            "information_rule": "<NO_INFORMATION_BEYOND_REC|FULL_INFORMATION|COARSE_INFORMATION>",
            "reasoning": "alternative approach"
        }}
    ],
    "confidence": "low/medium/high",
    "key_learnings": "What you've learned about the optimal hyperparameter space"
}}"""


def build_refine_prompt(structured_exploration: dict) -> str:
    return f"""You are a queuing systems optimizer that refines hyperparameter configurations for the Che-Tercieux queue model.

EXPLORATION ANALYSIS:
{json.dumps(structured_exploration, indent=2)}

Your job is to REFINE the exploration into an optimal queue design configuration:
1. Evaluate the candidate configurations from exploration
2. Consider queuing theory and the Che-Tercieux model dynamics
3. Select the best queue design and provide an alternative

NOTE: Economic parameters (V, C, R, alpha) are fixed by the problem context. Focus only on optimizing the queue design.

Provide Python lambda expressions for the rate functions. Examples:
- "lambda k: 2.0" (constant)
- "lambda k: 2.0 if k < 5 else 1.0" (threshold)
- "lambda k: max(0.1, 1.0 - k * 0.1)" (decay)
- "lambda k, l: (0.5, 0.9) if k > 10 else (0.0, 0.0)" (exit rule)

Respond in JSON format:
{{
    "evaluation": "Assessment of the exploration analysis and candidate configurations",
    "optimal_strategy": "Description of the optimization approach (e.g., 'Balance provider profit and agent utility', 'Optimize entry/exit rules for welfare')",
    "recommended_config": {{
        "arrival_rate_code": "lambda k: ...",
        "service_rate_code": "lambda k: ...",
        "entry_rule_code": "lambda k: ...",
        "exit_rule_code": "lambda k, l: ...",
        "queue_discipline": "<FCFS|LIFO|SIRO>",
        "information_rule": "<NO_INFORMATION_BEYOND_REC|FULL_INFORMATION|COARSE_INFORMATION>",
        "reasoning": "Detailed explanation of why this is the best configuration"
    }},
    "reasoning": "Detailed explanation of the optimization strategy",
    "expected_outcome": "What welfare score and metrics we expect from this configuration",
    "alternative_config": {{
        "arrival_rate_code": "lambda k: ...",
        "service_rate_code": "lambda k: ...",
        "entry_rule_code": "lambda k: ...",
        "exit_rule_code": "lambda k, l: ...",
        "queue_discipline": "<FCFS|LIFO|SIRO>",
        "information_rule": "<NO_INFORMATION_BEYOND_REC|FULL_INFORMATION|COARSE_INFORMATION>",
        "reasoning": "Backup configuration if recommended one doesn't work well"
    }}
}}"""


def build_act_prompt(structured_refinement: dict) -> str:
    return f"""You are the final decision maker that commits to a hyperparameter configuration for the Che-Tercieux queue model.

REFINED STRATEGY:
{json.dumps(structured_refinement, indent=2)}

Your job is to ACT and make the final decision:
1. Review the recommended queue design configuration
2. Do a final sanity check (ensure queue stability and valid parameter ranges)
3. Commit to specific queue design values as Python lambda expressions

NOTE: Economic parameters (V, C, R, alpha) are fixed by the problem context. You only control the queue design.

Provide valid Python lambda expressions. Examples:
- "lambda k: 2.0"
- "lambda k: 3.0 if k < 5 else 2.0"
- "lambda k, l: (0.0, 0.0)" for no exits

Respond in JSON format:
{{
    "final_config": {{
        "arrival_rate_code": "lambda k: ...",
        "service_rate_code": "lambda k: ...",
        "entry_rule_code": "lambda k: ...",
        "exit_rule_code": "lambda k, l: ...",
        "queue_discipline": "<FCFS|LIFO|SIRO>",
        "information_rule": "<NO_INFORMATION_BEYOND_REC|FULL_INFORMATION|COARSE_INFORMATION>",
        "reasoning": "Final justification for this configuration"
    }},
    "confidence_level": "low/medium/high",
    "reasoning": "Final justification for this configuration choice",
    "expected_metrics": "What welfare score and metrics (time spent at queue lengths, number served) we expect"
}}"""


def convert_config_to_model(
    config: Union[QueueConfigSchema, dict],
    V_surplus: float = 10.0,
    C_waiting_cost: float = 1.0,
    R_provider_profit: float = 5.0,
    alpha_weight: float = 0.5,
) -> CheTercieuxQueueModel:
    """
    Converts the agent's config to a CheTercieuxQueueModel.

    Args:
        config: QueueConfigSchema or dictionary with queue design parameters
        V_surplus: Net surplus from service (fixed by user)
        C_waiting_cost: Per-period waiting cost (fixed by user)
        R_provider_profit: Profit per served agent (fixed by user)
        alpha_weight: Weight on agents' welfare (fixed by user)

    Returns:
        A fully constructed CheTercieuxQueueModel ready for simulation
    """
    # Convert to dict if it's a Pydantic model
    if isinstance(config, QueueConfigSchema):
        config_dict = config.model_dump()
    else:
        config_dict = config

    # Parse rate functions from code strings
    try:
        arrival_fn = parse_code_to_function(
            config_dict.get("arrival_rate_code", "lambda k: 1.0")
        )
        service_fn = parse_code_to_function(
            config_dict.get("service_rate_code", "lambda k: 1.0")
        )
        entry_fn = parse_code_to_function(
            config_dict.get("entry_rule_code", "lambda k: 1.0")
        )
        exit_fn = parse_code_to_function(
            config_dict.get("exit_rule_code", "lambda k, l: (0.0, 0.0)")
        )
    except Exception as e:
        raise ValueError(f"Failed to parse function code: {e}")

    # Construct nested models
    primitive_process = PrimitiveProcess(
        arrival_rate_fn=arrival_fn, service_rate_fn=service_fn
    )

    design_rules = EntryExitRule(entry_rule_fn=entry_fn, exit_rule_fn=exit_fn)

    # Parse enums
    discipline_str = config_dict.get("queue_discipline", "FCFS")
    try:
        queue_discipline = QueueDiscipline[discipline_str]
    except (KeyError, TypeError):
        queue_discipline = QueueDiscipline.FCFS

    info_str = config_dict.get("information_rule", "NO_INFORMATION_BEYOND_REC")
    try:
        information_rule = InformationRule[info_str]
    except (KeyError, TypeError):
        information_rule = InformationRule.NO_INFORMATION_BEYOND_REC

    # Construct full model with user-provided economic parameters
    return CheTercieuxQueueModel(
        V_surplus=V_surplus,
        C_waiting_cost=C_waiting_cost,
        R_provider_profit=R_provider_profit,
        alpha_weight=alpha_weight,
        primitive_process=primitive_process,
        design_rules=design_rules,
        queue_discipline=queue_discipline,
        information_rule=information_rule,
    )
