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


class QueueConfigSchema(BaseModel):
    """Schema for queue model configuration returned by agents."""

    model_config = ConfigDict(extra="forbid")

    V_surplus: float = Field(description="Net surplus V from service (V > 0)")
    C_waiting_cost: float = Field(description="Per-period waiting cost C (C > 0)")
    R_provider_profit: float = Field(
        description="Profit R for each served agent (R > 0)"
    )
    alpha_weight: float = Field(
        description="Weight on agents' welfare (alpha in [0, 1])"
    )
    arrival_rate_fn: Union[float, dict[str, float]] = Field(
        description="Constant value or dict mapping queue_length -> arrival_rate"
    )
    service_rate_fn: Union[float, dict[str, float]] = Field(
        description="Constant value or dict mapping queue_length -> service_rate"
    )
    entry_rule_fn: Union[float, dict[str, float]] = Field(
        description="Constant value or dict mapping queue_length -> entry_probability"
    )
    exit_rule_fn: Union[dict[str, list[float]], None] = Field(
        description="Dict mapping 'k,l' -> [rate_y, prob_z] or null for no exits"
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

MODEL PARAMETERS TO OPTIMIZE:
- V_surplus: float (net surplus V from service, V > 0)
- C_waiting_cost: float (per-period waiting cost C, C > 0)
- R_provider_profit: float (profit R for each served agent, R > 0)
- alpha_weight: float (weight on agents' welfare, alpha in [0, 1])
- arrival_rate_fn: dict mapping queue_length (int) -> arrival_rate (float), or constant value
- service_rate_fn: dict mapping queue_length (int) -> service_rate (float), or constant value
- entry_rule_fn: dict mapping queue_length (int) -> entry_probability [0, 1] (float), or constant value
- exit_rule_fn: dict mapping "queue_length,position" (string like "k,l") -> tuple [rate_y (float), prob_z (float)], or null for no exits
- queue_discipline: string ("FCFS", "LIFO", or "SIRO")
- information_rule: string ("NO_INFORMATION_BEYOND_REC", "FULL_INFORMATION", or "COARSE_INFORMATION")

NOTE: For functions, you can provide:
- A constant value (e.g., 2.0) which will be used for all queue lengths
- A dictionary mapping specific queue_lengths to values (e.g., {{0: 2.0, 1: 2.5, 2: 3.0}})
- For exit_rule_fn, use null or empty dict for no exits, or provide entries like {{"1,1": [0.5, 0.8]}} meaning at queue_length=1, position=1, rate_y=0.5, prob_z=0.8

OBJECTIVE: Maximize designer's welfare score W (higher is better), which balances provider profit and agent utility.

Respond in JSON format:
{{
    "analysis": "Your analysis of what configurations worked well and why",
    "pattern_insights": "Key insights from the efficiency scores and feedback patterns",
    "candidate_configs": [
        {{
            "V_surplus": <float>,
            "C_waiting_cost": <float>,
            "R_provider_profit": <float>,
            "alpha_weight": <float in [0,1]>,
            "arrival_rate_fn": <float or dict>,
            "service_rate_fn": <float or dict>,
            "entry_rule_fn": <float or dict>,
            "exit_rule_fn": <null or dict>,
            "queue_discipline": "<FCFS|LIFO|SIRO>",
            "information_rule": "<NO_INFORMATION_BEYOND_REC|FULL_INFORMATION|COARSE_INFORMATION>",
            "reasoning": "why this configuration might work well"
        }},
        {{
            "V_surplus": <float>,
            "C_waiting_cost": <float>,
            "R_provider_profit": <float>,
            "alpha_weight": <float in [0,1]>,
            "arrival_rate_fn": <float or dict>,
            "service_rate_fn": <float or dict>,
            "entry_rule_fn": <float or dict>,
            "exit_rule_fn": <null or dict>,
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

Your job is to REFINE the exploration into an optimal hyperparameter configuration:
1. Evaluate the candidate configurations from exploration
2. Consider queuing theory and the Che-Tercieux model dynamics
3. Balance provider profit and agent utility based on alpha_weight
4. Select the best configuration and provide an alternative

NOTE: For functions, you can provide:
- A constant value (e.g., 2.0) which will be used for all queue lengths
- A dictionary mapping specific queue_lengths to values (e.g., {{0: 2.0, 1: 2.5, 2: 3.0}})
- For exit_rule_fn, use null or empty dict for no exits, or provide entries like {{"1,1": [0.5, 0.8]}}

Respond in JSON format:
{{
    "evaluation": "Assessment of the exploration analysis and candidate configurations",
    "optimal_strategy": "Description of the optimization approach (e.g., 'Balance provider profit and agent utility', 'Optimize entry/exit rules for welfare')",
    "recommended_config": {{
        "V_surplus": <float>,
        "C_waiting_cost": <float>,
        "R_provider_profit": <float>,
        "alpha_weight": <float in [0,1]>,
        "arrival_rate_fn": <float or dict>,
        "service_rate_fn": <float or dict>,
        "entry_rule_fn": <float or dict>,
        "exit_rule_fn": <null or dict>,
        "queue_discipline": "<FCFS|LIFO|SIRO>",
        "information_rule": "<NO_INFORMATION_BEYOND_REC|FULL_INFORMATION|COARSE_INFORMATION>",
        "reasoning": "Detailed explanation of why this is the best configuration"
    }},
    "reasoning": "Detailed explanation of the optimization strategy",
    "expected_outcome": "What welfare score and metrics we expect from this configuration",
    "alternative_config": {{
        "V_surplus": <float>,
        "C_waiting_cost": <float>,
        "R_provider_profit": <float>,
        "alpha_weight": <float in [0,1]>,
        "arrival_rate_fn": <float or dict>,
        "service_rate_fn": <float or dict>,
        "entry_rule_fn": <float or dict>,
        "exit_rule_fn": <null or dict>,
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
1. Review the recommended configuration
2. Do a final sanity check (ensure all values are valid: V_surplus > 0, C_waiting_cost > 0, R_provider_profit > 0, alpha_weight in [0,1])
3. Commit to specific hyperparameter values

NOTE: For functions, you can provide:
- A constant value (e.g., 2.0) which will be used for all queue lengths
- A dictionary mapping specific queue_lengths to values (e.g., {{0: 2.0, 1: 2.5, 2: 3.0}})
- For exit_rule_fn, use null or empty dict for no exits, or provide entries like {{"1,1": [0.5, 0.8]}}

Respond in JSON format:
{{
    "final_config": {{
        "V_surplus": <float>,
        "C_waiting_cost": <float>,
        "R_provider_profit": <float>,
        "alpha_weight": <float in [0,1]>,
        "arrival_rate_fn": <float or dict>,
        "service_rate_fn": <float or dict>,
        "entry_rule_fn": <float or dict>,
        "exit_rule_fn": <null or dict>,
        "queue_discipline": "<FCFS|LIFO|SIRO>",
        "information_rule": "<NO_INFORMATION_BEYOND_REC|FULL_INFORMATION|COARSE_INFORMATION>",
        "reasoning": "Final justification for this configuration"
    }},
    "confidence_level": "low/medium/high",
    "reasoning": "Final justification for this configuration choice",
    "expected_metrics": "What welfare score and metrics (time spent at queue lengths, number served) we expect"
}}"""


def _value_or_dict_to_function(
    value_or_dict: Union[float, int, dict, None], default_value: float = 0.0
) -> Callable[[int], float]:
    """
    Converts a constant value or dict to a function k -> value.

    Args:
        value_or_dict: Either a constant (float/int) or dict mapping queue_length -> value
        default_value: Default value to return if None or missing key

    Returns:
        A function that takes queue_length (int) and returns a float
    """
    if value_or_dict is None:
        return lambda k: default_value
    elif isinstance(value_or_dict, (int, float)):
        # Constant value
        constant = float(value_or_dict)
        return lambda k: constant
    elif isinstance(value_or_dict, dict):
        # Dict mapping
        def lookup_fn(k: int) -> float:
            return float(value_or_dict.get(k, value_or_dict.get(str(k), default_value)))

        return lookup_fn
    else:
        return lambda k: default_value


def _exit_dict_to_function(
    exit_dict: Union[dict, None]
) -> Callable[[int, int], tuple[float, float]]:
    """
    Converts an exit rule dict to a function (k, l) -> (rate_y, prob_z).

    Args:
        exit_dict: Dict mapping "k,l" -> [rate_y, prob_z] or None

    Returns:
        A function that takes (queue_length, position) and returns (rate_y, prob_z)
    """
    if exit_dict is None or not exit_dict:
        # No exits
        return lambda k, l: (0.0, 0.0)

    def lookup_fn(k: int, l: int) -> tuple[float, float]:
        key = f"{k},{l}"
        if key in exit_dict:
            val = exit_dict[key]
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                return (float(val[0]), float(val[1]))
        return (0.0, 0.0)

    return lookup_fn


def convert_config_to_model(
    config: Union[QueueConfigSchema, dict]
) -> CheTercieuxQueueModel:
    """
    Converts the agent's config to a CheTercieuxQueueModel.

    Args:
        config: QueueConfigSchema or dictionary with model parameters and function specs

    Returns:
        A fully constructed CheTercieuxQueueModel ready for simulation
    """
    # Convert to dict if it's a Pydantic model
    if isinstance(config, QueueConfigSchema):
        config_dict = config.model_dump()
    else:
        config_dict = config

    # Parse rate functions
    arrival_fn = _value_or_dict_to_function(
        config_dict.get("arrival_rate_fn"), default_value=1.0
    )
    service_fn = _value_or_dict_to_function(
        config_dict.get("service_rate_fn"), default_value=1.0
    )
    entry_fn = _value_or_dict_to_function(
        config_dict.get("entry_rule_fn"), default_value=1.0
    )
    exit_fn = _exit_dict_to_function(config_dict.get("exit_rule_fn"))

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

    # Construct full model
    return CheTercieuxQueueModel(
        V_surplus=float(config_dict.get("V_surplus", 10.0)),
        C_waiting_cost=float(config_dict.get("C_waiting_cost", 1.0)),
        R_provider_profit=float(config_dict.get("R_provider_profit", 5.0)),
        alpha_weight=float(config_dict.get("alpha_weight", 0.5)),
        primitive_process=primitive_process,
        design_rules=design_rules,
        queue_discipline=queue_discipline,
        information_rule=information_rule,
    )
