from typing import Annotated, Union
from openai import OpenAI
import os
from database import Database
import json
from pydantic import BaseModel
from evolve_types import (
    CheTercieuxQueueModel,
    PrimitiveProcess,
    EntryExitRule,
    QueueDiscipline,
    InformationRule,
)

__all__ = [
    "QueueModelConfig",
    "ExploreResponse",
    "RefineResponse",
    "ActResponse",
    "parse_code_to_function",
    "convert_agent_config_to_model",
    "get_model_names",
    "build_explore_prompt",
    "build_refine_prompt",
    "build_act_prompt",
]


def parse_code_to_function(code_str: str):
    """
    Parses a string containing a Python lambda or function definition into a callable.
    WARNING: Uses eval(), so only use with trusted/generated code.
    """
    if not code_str:
        return None

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
        # Try eval first (for lambdas)
        return eval(clean_code)
    except SyntaxError:
        # If that fails, it might be a def (though eval usually handles lambdas fine)
        # For 'def', we'd need exec.
        local_scope = {}
        exec(clean_code, {}, local_scope)
        # Assume the last defined function is the one we want
        if local_scope:
            return list(local_scope.values())[-1]
        raise ValueError(f"Could not parse code: {code_str}")


def convert_agent_config_to_model(
    config: Union["QueueModelConfig", dict]
) -> CheTercieuxQueueModel:
    """
    Converts the config from the agent (with code strings)
    into a fully typed CheTercieuxQueueModel.

    Args:
        config: Either a QueueModelConfig or a dict with the same structure
    """
    # Handle both dict and QueueModelConfig
    if isinstance(config, QueueModelConfig):
        config_dict = config.model_dump()
    else:
        config_dict = config

    # 1. Parse Functions
    # Note: Agent might return None or empty string for exit_rule_code
    arrival_fn = parse_code_to_function(config_dict.get("arrival_rate_code"))
    service_fn = parse_code_to_function(config_dict.get("service_rate_code"))
    entry_fn = parse_code_to_function(config_dict.get("entry_rule_code"))
    exit_fn = parse_code_to_function(config_dict.get("exit_rule_code"))

    # 2. Construct PrimitiveProcess
    primitive_process = PrimitiveProcess(
        arrival_rate_fn=arrival_fn, service_rate_fn=service_fn
    )

    # 3. Construct EntryExitRule
    design_rules = EntryExitRule(entry_rule_fn=entry_fn, exit_rule_fn=exit_fn)

    # 4. Parse Enums
    discipline_str = config_dict.get("queue_discipline", "FCFS")
    try:
        queue_discipline = QueueDiscipline[discipline_str]
    except KeyError:
        if "FCFS" in discipline_str:
            queue_discipline = QueueDiscipline.FCFS
        elif "LIFO" in discipline_str:
            queue_discipline = QueueDiscipline.LIFO
        elif "SIRO" in discipline_str or "RANDOM" in discipline_str.upper():
            queue_discipline = QueueDiscipline.SIRO
        else:
            queue_discipline = QueueDiscipline.FCFS

    info_str = config_dict.get("information_rule", "NO_INFORMATION_BEYOND_REC")
    try:
        information_rule = InformationRule[info_str]
    except KeyError:
        if "NO_INFO" in info_str:
            information_rule = InformationRule.NO_INFORMATION_BEYOND_REC
        elif "FULL" in info_str:
            information_rule = InformationRule.FULL_INFORMATION
        elif "COARSE" in info_str:
            information_rule = InformationRule.COARSE_INFORMATION
        else:
            information_rule = InformationRule.NO_INFORMATION_BEYOND_REC

    # 5. Construct Full Model
    return CheTercieuxQueueModel(
        V_surplus=float(config_dict.get("V_surplus", 10.0)),
        C_waiting_cost=float(config_dict.get("C_waiting_cost", 1.0)),
        R_provider_profit=float(config_dict.get("R_provider_profit", 1.0)),
        alpha_weight=float(config_dict.get("alpha_weight", 0.5)),
        primitive_process=primitive_process,
        design_rules=design_rules,
        queue_discipline=queue_discipline,
        information_rule=information_rule,
    )


class QueueModelConfig(BaseModel):
    """Configuration for CheTercieuxQueueModel with code strings for functions."""

    V_surplus: float
    C_waiting_cost: float
    R_provider_profit: float
    alpha_weight: float
    arrival_rate_code: str
    service_rate_code: str
    entry_rule_code: str
    exit_rule_code: str
    queue_discipline: str
    information_rule: str
    reasoning: str = ""


class ExploreResponse(BaseModel):
    analysis: str
    pattern_insights: str
    candidate_configs: list[QueueModelConfig]
    confidence: str
    key_learnings: str


class RefineResponse(BaseModel):
    evaluation: str
    optimal_strategy: str
    recommended_config: QueueModelConfig
    reasoning: str
    expected_outcome: str
    alternative_config: QueueModelConfig


class ActResponse(BaseModel):
    final_config: QueueModelConfig
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
                "hyperparameters": attempt.attempt,  # This will be a dict with code strings
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
- arrival_rate_code: string (Python code for function k -> arrival_rate)
- service_rate_code: string (Python code for function k -> service_rate)
- entry_rule_code: string (Python code for function k -> entry_probability [0, 1])
- exit_rule_code: string (Python code for function (k, l) -> (rate_y, prob_z))
- queue_discipline: string ("FCFS", "LIFO", or "SIRO")
- information_rule: string ("NO_INFORMATION_BEYOND_REC", "FULL_INFORMATION", or "COARSE_INFORMATION")

NOTE: For functions, provide valid Python lambda expressions or one-line functions as strings.
Examples:
- arrival_rate_code: "lambda k: 2.0"
- service_rate_code: "lambda k: 2.0 if k < 5 else 3.0"
- entry_rule_code: "lambda k: 1.0 / (1.0 + k)"
- exit_rule_code: "lambda k, l: (0.5, 0.1)" (returns tuple (rate_y, prob_z))
- exit_rule_code: "lambda k, l: (0.0, 0.0)" (for no exits)

OBJECTIVE: Maximize designer's welfare score W (higher is better), which balances provider profit and agent utility.

Respond in JSON format:
{{
    "analysis": "Your analysis...",
    "pattern_insights": "Key insights...",
    "candidate_configs": [
        {{
            "V_surplus": <float>,
            "C_waiting_cost": <float>,
            "R_provider_profit": <float>,
            "alpha_weight": <float in [0,1]>,
            "arrival_rate_code": "lambda k: ...",
            "service_rate_code": "lambda k: ...",
            "entry_rule_code": "lambda k: ...",
            "exit_rule_code": "lambda k, l: ...",
            "queue_discipline": "<FCFS|LIFO|SIRO>",
            "information_rule": "<NO_INFORMATION_BEYOND_REC|FULL_INFORMATION|COARSE_INFORMATION>",
            "reasoning": "why this configuration might work well"
        }},
        ...
    ],
    "confidence": "...",
    "key_learnings": "..."
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

NOTE: For functions, provide valid Python lambda expressions as strings.
- arrival/service/entry: lambda k: value
- exit: lambda k, l: (rate_y, prob_z)

Respond in JSON format:
{{
    "evaluation": "Assessment...",
    "optimal_strategy": "Description...",
    "recommended_config": {{
        "V_surplus": <float>,
        "C_waiting_cost": <float>,
        "R_provider_profit": <float>,
        "alpha_weight": <float in [0,1]>,
        "arrival_rate_code": "lambda k: ...",
        "service_rate_code": "lambda k: ...",
        "entry_rule_code": "lambda k: ...",
        "exit_rule_code": "lambda k, l: ...",
        "queue_discipline": "<FCFS|LIFO|SIRO>",
        "information_rule": "<NO_INFORMATION_BEYOND_REC|FULL_INFORMATION|COARSE_INFORMATION>",
        "reasoning": "Detailed explanation"
    }},
    "reasoning": "Strategy explanation",
    "expected_outcome": "Expected metrics",
    "alternative_config": {{
        ...
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

NOTE: For functions, provide valid Python lambda expressions as strings.

Respond in JSON format:
{{
    "final_config": {{
        "V_surplus": <float>,
        "C_waiting_cost": <float>,
        "R_provider_profit": <float>,
        "alpha_weight": <float in [0,1]>,
        "arrival_rate_code": "lambda k: ...",
        "service_rate_code": "lambda k: ...",
        "entry_rule_code": "lambda k: ...",
        "exit_rule_code": "lambda k, l: ...",
        "queue_discipline": "<FCFS|LIFO|SIRO>",
        "information_rule": "<NO_INFORMATION_BEYOND_REC|FULL_INFORMATION|COARSE_INFORMATION>",
        "reasoning": "Final justification"
    }},
    "confidence_level": "low/medium/high",
    "reasoning": "Justification",
    "expected_metrics": "Expected welfare/metrics"
}}"""
