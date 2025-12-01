from typing import Annotated
from openai import OpenAI
import os
from database import Database
import json
from pydantic import BaseModel


class ExploreResponse(BaseModel):
    analysis: str
    pattern_insights: str
    candidate_configs: list[
        dict
    ]  # Simplified dict representation that can be converted to CheTercieuxQueueModel
    confidence: str
    key_learnings: str


class RefineResponse(BaseModel):
    evaluation: str
    optimal_strategy: str
    recommended_config: dict  # Simplified dict representation that can be converted to CheTercieuxQueueModel
    reasoning: str
    expected_outcome: str
    alternative_config: dict  # Simplified dict representation that can be converted to CheTercieuxQueueModel


class ActResponse(BaseModel):
    final_config: dict  # Simplified dict representation that can be converted to CheTercieuxQueueModel
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
