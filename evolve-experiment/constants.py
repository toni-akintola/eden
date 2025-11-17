from openai import OpenAI
import os
from database import Database
import json
from pydantic import BaseModel


class HyperparameterConfig(BaseModel):
    num_servers: int
    service_rate: float
    queue_discipline: str  # "FIFO", "LIFO", or "PRIORITY"
    reasoning: str


class ExploreResponse(BaseModel):
    analysis: str
    pattern_insights: str
    candidate_configs: list[HyperparameterConfig]
    confidence: str
    key_learnings: str


class RefineResponse(BaseModel):
    evaluation: str
    optimal_strategy: str
    recommended_config: HyperparameterConfig
    reasoning: str
    expected_outcome: str
    alternative_config: HyperparameterConfig


class ActResponse(BaseModel):
    final_config: HyperparameterConfig
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

    return f"""You are an expert queuing systems analyst optimizing hyperparameters for a multi-server queue system.

TASK: {task}

PAST CONFIGURATIONS TESTED:
{json.dumps(attempts_data, indent=2) if attempts_data else "No previous configurations tested yet."}

Your job is to EXPLORE and analyze the hyperparameter space. Based on past configurations:
1. Analyze which hyperparameter combinations performed well or poorly
2. Identify patterns in efficiency scores and feedback
3. Consider queuing theory principles (e.g., stability: arrival_rate < num_servers * service_rate)
4. Propose 2-3 candidate hyperparameter configurations with reasoning

HYPERPARAMETERS TO OPTIMIZE:
- num_servers: integer (number of parallel servers, typically 1-10)
- service_rate: float (service rate per server, typically 0.1-5.0)
- queue_discipline: string ("FIFO", "LIFO", or "PRIORITY")

EFFICIENCY METRICS (lower score is better):
- Average wait time (primary)
- Average queue length
- Server utilization (target ~80%)

Respond in JSON format:
{{
    "analysis": "Your analysis of what configurations worked well and why",
    "pattern_insights": "Key insights from the efficiency scores and feedback patterns",
    "candidate_configs": [
        {{
            "num_servers": <integer>,
            "service_rate": <float>,
            "queue_discipline": "<FIFO|LIFO|PRIORITY>",
            "reasoning": "why this configuration might work well"
        }},
        {{
            "num_servers": <integer>,
            "service_rate": <float>,
            "queue_discipline": "<FIFO|LIFO|PRIORITY>",
            "reasoning": "alternative approach"
        }}
    ],
    "confidence": "low/medium/high",
    "key_learnings": "What you've learned about the optimal hyperparameter space"
}}"""


def build_refine_prompt(structured_exploration: dict) -> str:
    return f"""You are a queuing systems optimizer that refines hyperparameter configurations.

EXPLORATION ANALYSIS:
{json.dumps(structured_exploration, indent=2)}

Your job is to REFINE the exploration into an optimal hyperparameter configuration:
1. Evaluate the candidate configurations from exploration
2. Consider queuing theory (Little's Law, stability conditions, etc.)
3. Balance wait time, queue length, and server utilization
4. Select the best configuration and provide an alternative

Respond in JSON format:
{{
    "evaluation": "Assessment of the exploration analysis and candidate configurations",
    "optimal_strategy": "Description of the optimization approach (e.g., 'Balance capacity and utilization', 'Minimize wait time while maintaining stability')",
    "recommended_config": {{
        "num_servers": <integer>,
        "service_rate": <float>,
        "queue_discipline": "<FIFO|LIFO|PRIORITY>",
        "reasoning": "Detailed explanation of why this is the best configuration"
    }},
    "reasoning": "Detailed explanation of the optimization strategy",
    "expected_outcome": "What efficiency metrics we expect from this configuration",
    "alternative_config": {{
        "num_servers": <integer>,
        "service_rate": <float>,
        "queue_discipline": "<FIFO|LIFO|PRIORITY>",
        "reasoning": "Backup configuration if recommended one doesn't work well"
    }}
}}"""


def build_act_prompt(structured_refinement: dict) -> str:
    return f"""You are the final decision maker that commits to a hyperparameter configuration.

REFINED STRATEGY:
{json.dumps(structured_refinement, indent=2)}

Your job is to ACT and make the final decision:
1. Review the recommended configuration
2. Do a final sanity check (ensure stability: arrival_rate < num_servers * service_rate)
3. Commit to specific hyperparameter values

Respond in JSON format:
{{
    "final_config": {{
        "num_servers": <integer>,
        "service_rate": <float>,
        "queue_discipline": "<FIFO|LIFO|PRIORITY>",
        "reasoning": "Final justification for this configuration"
    }},
    "confidence_level": "low/medium/high",
    "reasoning": "Final justification for this configuration choice",
    "expected_metrics": "What efficiency metrics (wait time, queue length, utilization) we expect"
}}"""
