from openai import OpenAI
import os
from database import Database
import json
from pydantic import BaseModel


class CurrentRange(BaseModel):
    min: int
    max: int


class CandidateGuess(BaseModel):
    value: int
    reasoning: str


class ExploreResponse(BaseModel):
    analysis: str
    current_range: CurrentRange
    pattern_insights: str
    candidate_guesses: list[CandidateGuess]
    confidence: str


class RefineResponse(BaseModel):
    evaluation: str
    optimal_strategy: str
    recommended_guess: int
    reasoning: str
    expected_outcome: str
    backup_guess: int


class ActResponse(BaseModel):
    final_guess: int
    confidence_level: str
    reasoning: str
    expected_feedback: str


def get_model_names():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return [model.id for model in client.models.list()]


def build_explore_prompt(task: str, database: Database) -> str:
    attempts_data = []
    for attempt in database.get_attempts():
        attempts_data.append(
            {
                "guess": attempt.attempt,
                "feedback": attempt.observations,
                "distance_score": attempt.score,
            }
        )

    return f"""You are an expert number-guessing strategist analyzing patterns to find a secret number.

TASK: {task}

PAST ATTEMPTS:
{json.dumps(attempts_data, indent=2) if attempts_data else "No previous attempts yet."}

Your job is to EXPLORE and analyze the search space. Based on past attempts:
1. Identify the current search boundaries (min/max possible values)
2. Analyze the pattern of feedback to narrow down the range
3. Consider what strategies have worked or failed
4. Propose 2-3 potential next guesses with reasoning

Respond in JSON format:
{{
    "analysis": "Your analysis of what we know so far",
    "current_range": {{"min": <number>, "max": <number>}},
    "pattern_insights": "Key insights from the feedback pattern",
    "candidate_guesses": [
        {{"value": <number>, "reasoning": "why this guess makes sense"}},
        {{"value": <number>, "reasoning": "why this guess makes sense"}}
    ],
    "confidence": "low/medium/high"
}}"""


def build_refine_prompt(structured_exploration: dict) -> str:
    return f"""You are a strategic optimizer that refines number-guessing strategies.

EXPLORATION ANALYSIS:
{json.dumps(structured_exploration, indent=2)}

Your job is to REFINE the exploration into an optimal strategy:

Respond in JSON format:
{{
    "evaluation": "Assessment of the exploration analysis",
    "optimal_strategy": "Binary search / Linear search / Smart elimination / etc.",
    "recommended_guess": <number>,
    "reasoning": "Detailed explanation of why this is the best guess",
    "expected_outcome": "What we expect to learn from this guess",
    "backup_guess": <number>
}}"""


def build_act_prompt(structured_refinement: dict) -> str:
    return f"""You are the final decision maker that commits to a guess.

REFINED STRATEGY:
{json.dumps(structured_refinement, indent=2)}

Your job is to ACT and make the final decision:
1. Review the recommended guess
2. Do a final sanity check
3. Commit to a specific number to guess

Respond in JSON format:
{{
    "final_guess": <number>,
    "confidence_level": "low/medium/high",
    "reasoning": "Final justification for this guess",
    "expected_feedback": "What feedback we expect to receive"
}}"""
