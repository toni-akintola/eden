import os
from typing import List
from utils import (
    build_explore_prompt,
    build_refine_prompt,
    build_act_prompt,
    ExploreResponse,
    RefineResponse,
    ActResponse,
)
from database import Database
from openai import OpenAI


class Ensemble:
    def __init__(self, model_names: List[str], task: str):
        # Map each model to a 'function' of the model ensemble
        self.models = {
            "explore": model_names[0],
            "refine": model_names[1],
            "act": model_names[2],
        }
        self.database = Database()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.task = task

    def pipeline(self) -> dict:
        """Run the complete explore-refine-act pipeline and return the final decision"""
        exploration = self._explore()
        refinement = self._refine(exploration)
        action = self._act(refinement)
        return action

    def _explore(self) -> dict:
        """Explore phase: Analyze past configurations and generate candidate hyperparameters"""
        system_prompt = build_explore_prompt(self.task, self.database)
        response = self.client.responses.parse(
            model=self.models["explore"],
            input=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "Please analyze the past configurations and provide your exploration.",
                },
            ],
            text_format=ExploreResponse,
        )
        return response.output_parsed.model_dump()

    def _refine(self, exploration: dict) -> dict:
        """Refine phase: Optimize the exploration into an optimal hyperparameter configuration"""
        system_prompt = build_refine_prompt(exploration)
        response = self.client.responses.parse(
            model=self.models["refine"],
            input=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "Please refine the exploration into an optimal hyperparameter configuration.",
                },
            ],
            text_format=RefineResponse,
        )
        return response.output_parsed.model_dump()

    def _act(self, refinement: dict) -> dict:
        """Act phase: Make the final decision on hyperparameter configuration"""
        system_prompt = build_act_prompt(refinement)
        response = self.client.responses.parse(
            model=self.models["act"],
            input=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "Please make your final hyperparameter configuration decision.",
                },
            ],
            text_format=ActResponse,
        )
        # Return the structured response directly
        return response.output_parsed.model_dump()
