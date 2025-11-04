import os
from typing import List
from constants import build_explore_prompt, build_refine_prompt, build_act_prompt
from database import Attempt, Database
from openai import OpenAI
import json


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
        """Explore phase: Analyze past attempts and generate candidate guesses"""
        system_prompt = build_explore_prompt(self.task, self.database)
        response = self.client.chat.completions.create(
            model=self.models["explore"],
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "Please analyze the attempts and provide your exploration.",
                },
            ],
        )
        print(response.choices[0].message.content)
        return json.loads(response.choices[0].message.content)

    def _refine(self, exploration: dict) -> dict:
        """Refine phase: Optimize the exploration into a concrete strategy"""
        system_prompt = build_refine_prompt(exploration)
        response = self.client.chat.completions.create(
            model=self.models["refine"],
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "Please refine the exploration into an optimal strategy.",
                },
            ],
        )
        print(response.choices[0].message.content)
        return json.loads(response.choices[0].message.content)

    def _act(self, refinement: dict) -> dict:
        """Act phase: Make the final decision on what number to guess"""
        system_prompt = build_act_prompt(refinement)
        response = self.client.chat.completions.create(
            model=self.models["act"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Please make your final decision."},
            ],
        )
        print(response.choices[0].message.content)
        return json.loads(response.choices[0].message.content)
