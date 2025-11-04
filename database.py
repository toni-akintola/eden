from typing import List


class Attempt:
    def __init__(self, attempt: int, score: int, observations: str):
        self.attempt = attempt
        self.score = score
        self.observations = observations


class Database:
    def __init__(self):
        self._data = []

    def add_attempt(self, attempt: Attempt):
        self._data.append(attempt)

    def get_attempts(self) -> List[Attempt]:
        return self._data
