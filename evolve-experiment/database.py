from typing import List, Dict, Any


class Attempt:
    def __init__(self, attempt: Dict[str, Any], score: float, observations: str):
        """
        Represents an attempt with hyperparameters.

        Args:
            attempt: Dictionary with hyperparameters (num_servers, service_rate, queue_discipline)
            score: Efficiency score (lower is better)
            observations: Human-readable feedback about the configuration
        """
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
