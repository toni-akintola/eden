from typing import List, Tuple


class Evaluator:
    def __init__(self, answer: int = 42):
        self.answer = answer

    def evaluate(self, guess: int) -> int:
        """
        Evaluate the guess and return the distance between the guess and the answer.
        """
        return (guess - self.answer) ** 2
