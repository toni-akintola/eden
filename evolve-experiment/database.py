from dataclasses import dataclass, field
from typing import List, Optional
import random
import uuid


@dataclass
class Organism:
    """Represents a program (entry/exit rules and queue discipline) in the evolutionary population."""

    entry_rule_code: str
    exit_rule_code: str
    queue_discipline: str  # "FCFS", "LIFO", or "SIRO"
    generation: int = 0
    fitness: Optional[float] = None
    parent_id: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


class Database:
    """Population database with fitness-weighted sampling."""

    def __init__(self):
        self._organisms: List[Organism] = []

    def add(self, organism: Organism):
        """Add an organism to the population."""
        self._organisms.append(organism)

    def sample(
        self, fitness_weight: float = 0.7, recency_weight: float = 0.3
    ) -> tuple[Organism, List[Organism]]:
        """Sample a parent organism and inspirations using weighted selection."""
        if not self._organisms:
            raise ValueError("Database is empty")

        weights = self._calculate_weights(fitness_weight, recency_weight)
        parent = random.choices(self._organisms, weights=weights, k=1)[0]
        inspirations = self.get_inspirations(k=3, exclude_id=parent.id)

        return parent, inspirations

    def _calculate_weights(
        self, fitness_weight: float, recency_weight: float
    ) -> List[float]:
        """Calculate sampling weights balancing fitness and recency."""
        if len(self._organisms) == 1:
            return [1.0]

        fitnesses = [
            o.fitness if o.fitness is not None else 0.0 for o in self._organisms
        ]
        generations = [o.generation for o in self._organisms]

        min_f, max_f = min(fitnesses), max(fitnesses)
        if max_f > min_f:
            norm_fitness = [(f - min_f) / (max_f - min_f) for f in fitnesses]
        else:
            norm_fitness = [1.0] * len(fitnesses)

        min_g, max_g = min(generations), max(generations)
        if max_g > min_g:
            norm_recency = [(g - min_g) / (max_g - min_g) for g in generations]
        else:
            norm_recency = [1.0] * len(generations)

        weights = [
            fitness_weight * f + recency_weight * r + 0.1
            for f, r in zip(norm_fitness, norm_recency)
        ]

        return weights

    def get_inspirations(
        self, k: int = 3, exclude_id: Optional[str] = None
    ) -> List[Organism]:
        """Get top k organisms by fitness for inspiration."""
        candidates = [
            o for o in self._organisms if o.id != exclude_id and o.fitness is not None
        ]
        if not candidates:
            return []

        sorted_candidates = sorted(
            candidates, key=lambda o: o.fitness or 0, reverse=True
        )
        return sorted_candidates[:k]

    def get_best(self) -> Optional[Organism]:
        """Get the organism with highest fitness."""
        evaluated = [o for o in self._organisms if o.fitness is not None]
        if not evaluated:
            return None
        return max(evaluated, key=lambda o: o.fitness)

    def size(self) -> int:
        return len(self._organisms)

    def all(self) -> List[Organism]:
        return self._organisms.copy()
