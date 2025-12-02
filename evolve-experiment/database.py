from dataclasses import dataclass, field
from typing import List, Optional
import random
import uuid
import threading


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
    """Thread-safe population database with fitness-weighted sampling."""

    def __init__(self):
        self._organisms: List[Organism] = []
        self._lock = threading.Lock()

    def add(self, organism: Organism):
        """Add an organism to the population (thread-safe)."""
        with self._lock:
            self._organisms.append(organism)

    def sample(
        self, fitness_weight: float = 0.7, recency_weight: float = 0.3
    ) -> tuple[Organism, List[Organism]]:
        """Sample a parent organism and inspirations using weighted selection (thread-safe)."""
        with self._lock:
            if not self._organisms:
                raise ValueError("Database is empty")

            weights = self._calculate_weights(fitness_weight, recency_weight)
            parent = random.choices(self._organisms, weights=weights, k=1)[0]
            inspirations = self._get_inspirations_unlocked(k=3, exclude_id=parent.id)

            return parent, inspirations

    def _calculate_weights(
        self, fitness_weight: float, recency_weight: float
    ) -> List[float]:
        """Calculate sampling weights balancing fitness and recency (assumes lock held)."""
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

    def _get_inspirations_unlocked(
        self, k: int = 3, exclude_id: Optional[str] = None
    ) -> List[Organism]:
        """Get top k organisms by fitness for inspiration (assumes lock held)."""
        candidates = [
            o for o in self._organisms if o.id != exclude_id and o.fitness is not None
        ]
        if not candidates:
            return []

        sorted_candidates = sorted(
            candidates, key=lambda o: o.fitness or 0, reverse=True
        )
        return sorted_candidates[:k]

    def get_inspirations(
        self, k: int = 3, exclude_id: Optional[str] = None
    ) -> List[Organism]:
        """Get top k organisms by fitness for inspiration (thread-safe)."""
        with self._lock:
            return self._get_inspirations_unlocked(k, exclude_id)

    def get_best(self) -> Optional[Organism]:
        """Get the organism with highest fitness (thread-safe)."""
        with self._lock:
            evaluated = [o for o in self._organisms if o.fitness is not None]
            if not evaluated:
                return None
            return max(evaluated, key=lambda o: o.fitness)

    def size(self) -> int:
        """Get the number of organisms (thread-safe)."""
        with self._lock:
            return len(self._organisms)

    def all(self) -> List[Organism]:
        """Get a copy of all organisms (thread-safe)."""
        with self._lock:
            return self._organisms.copy()
