from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
import random
import uuid
import threading
import heapq


@dataclass
class Organism:
    """Represents a program (entry/exit rules and queue discipline) in the evolutionary population."""

    entry_rule_code: str
    exit_rule_code: str
    queue_discipline: str  # "FCFS", "LIFO", or "SIRO"
    information_rule: str = (
        "NO_INFORMATION"  # "NO_INFORMATION", "FULL_INFORMATION", or "COARSE_INFORMATION"
    )
    generation: int = 0
    fitness: Optional[float] = None
    parent_id: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    # Cached simulation results to avoid re-running expensive simulations
    cached_simulation_results: Optional[Any] = field(default=None, repr=False)


class Database:
    """
    Thread-safe population database with fitness-weighted sampling.

    Optimized for large populations (20,000+ organisms) with:
    - Incremental weight caching to avoid O(N) recalculation
    - Heap-based top-k tracking for O(log k) inspiration lookup
    - Population pruning to limit memory usage
    """

    def __init__(self, max_population: int = 10000, prune_keep_ratio: float = 0.5):
        """
        Initialize the database.

        Args:
            max_population: Maximum population size before pruning triggers
            prune_keep_ratio: Fraction of population to keep when pruning (by fitness)
        """
        self._organisms: List[Organism] = []
        self._lock = threading.Lock()

        # Population management
        self._max_population = max_population
        self._prune_keep_ratio = prune_keep_ratio

        # Cached statistics for incremental weight calculation
        self._cached_min_fitness: Optional[float] = None
        self._cached_max_fitness: Optional[float] = None
        self._cached_min_gen: Optional[int] = None
        self._cached_max_gen: Optional[int] = None
        self._cache_valid = False

        # Heap for top-k tracking (stores (-fitness, id, organism) for max-heap behavior)
        self._top_k_heap: List[tuple] = []
        self._top_k_size = 10  # Track top 10 for inspirations
        self._top_k_map: Dict[str, Organism] = {}  # id -> organism for O(1) lookup

    def add(self, organism: Organism):
        """Add an organism to the population (thread-safe)."""
        with self._lock:
            self._organisms.append(organism)
            self._update_caches_on_add(organism)
            self._update_top_k_on_add(organism)

            # Trigger pruning if population exceeds limit
            if len(self._organisms) > self._max_population:
                self._prune_population_unlocked()

    def _update_caches_on_add(self, organism: Organism):
        """Incrementally update cached statistics when adding an organism."""
        fitness = organism.fitness if organism.fitness is not None else 0.0
        gen = organism.generation

        if not self._cache_valid or len(self._organisms) == 1:
            # First organism or cache invalidated - initialize
            self._cached_min_fitness = fitness
            self._cached_max_fitness = fitness
            self._cached_min_gen = gen
            self._cached_max_gen = gen
            self._cache_valid = True
        else:
            # Incremental update
            self._cached_min_fitness = min(self._cached_min_fitness, fitness)
            self._cached_max_fitness = max(self._cached_max_fitness, fitness)
            self._cached_min_gen = min(self._cached_min_gen, gen)
            self._cached_max_gen = max(self._cached_max_gen, gen)

    def _update_top_k_on_add(self, organism: Organism):
        """Update the top-k heap when adding an organism."""
        if organism.fitness is None:
            return

        # Use negative fitness for max-heap behavior with Python's min-heap
        entry = (-organism.fitness, organism.id, organism)

        if len(self._top_k_heap) < self._top_k_size:
            heapq.heappush(self._top_k_heap, entry)
            self._top_k_map[organism.id] = organism
        elif organism.fitness > -self._top_k_heap[0][0]:
            # New organism is better than worst in top-k
            old_entry = heapq.heapreplace(self._top_k_heap, entry)
            del self._top_k_map[old_entry[1]]
            self._top_k_map[organism.id] = organism

    def _prune_population_unlocked(self):
        """
        Prune population to keep only the best organisms (assumes lock held).

        Keeps the top `prune_keep_ratio` fraction by fitness, plus all organisms
        from the most recent generation to preserve diversity.
        """
        if len(self._organisms) <= 1:
            return

        keep_count = int(len(self._organisms) * self._prune_keep_ratio)
        keep_count = max(keep_count, 100)  # Always keep at least 100

        # Sort by fitness (descending)
        evaluated = [
            (o, o.fitness if o.fitness is not None else float("-inf"))
            for o in self._organisms
        ]
        evaluated.sort(key=lambda x: x[1], reverse=True)

        # Keep top performers
        kept = set()
        for o, _ in evaluated[:keep_count]:
            kept.add(o.id)

        # Also keep all organisms from the most recent generation
        max_gen = max(o.generation for o in self._organisms)
        for o in self._organisms:
            if o.generation == max_gen:
                kept.add(o.id)

        # Filter organisms
        self._organisms = [o for o in self._organisms if o.id in kept]

        # Invalidate and rebuild caches
        self._cache_valid = False
        self._rebuild_caches_unlocked()
        self._rebuild_top_k_unlocked()

    def _rebuild_caches_unlocked(self):
        """Rebuild all caches from scratch (assumes lock held)."""
        if not self._organisms:
            self._cache_valid = False
            return

        fitnesses = [
            o.fitness if o.fitness is not None else 0.0 for o in self._organisms
        ]
        generations = [o.generation for o in self._organisms]

        self._cached_min_fitness = min(fitnesses)
        self._cached_max_fitness = max(fitnesses)
        self._cached_min_gen = min(generations)
        self._cached_max_gen = max(generations)
        self._cache_valid = True

    def _rebuild_top_k_unlocked(self):
        """Rebuild top-k heap from scratch (assumes lock held)."""
        self._top_k_heap = []
        self._top_k_map = {}

        evaluated = [o for o in self._organisms if o.fitness is not None]
        evaluated.sort(key=lambda o: o.fitness, reverse=True)

        for o in evaluated[: self._top_k_size]:
            entry = (-o.fitness, o.id, o)
            heapq.heappush(self._top_k_heap, entry)
            self._top_k_map[o.id] = o

    def sample(
        self, fitness_weight: float = 0.7, recency_weight: float = 0.3
    ) -> tuple[Organism, List[Organism]]:
        """Sample a parent organism and inspirations using weighted selection (thread-safe)."""
        with self._lock:
            if not self._organisms:
                raise ValueError("Database is empty")

            weights = self._calculate_weights_cached(fitness_weight, recency_weight)
            parent = random.choices(self._organisms, weights=weights, k=1)[0]
            inspirations = self._get_inspirations_from_heap(exclude_id=parent.id)

            return parent, inspirations

    def _calculate_weights_cached(
        self, fitness_weight: float, recency_weight: float
    ) -> List[float]:
        """Calculate sampling weights using cached min/max values (assumes lock held)."""
        n = len(self._organisms)
        if n == 1:
            return [1.0]

        # Use cached bounds for normalization
        if not self._cache_valid:
            self._rebuild_caches_unlocked()

        min_f, max_f = self._cached_min_fitness, self._cached_max_fitness
        min_g, max_g = self._cached_min_gen, self._cached_max_gen

        fitness_range = max_f - min_f if max_f > min_f else 1.0
        gen_range = max_g - min_g if max_g > min_g else 1.0

        weights = []
        for o in self._organisms:
            f = o.fitness if o.fitness is not None else 0.0
            g = o.generation

            # Normalize using cached bounds
            if max_f > min_f:
                norm_f = (f - min_f) / fitness_range
            else:
                norm_f = 1.0

            if max_g > min_g:
                norm_r = (g - min_g) / gen_range
            else:
                norm_r = 1.0

            weight = fitness_weight * norm_f + recency_weight * norm_r + 0.1
            weights.append(weight)

        return weights

    def _get_inspirations_from_heap(
        self, exclude_id: Optional[str] = None, k: int = 3
    ) -> List[Organism]:
        """Get top k organisms by fitness using the heap (assumes lock held)."""
        # Get from cached top-k, excluding the parent
        result = []
        for _, org_id, org in sorted(self._top_k_heap):
            if org_id != exclude_id:
                result.append(org)
                if len(result) >= k:
                    break

        # If we don't have enough from heap (rare), fall back to scan
        if len(result) < k:
            candidates = [
                o
                for o in self._organisms
                if o.id != exclude_id
                and o.fitness is not None
                and o.id not in {r.id for r in result}
            ]
            candidates.sort(key=lambda o: o.fitness or 0, reverse=True)
            result.extend(candidates[: k - len(result)])

        return result

    def _get_inspirations_unlocked(
        self, k: int = 3, exclude_id: Optional[str] = None
    ) -> List[Organism]:
        """Get top k organisms by fitness for inspiration (assumes lock held)."""
        return self._get_inspirations_from_heap(exclude_id, k)

    def get_inspirations(
        self, k: int = 3, exclude_id: Optional[str] = None
    ) -> List[Organism]:
        """Get top k organisms by fitness for inspiration (thread-safe)."""
        with self._lock:
            return self._get_inspirations_unlocked(k, exclude_id)

    def get_best(self) -> Optional[Organism]:
        """Get the organism with highest fitness (thread-safe)."""
        with self._lock:
            # Use top-k heap for O(1) lookup
            if self._top_k_heap:
                # Heap stores (-fitness, id, organism), so first element has highest fitness
                return min(self._top_k_heap, key=lambda x: x[0])[2]

            # Fallback if heap is empty
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

    def get_stats(self) -> dict:
        """Get database statistics (thread-safe)."""
        with self._lock:
            return {
                "population_size": len(self._organisms),
                "max_population": self._max_population,
                "min_fitness": self._cached_min_fitness,
                "max_fitness": self._cached_max_fitness,
                "min_generation": self._cached_min_gen,
                "max_generation": self._cached_max_gen,
                "top_k_size": len(self._top_k_heap),
            }
