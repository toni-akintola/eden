"""
Population database for evolved auction mechanisms (mutator.Organism).

Thread-safe store with fitness-weighted parent sampling and top-k elites
for mutation context — same structural pattern as evolve-experiment/database.py,
but keyed to alloc/pay lambda organisms instead of queue programs.
"""

from __future__ import annotations

import heapq
import logging
import random
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mutator import Organism

logger = logging.getLogger(__name__)


@dataclass
class Database:
    """
    Thread-safe population of auction mechanisms with fitness-weighted sampling.

    Organisms are mutator.Organism instances; fitness must be set before add()
    if they should participate in top-k and sampling weights (unevaluated
    organisms are still stored but weighted like fitness 0.0 in caches).
    """

    max_population: int = 10_000
    prune_keep_ratio: float = 0.5

    def __post_init__(self) -> None:
        self._organisms: List[Organism] = []
        self._lock = threading.Lock()
        self._cached_min_fitness: Optional[float] = None
        self._cached_max_fitness: Optional[float] = None
        self._cached_min_gen: Optional[int] = None
        self._cached_max_gen: Optional[int] = None
        self._cache_valid = False
        self._top_k_heap: List[tuple] = []
        self._top_k_size = 10
        self._top_k_map: Dict[str, Organism] = {}

    def add(self, organism: Organism) -> None:
        with self._lock:
            self._organisms.append(organism)
            self._update_caches_on_add(organism)
            self._update_top_k_on_add(organism)
            pop = len(self._organisms)
            fit = organism.fitness
            fit_s = f"{fit:.4f}" if fit is not None else "None"
            logger.debug(
                "add id=%s gen=%s fitness=%s population=%d/%d",
                organism.id,
                organism.generation,
                fit_s,
                pop,
                self.max_population,
            )
            if pop > self.max_population:
                self._prune_population_unlocked()

    def _fitness_value(self, organism: Organism) -> float:
        f = organism.fitness
        return float(f) if f is not None else 0.0

    def _update_caches_on_add(self, organism: Organism) -> None:
        fitness = self._fitness_value(organism)
        gen = organism.generation

        if not self._cache_valid or len(self._organisms) == 1:
            self._cached_min_fitness = fitness
            self._cached_max_fitness = fitness
            self._cached_min_gen = gen
            self._cached_max_gen = gen
            self._cache_valid = True
        else:
            assert self._cached_min_fitness is not None
            assert self._cached_max_fitness is not None
            assert self._cached_min_gen is not None
            assert self._cached_max_gen is not None
            self._cached_min_fitness = min(self._cached_min_fitness, fitness)
            self._cached_max_fitness = max(self._cached_max_fitness, fitness)
            self._cached_min_gen = min(self._cached_min_gen, gen)
            self._cached_max_gen = max(self._cached_max_gen, gen)

    def _update_top_k_on_add(self, organism: Organism) -> None:
        if organism.fitness is None:
            return
        entry = (-float(organism.fitness), organism.id, organism)
        if len(self._top_k_heap) < self._top_k_size:
            heapq.heappush(self._top_k_heap, entry)
            self._top_k_map[organism.id] = organism
        elif float(organism.fitness) > -self._top_k_heap[0][0]:
            old_entry = heapq.heapreplace(self._top_k_heap, entry)
            del self._top_k_map[old_entry[1]]
            self._top_k_map[organism.id] = organism

    def _prune_population_unlocked(self) -> None:
        before = len(self._organisms)
        if before <= 1:
            return

        keep_count = max(int(before * self.prune_keep_ratio), 100)
        evaluated = [(o, self._fitness_value(o)) for o in self._organisms]
        evaluated.sort(key=lambda x: x[1], reverse=True)

        kept: set[str] = set()
        for o, _ in evaluated[:keep_count]:
            kept.add(o.id)

        max_gen = max(o.generation for o in self._organisms)
        for o in self._organisms:
            if o.generation == max_gen:
                kept.add(o.id)

        self._organisms = [o for o in self._organisms if o.id in kept]
        after = len(self._organisms)
        logger.info(
            "pruned population %d -> %d (cap=%d keep_ratio=%.2f keep_count=%d)",
            before,
            after,
            self.max_population,
            self.prune_keep_ratio,
            keep_count,
        )
        self._cache_valid = False
        self._rebuild_caches_unlocked()
        self._rebuild_top_k_unlocked()

    def _rebuild_caches_unlocked(self) -> None:
        if not self._organisms:
            self._cache_valid = False
            return
        fitnesses = [self._fitness_value(o) for o in self._organisms]
        generations = [o.generation for o in self._organisms]
        self._cached_min_fitness = min(fitnesses)
        self._cached_max_fitness = max(fitnesses)
        self._cached_min_gen = min(generations)
        self._cached_max_gen = max(generations)
        self._cache_valid = True

    def _rebuild_top_k_unlocked(self) -> None:
        self._top_k_heap = []
        self._top_k_map = {}
        evaluated = [o for o in self._organisms if o.fitness is not None]
        evaluated.sort(key=lambda o: float(o.fitness), reverse=True)
        for o in evaluated[: self._top_k_size]:
            entry = (-float(o.fitness), o.id, o)
            heapq.heappush(self._top_k_heap, entry)
            self._top_k_map[o.id] = o

    def sample(
        self,
        fitness_weight: float = 0.7,
        recency_weight: float = 0.3,
    ) -> tuple[Organism, List[Organism]]:
        with self._lock:
            if not self._organisms:
                raise ValueError("Database is empty")
            weights = self._calculate_weights_cached(fitness_weight, recency_weight)
            parent = random.choices(self._organisms, weights=weights, k=1)[0]
            inspirations = self._get_inspirations_from_heap(exclude_id=parent.id)
            logger.debug(
                "sample parent=%s inspirations=%s (n=%d organisms)",
                parent.id,
                [o.id for o in inspirations],
                len(self._organisms),
            )
            return parent, inspirations

    def _calculate_weights_cached(
        self,
        fitness_weight: float,
        recency_weight: float,
    ) -> List[float]:
        n = len(self._organisms)
        if n == 1:
            return [1.0]
        if not self._cache_valid:
            self._rebuild_caches_unlocked()
        assert self._cached_min_fitness is not None
        assert self._cached_max_fitness is not None
        assert self._cached_min_gen is not None
        assert self._cached_max_gen is not None

        min_f, max_f = self._cached_min_fitness, self._cached_max_fitness
        min_g, max_g = self._cached_min_gen, self._cached_max_gen
        fitness_range = max_f - min_f if max_f > min_f else 1.0
        gen_range = max_g - min_g if max_g > min_g else 1.0

        weights: List[float] = []
        for o in self._organisms:
            f = self._fitness_value(o)
            g = o.generation
            norm_f = (f - min_f) / fitness_range if max_f > min_f else 1.0
            norm_r = (g - min_g) / gen_range if max_g > min_g else 1.0
            weights.append(fitness_weight * norm_f + recency_weight * norm_r + 0.1)
        return weights

    def _get_inspirations_from_heap(
        self,
        exclude_id: Optional[str] = None,
        k: int = 3,
    ) -> List[Organism]:
        result: List[Organism] = []
        for _, org_id, org in sorted(self._top_k_heap):
            if org_id != exclude_id:
                result.append(org)
                if len(result) >= k:
                    break
        if len(result) < k:
            candidates = [
                o
                for o in self._organisms
                if o.id != exclude_id
                and o.fitness is not None
                and o.id not in {r.id for r in result}
            ]
            candidates.sort(key=lambda o: float(o.fitness or 0.0), reverse=True)
            result.extend(candidates[: k - len(result)])
        return result

    def get_inspirations(
        self,
        k: int = 3,
        exclude_id: Optional[str] = None,
    ) -> List[Organism]:
        with self._lock:
            return self._get_inspirations_from_heap(exclude_id, k)

    def get_best(self) -> Optional[Organism]:
        with self._lock:
            if self._top_k_heap:
                return min(self._top_k_heap, key=lambda x: x[0])[2]
            evaluated = [o for o in self._organisms if o.fitness is not None]
            if not evaluated:
                return None
            return max(evaluated, key=lambda o: float(o.fitness))

    def size(self) -> int:
        with self._lock:
            return len(self._organisms)

    def all(self) -> List[Organism]:
        with self._lock:
            return self._organisms.copy()

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "population_size": len(self._organisms),
                "max_population": self.max_population,
                "min_fitness": self._cached_min_fitness,
                "max_fitness": self._cached_max_fitness,
                "min_generation": self._cached_min_gen,
                "max_generation": self._cached_max_gen,
                "top_k_size": len(self._top_k_heap),
            }

    def get_by_id(self, organism_id: str) -> Optional[Organism]:
        with self._lock:
            for o in self._organisms:
                if o.id == organism_id:
                    return o
            return None

    def get_lineage(self, organism: Organism, max_depth: int = 5) -> List[Organism]:
        with self._lock:
            lineage: List[Organism] = [organism]
            current = organism
            for _ in range(max_depth):
                if current.parent_id is None:
                    break
                parent = None
                for o in self._organisms:
                    if o.id == current.parent_id:
                        parent = o
                        break
                if parent is None:
                    break
                lineage.append(parent)
                current = parent
            return lineage

    def get_mutation_history(
        self,
        organism: Organism,
        max_depth: int = 5,
    ) -> List[Dict[str, Any]]:
        lineage = self.get_lineage(organism, max_depth)
        history: List[Dict[str, Any]] = []
        for org in lineage:
            history.append(
                {
                    "generation": org.generation,
                    "fitness": org.fitness,
                    "id": org.id,
                    "alloc_src": org.alloc_src,
                    "pay_src": org.pay_src,
                    "mutation_reasoning": getattr(org, "mutation_reasoning", ""),
                }
            )
        return history
