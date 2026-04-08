#!/usr/bin/env python3
"""
CLI for running auction mechanism evolution (evaluate → database → mutate / crossover).

Run from the econ-bench directory (or ensure it is on PYTHONPATH):

  cd econ-bench && python evolve.py run --steps 5 --no-llm

With LLM mutations (requires OPENAI_API_KEY):

  python evolve.py run --steps 10 --model gpt-4o
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from auction import AuctionInstance, evaluate
from database import Database
from mutator import Mutator, Organism, SEED_ORGANISMS, crossover

logger = logging.getLogger(__name__)


def _eval_organism(
    organism: Organism,
    profiles: np.ndarray,
    instance: AuctionInstance,
    n_misreports: int,
    penalty: float,
    rng: np.random.Generator,
) -> bool:
    """
    Evaluate organism in-place. Returns False if compilation or evaluation fails.
    """
    try:
        mechanism = organism.to_callables()
        result = evaluate(
            mechanism,
            profiles,
            n_misreports=n_misreports,
            penalty=penalty,
            rng=rng,
            instance=instance,
        )
    except Exception as e:
        logger.warning(
            "evaluation failed for organism %s: %s",
            organism.id,
            e,
        )
        logger.debug("evaluation traceback", exc_info=True)
        organism.fitness = float("-inf")
        organism.revenue = 0.0
        organism.regret = float("inf")
        organism.welfare = 0.0
        return False
    organism.revenue = result.revenue
    organism.regret = result.regret
    organism.welfare = result.welfare
    organism.fitness = result.fitness
    return True


def _organism_from_seed(template: Organism) -> Organism:
    return Organism(
        alloc_src=template.alloc_src,
        pay_src=template.pay_src,
        generation=0,
        mutation_reasoning=template.mutation_reasoning,
    )


def _pick_mate(db: Database, parent: Organism, rng: random.Random) -> Organism | None:
    others = [o for o in db.all() if o.id != parent.id]
    if not others:
        return None
    return rng.choice(others)


def run_evolution(args: argparse.Namespace) -> int:
    rng_np = np.random.default_rng(args.seed)
    rng_py = random.Random(args.seed)

    logger.info(
        "evolution run: steps=%d profiles=%d misreports=%d penalty=%g correlation=%g "
        "max_population=%d prune_keep=%g fitness_w=%g recency_w=%g seed=%d no_llm=%s",
        args.steps,
        args.profiles,
        args.misreports,
        args.penalty,
        args.correlation,
        args.max_population,
        args.prune_keep,
        args.fitness_weight,
        args.recency_weight,
        args.seed,
        args.no_llm,
    )
    if args.budgets is not None:
        logger.info("budgets=%s", args.budgets)
    if not args.no_llm:
        logger.info("LLM model=%s (crossover prob adaptive)", args.model)
    else:
        logger.info("crossover probability (fixed)=%g", args.crossover_prob)

    instance = AuctionInstance(
        correlation=args.correlation,
        budgets=args.budgets,
    )
    profiles = instance.sample(n=args.profiles, rng=rng_np)
    logger.info("sampled %d type profiles for evaluation", len(profiles))

    db = Database(max_population=args.max_population, prune_keep_ratio=args.prune_keep)

    for template in SEED_ORGANISMS:
        org = _organism_from_seed(template)
        eval_rng = np.random.default_rng(rng_np.integers(0, 2**31 - 1))
        ok = _eval_organism(
            org,
            profiles,
            instance,
            args.misreports,
            args.penalty,
            eval_rng,
        )
        if not ok:
            logger.warning("seed organism failed evaluation; skipping id=%s", org.id)
            continue
        db.add(org)
        logger.info(
            "seed added id=%s fitness=%.4f (population=%d)",
            org.id,
            org.fitness if org.fitness is not None else float("nan"),
            db.size(),
        )

    if db.size() == 0:
        logger.error("no valid seed organisms in database")
        return 1

    mutator: Mutator | None = None
    if not args.no_llm:
        mutator = Mutator(model=args.model)

    best = db.get_best()
    assert best is not None
    print(
        f"initialized population={db.size()}  best_fitness={best.fitness:.4f}  "
        f"best_id={best.id}",
        flush=True,
    )

    for step in range(1, args.steps + 1):
        parent, inspirations = db.sample(
            fitness_weight=args.fitness_weight,
            recency_weight=args.recency_weight,
        )
        pf = parent.fitness
        pf_s = f"{pf:.4f}" if pf is not None else "None"
        logger.info(
            "step %d/%d: sampled parent=%s gen=%s fitness=%s; inspirations=%s",
            step,
            args.steps,
            parent.id,
            parent.generation,
            pf_s,
            [i.id for i in inspirations],
        )

        if mutator is not None:
            best_cur = db.get_best()
            assert best_cur is not None
            mutator.update_adaptive_state(float(best_cur.fitness))
            info = mutator.get_mutation_info()
            p_cross = info["crossover_probability"]
            logger.debug(
                "adaptive state: temp=%.4g strength=%s p_cross=%.4g steps_no_improve=%d",
                info["temperature"],
                info["mutation_strength"],
                p_cross,
                info["steps_without_improvement"],
            )
        else:
            p_cross = args.crossover_prob

        mate = _pick_mate(db, parent, rng_py)
        roll = rng_py.random()
        crossed_early = mate is not None and roll < p_cross
        if crossed_early:
            assert mate is not None
            base = crossover(parent, mate)
            logger.info(
                "crossover before mutate: mate=%s roll=%.4f p_cross=%.4f -> base=%s gen=%s",
                mate.id,
                roll,
                p_cross,
                base.id,
                base.generation,
            )
        else:
            base = parent
            logger.info(
                "no pre-mutate crossover: mate=%s roll=%.4f p_cross=%.4f; base=parent %s",
                mate.id if mate else None,
                roll,
                p_cross,
                parent.id,
            )

        if mutator is not None:
            child = mutator.mutate(base, inspirations)
            logger.info(
                "child from LLM mutate: id=%s gen=%s", child.id, child.generation
            )
        elif mate is not None and base is parent:
            child = crossover(parent, mate)
            logger.info(
                "child from crossover (no LLM): %s x %s -> %s",
                parent.id,
                mate.id,
                child.id,
            )
        elif mate is not None:
            child = base
            logger.info("child is crossover base (no LLM): id=%s", child.id)
        else:
            child = crossover(parent, parent)
            logger.warning(
                "degenerate crossover(parent, parent) — only parent=%s",
                parent.id,
            )

        eval_rng = np.random.default_rng(rng_np.integers(0, 2**31 - 1))
        ok = _eval_organism(
            child,
            profiles,
            instance,
            args.misreports,
            args.penalty,
            eval_rng,
        )
        if not ok:
            logger.warning(
                "step %d: child %s failed evaluation; not added",
                step,
                child.id,
            )
            continue

        db.add(child)
        best = db.get_best()
        stats = db.get_stats()
        assert best is not None
        print(
            f"step {step}/{args.steps}  pop={stats['population_size']}  "
            f"child_fit={child.fitness:.4f}  best_fit={best.fitness:.4f}  best_id={best.id}",
            flush=True,
        )

    print("\n=== best mechanism ===", flush=True)
    best = db.get_best()
    if best is not None:
        print(best.summary(), flush=True)
    return 0


def _parse_budgets(s: str | None) -> list[float] | None:
    if s is None or s.strip() == "":
        return None
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "budgets must be two comma-separated floats (bidder1,bidder2)"
        )
    return parts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evolve 2-bidder 2-item auction mechanisms."
    )
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run evolution for a number of timesteps")
    run_p.add_argument(
        "--steps", type=int, default=10, help="Number of evolution steps"
    )
    run_p.add_argument("--seed", type=int, default=42, help="RNG seed")
    run_p.add_argument(
        "--profiles",
        type=int,
        default=5_000,
        help="Number of type profiles per evaluation batch",
    )
    run_p.add_argument(
        "--misreports",
        type=int,
        default=500,
        help="Misreport samples per bidder per profile (regret estimate)",
    )
    run_p.add_argument(
        "--penalty", type=float, default=1.0, help="Regret penalty λ in fitness"
    )
    run_p.add_argument(
        "--correlation", type=float, default=0.0, help="Copula ρ for item values"
    )
    run_p.add_argument(
        "--budgets",
        type=_parse_budgets,
        default=None,
        help="Optional per-bidder caps, e.g. 1.0,1.0 (comma-separated)",
    )
    run_p.add_argument(
        "--max-population", type=int, default=2_000, help="Max organisms before prune"
    )
    run_p.add_argument(
        "--prune-keep",
        type=float,
        default=0.5,
        help="Fraction of population to keep when pruning",
    )
    run_p.add_argument(
        "--fitness-weight",
        type=float,
        default=0.7,
        help="Weight on fitness vs recency when sampling parents",
    )
    run_p.add_argument(
        "--recency-weight",
        type=float,
        default=0.3,
        help="Weight on generation recency when sampling parents",
    )
    run_p.add_argument(
        "--crossover-prob",
        type=float,
        default=0.3,
        help="Crossover probability when --no-llm (adaptive when LLM is on)",
    )
    run_p.add_argument(
        "--no-llm",
        action="store_true",
        help="Do not call the LLM; evolve via crossover only",
    )
    run_p.add_argument(
        "--model", type=str, default="gpt-5.1", help="OpenAI model id for Mutator"
    )
    run_p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity for logging to stderr (default INFO)",
    )
    run_p.set_defaults(func=run_evolution)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
