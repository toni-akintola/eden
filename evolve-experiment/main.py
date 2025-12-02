#!/usr/bin/env python3
"""Evolutionary optimizer for Che-Tercieux queue models."""

import os
import sys
import json
import random
from concurrent.futures import ThreadPoolExecutor
from typing import List
import click
from database import Database, Organism
from ensemble import Mutator
from queue_simulator import QueueSimulator
from evaluator import evaluate_designer_performance
from evolve_types import (
    CheTercieuxQueueModel,
    PrimitiveProcess,
    EntryExitRule,
    QueueDiscipline,
    InformationRule,
)
from utils import parse_code_to_function


def organism_to_model(
    organism: Organism,
    arrival_rate: float,
    service_rate: float,
    V_surplus: float,
    C_waiting_cost: float,
    R_provider_profit: float,
    alpha_weight: float,
) -> CheTercieuxQueueModel:
    """Convert an Organism's code to a CheTercieuxQueueModel."""
    entry_fn = parse_code_to_function(organism.entry_rule_code)
    exit_fn = parse_code_to_function(organism.exit_rule_code)

    # Parse queue discipline string to enum
    discipline_map = {
        "FCFS": QueueDiscipline.FCFS,
        "LIFO": QueueDiscipline.LIFO,
        "SIRO": QueueDiscipline.SIRO,
    }
    queue_discipline = discipline_map.get(
        organism.queue_discipline.upper(), QueueDiscipline.FCFS
    )

    return CheTercieuxQueueModel(
        V_surplus=V_surplus,
        C_waiting_cost=C_waiting_cost,
        R_provider_profit=R_provider_profit,
        alpha_weight=alpha_weight,
        primitive_process=PrimitiveProcess(
            arrival_rate_fn=lambda k: arrival_rate,
            service_rate_fn=lambda k: service_rate,
        ),
        design_rules=EntryExitRule(entry_rule_fn=entry_fn, exit_rule_fn=exit_fn),
        queue_discipline=queue_discipline,
        information_rule=InformationRule.NO_INFORMATION_BEYOND_REC,
    )


def generate_random_entry_rule() -> str:
    """Generate a random entry rule function."""
    rule_type = random.choice(["constant", "linear_decrease", "threshold", "inverse"])

    if rule_type == "constant":
        prob = random.uniform(0.1, 1.0)
        return f"lambda k: {prob:.3f}"
    elif rule_type == "linear_decrease":
        start = random.uniform(0.5, 1.0)
        slope = random.uniform(-0.1, -0.01)
        return f"lambda k: max(0.0, min(1.0, {start:.3f} + {slope:.3f} * k))"
    elif rule_type == "threshold":
        threshold = random.randint(1, 10)
        prob_above = random.uniform(0.0, 0.5)
        prob_below = random.uniform(0.5, 1.0)
        return f"lambda k: {prob_below:.3f} if k < {threshold} else {prob_above:.3f}"
    else:  # inverse
        base = random.uniform(0.3, 0.9)
        factor = random.uniform(0.1, 0.5)
        return f"lambda k: max(0.0, min(1.0, {base:.3f} / (1.0 + {factor:.3f} * k)))"


def generate_random_exit_rule() -> str:
    """Generate a random exit rule function."""
    rule_type = random.choice(
        ["no_exit", "position_based", "queue_length_based", "constant_rate"]
    )

    if rule_type == "no_exit":
        return "lambda k, l: (0.0, 0.0)"
    elif rule_type == "position_based":
        rate = random.uniform(0.1, 2.0)
        prob = random.uniform(0.1, 0.5)
        return f"lambda k, l: ({rate:.3f}, {prob:.3f}) if l > {random.randint(2, 5)} else (0.0, 0.0)"
    elif rule_type == "queue_length_based":
        threshold = random.randint(3, 8)
        rate = random.uniform(0.1, 1.5)
        prob = random.uniform(0.1, 0.4)
        return (
            f"lambda k, l: ({rate:.3f}, {prob:.3f}) if k > {threshold} else (0.0, 0.0)"
        )
    else:  # constant_rate
        rate = random.uniform(0.05, 1.0)
        prob = random.uniform(0.05, 0.3)
        return f"lambda k, l: ({rate:.3f}, {prob:.3f})"


def create_seed_organism(random_seed: int = 42) -> Organism:
    """Create a randomly generated seed organism to bootstrap evolution."""
    # Set seed for reproducibility
    random.seed(random_seed)

    return Organism(
        entry_rule_code=generate_random_entry_rule(),
        exit_rule_code=generate_random_exit_rule(),
        queue_discipline=random.choice(["FCFS", "LIFO", "SIRO"]),
        generation=0,
    )


def evaluate_organism(
    organism: Organism,
    simulation_time: float,
    arrival_rate: float,
    service_rate: float,
    V_surplus: float,
    C_waiting_cost: float,
    R_provider_profit: float,
    alpha_weight: float,
    return_results: bool = False,
):
    """
    Evaluate an organism and return its fitness score.

    Args:
        return_results: If True, returns (fitness, results) tuple instead of just fitness

    Returns:
        fitness score, or (fitness, results) if return_results=True
    """
    model = organism_to_model(
        organism,
        arrival_rate,
        service_rate,
        V_surplus,
        C_waiting_cost,
        R_provider_profit,
        alpha_weight,
    )
    simulator = QueueSimulator(model)
    results = simulator.run_simulation(max_time=simulation_time)
    fitness = evaluate_designer_performance(model, results)

    if return_results:
        return fitness, results
    return fitness


def mutate_and_evaluate(
    parent: Organism,
    inspirations: List[Organism],
    mutator: Mutator,
    simulation_time: float,
    arrival_rate: float,
    service_rate: float,
    V_surplus: float,
    C_waiting_cost: float,
    R_provider_profit: float,
    alpha_weight: float,
) -> Organism:
    """Mutate and evaluate a single organism."""
    # Get parent's simulation results for behavior insights
    parent_results = None
    if parent.fitness is not None:
        # Re-run parent simulation to get detailed results
        parent_fitness, parent_results = evaluate_organism(
            parent,
            simulation_time,
            arrival_rate,
            service_rate,
            V_surplus,
            C_waiting_cost,
            R_provider_profit,
            alpha_weight,
            return_results=True,
        )

    # Mutate to create child (with parent behavior data)
    child = mutator.mutate(
        parent,
        inspirations,
        arrival_rate,
        service_rate,
        parent_simulation_results=parent_results,
    )

    # Evaluate child
    child.fitness = evaluate_organism(
        child,
        simulation_time,
        arrival_rate,
        service_rate,
        V_surplus,
        C_waiting_cost,
        R_provider_profit,
        alpha_weight,
    )

    return child


def run_evolution(
    num_steps: int,
    simulation_time: float,
    model_name: str,
    arrival_rate: float,
    service_rate: float,
    V_surplus: float,
    C_waiting_cost: float,
    R_provider_profit: float,
    alpha_weight: float,
    random_seed: int,
    num_workers: int,
    verbose: bool,
    output_file: str | None,
) -> dict:
    """Run the evolutionary optimization loop with parallel workers."""
    # Set random seed for reproducibility
    random.seed(random_seed)

    database = Database()
    mutator = Mutator(model=model_name)

    # Seed the database with randomly generated organism
    seed = create_seed_organism(random_seed)
    seed.fitness = evaluate_organism(
        seed,
        simulation_time,
        arrival_rate,
        service_rate,
        V_surplus,
        C_waiting_cost,
        R_provider_profit,
        alpha_weight,
    )
    database.add(seed)

    if verbose:
        click.echo(f"Random seed: {random_seed}")
        click.echo(f"Workers: {num_workers}")
        click.echo(f"Fixed parameters: lambda={arrival_rate}, mu={service_rate}")
        click.echo(f"Seed organism:")
        click.echo(f"  entry: {seed.entry_rule_code}")
        click.echo(f"  exit: {seed.exit_rule_code}")
        click.echo(f"  queue_discipline: {seed.queue_discipline}")
        click.echo(f"Seed organism fitness: {seed.fitness:.4f}")
        click.echo(f"Starting evolution for {num_steps} steps...\n")

    history = []

    # Synchronized batch evolution: each step runs all workers in parallel
    for step in range(1, num_steps + 1):
        if verbose:
            click.echo(f"Step {step}/{num_steps}")

        try:
            # Sample parents for all workers (synchronized - all from same DB state)
            parent_inspiration_pairs = []
            for _ in range(num_workers):
                parent, inspirations = database.sample()
                parent_inspiration_pairs.append((parent, inspirations))

            # Run all mutations and evaluations in parallel
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        mutate_and_evaluate,
                        parent=parent,
                        inspirations=inspirations,
                        mutator=mutator,
                        simulation_time=simulation_time,
                        arrival_rate=arrival_rate,
                        service_rate=service_rate,
                        V_surplus=V_surplus,
                        C_waiting_cost=C_waiting_cost,
                        R_provider_profit=R_provider_profit,
                        alpha_weight=alpha_weight,
                    )
                    for parent, inspirations in parent_inspiration_pairs
                ]
                children = []
                for f in futures:
                    try:
                        child = f.result()
                        children.append(child)
                    except Exception as e:
                        if verbose:
                            click.echo(f"  Error in worker: {e}", err=True)

            # Aggregate: add all children to database
            for child in children:
                database.add(child)

            # Record step results
            best = database.get_best()
            if children:
                # Calculate statistics for this step
                child_fitnesses = [c.fitness for c in children if c.fitness is not None]
                avg_fitness = (
                    sum(child_fitnesses) / len(child_fitnesses)
                    if child_fitnesses
                    else 0.0
                )
                best_child_fitness = max(child_fitnesses) if child_fitnesses else None

                history.append(
                    {
                        "step": step,
                        "num_children": len(children),
                        "avg_child_fitness": avg_fitness,
                        "best_child_fitness": best_child_fitness,
                        "best_fitness": best.fitness if best else None,
                    }
                )

                if verbose:
                    best_child_str = (
                        f"{best_child_fitness:.4f}"
                        if best_child_fitness is not None
                        else "N/A"
                    )
                    best_overall_str = (
                        f"{best.fitness:.4f}"
                        if best and best.fitness is not None
                        else "N/A"
                    )
                    click.echo(
                        f"  {len(children)} children evaluated, "
                        f"avg={avg_fitness:.4f}, "
                        f"best_child={best_child_str}, "
                        f"overall_best={best_overall_str}"
                    )

        except Exception as e:
            click.echo(f"  Error in step {step}: {e}", err=True)
            continue

    # Get best organism
    best = database.get_best()
    results = {
        "best_fitness": best.fitness if best else None,
        "best_organism": (
            {
                "entry_rule_code": best.entry_rule_code,
                "exit_rule_code": best.exit_rule_code,
                "queue_discipline": best.queue_discipline,
                "generation": best.generation,
            }
            if best
            else None
        ),
        "population_size": database.size(),
        "history": history,
        "database": database,
    }

    if verbose:
        if best:
            click.echo(f"\nEvolution complete. Best fitness: {best.fitness:.4f}")
            click.echo(f"Best organism (gen {best.generation}):")
            click.echo(f"  entry: {best.entry_rule_code}")
            click.echo(f"  exit: {best.exit_rule_code}")
            click.echo(f"  queue_discipline: {best.queue_discipline}")
        else:
            click.echo("\nEvolution complete. No organisms evaluated.")

    if output_file:
        json_results = {
            "best_fitness": results["best_fitness"],
            "best_organism": results["best_organism"],
            "population_size": results["population_size"],
            "history": history,
        }
        with open(output_file, "w") as f:
            json.dump(json_results, f, indent=2)
        if verbose:
            click.echo(f"\nResults saved to {output_file}")

    return results


@click.command()
@click.option(
    "-n", "--steps", default=10, show_default=True, help="Number of evolution steps"
)
@click.option(
    "-t",
    "--sim-time",
    default=10000.0,
    show_default=True,
    help="Simulation time per evaluation",
)
@click.option(
    "-m",
    "--model",
    default="gpt-5.1",
    show_default=True,
    help="LLM model for mutations",
)
@click.option(
    "-w",
    "--workers",
    default=1,
    show_default=True,
    help="Number of parallel workers",
)
@click.option(
    "--lambda",
    "arrival_rate",
    default=2.0,
    show_default=True,
    help="Arrival rate (exogenous)",
)
@click.option(
    "--mu",
    "service_rate",
    default=3.0,
    show_default=True,
    help="Service rate (exogenous)",
)
@click.option(
    "--v-surplus", default=10.0, show_default=True, help="V surplus parameter"
)
@click.option(
    "--c-cost", default=1.0, show_default=True, help="C waiting cost parameter"
)
@click.option(
    "--r-profit", default=5.0, show_default=True, help="R provider profit parameter"
)
@click.option("--alpha", default=0.5, show_default=True, help="Alpha weight parameter")
@click.option(
    "--seed",
    default=42,
    show_default=True,
    help="Random seed for reproducibility",
    type=int,
)
@click.option("-o", "--output", default=None, help="Output JSON file")
@click.option("-q", "--quiet", is_flag=True, help="Quiet mode")
@click.option("--visualize", is_flag=True, help="Generate visualization plots")
def main(
    steps,
    sim_time,
    model,
    workers,
    arrival_rate,
    service_rate,
    v_surplus,
    c_cost,
    r_profit,
    alpha,
    seed,
    output,
    quiet,
    visualize,
):
    """Evolutionary optimizer for Che-Tercieux queue models."""
    if not os.getenv("OPENAI_API_KEY"):
        click.echo("Error: OPENAI_API_KEY not set", err=True)
        sys.exit(1)

    results = run_evolution(
        num_steps=steps,
        simulation_time=sim_time,
        model_name=model,
        arrival_rate=arrival_rate,
        service_rate=service_rate,
        V_surplus=v_surplus,
        C_waiting_cost=c_cost,
        R_provider_profit=r_profit,
        alpha_weight=alpha,
        random_seed=seed,
        num_workers=workers,
        verbose=not quiet,
        output_file=output,
    )

    if visualize:
        from visualize import (
            plot_fitness_progression,
            plot_population_stats,
            create_summary_report,
        )

        history = results.get("history", [])
        database = results.get("database")

        if history and database:
            plot_fitness_progression(history, "fitness_progression.png")
            plot_population_stats(database, "population_stats.png")
            create_summary_report(database, history, "evolution_report.txt")


if __name__ == "__main__":
    main()
