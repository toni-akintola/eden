#!/usr/bin/env python3
"""Evolutionary optimizer for Che-Tercieux queue models."""

import os
import sys
import json
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
        queue_discipline=QueueDiscipline.FCFS,
        information_rule=InformationRule.NO_INFORMATION_BEYOND_REC,
    )


def create_seed_organism() -> Organism:
    """Create a default seed organism to bootstrap evolution."""
    return Organism(
        entry_rule_code="lambda k: 1.0",
        exit_rule_code="lambda k, l: (0.0, 0.0)",
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
) -> float:
    """Evaluate an organism and return its fitness score."""
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
    return evaluate_designer_performance(model, results)


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
    verbose: bool,
    output_file: str | None,
) -> dict:
    """Run the evolutionary optimization loop."""
    database = Database()
    mutator = Mutator(model=model_name)

    # Seed the database
    seed = create_seed_organism()
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
        click.echo(f"Fixed parameters: lambda={arrival_rate}, mu={service_rate}")
        click.echo(f"Seed organism fitness: {seed.fitness:.4f}")
        click.echo(f"Starting evolution for {num_steps} steps...\n")

    history = []
    for step in range(1, num_steps + 1):
        if verbose:
            click.echo(f"Step {step}/{num_steps}")

        try:
            # Sample parent and inspirations
            parent, inspirations = database.sample()

            # Mutate to create child
            child = mutator.mutate(parent, inspirations, arrival_rate, service_rate)

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

            # Add to database
            database.add(child)

            best = database.get_best()
            history.append(
                {
                    "step": step,
                    "child_fitness": child.fitness,
                    "best_fitness": best.fitness,
                }
            )

            if verbose:
                click.echo(
                    f"  Child fitness: {child.fitness:.4f}, Best: {best.fitness:.4f}"
                )

        except Exception as e:
            click.echo(f"  Error: {e}", err=True)
            continue

    # Get best organism
    best = database.get_best()
    results = {
        "best_fitness": best.fitness if best else None,
        "best_organism": (
            {
                "entry_rule_code": best.entry_rule_code,
                "exit_rule_code": best.exit_rule_code,
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
        click.echo(f"\nEvolution complete. Best fitness: {best.fitness:.4f}")
        click.echo(f"Best organism (gen {best.generation}):")
        click.echo(f"  entry: {best.entry_rule_code}")
        click.echo(f"  exit: {best.exit_rule_code}")

    if output_file:
        json_results = {
            "best_fitness": results["best_fitness"],
            "best_organism": results["best_organism"],
            "population_size": results["population_size"],
            "history": results["history"],
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
    default="gpt-4o-mini",
    show_default=True,
    help="LLM model for mutations",
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
@click.option("-o", "--output", default=None, help="Output JSON file")
@click.option("-q", "--quiet", is_flag=True, help="Quiet mode")
@click.option("--visualize", is_flag=True, help="Generate visualization plots")
def main(
    steps,
    sim_time,
    model,
    arrival_rate,
    service_rate,
    v_surplus,
    c_cost,
    r_profit,
    alpha,
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
