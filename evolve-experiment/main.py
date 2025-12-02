#!/usr/bin/env python3
"""
Queue Model Evolution Optimizer

An evolutionary optimization system for the Che-Tercieux queue model using
LLM-based agents to explore, refine, and act on configuration space.
"""

import os
import sys
import json
import click
from typing import Optional
from ensemble import Ensemble
from database import Database, Attempt
from queue_simulator import QueueSimulator
from evaluator import evaluate_designer_performance
from evolve_types import SimulationResults
from utils import convert_config_to_model


def run_evolution(
    num_iterations: int,
    simulation_time: float,
    model_names: Optional[list[str]],
    task: Optional[str],
    verbose: bool,
    output_file: Optional[str],
    V_surplus: float = 10.0,
    C_waiting_cost: float = 1.0,
    R_provider_profit: float = 5.0,
    alpha_weight: float = 0.5,
) -> dict:
    """
    Run the evolution loop to optimize queue model parameters.

    Args:
        num_iterations: Number of explore-refine-act cycles to run
        simulation_time: How long to run each simulation
        model_names: List of 3 model names for [explore, refine, act]
        task: Task description for the ensemble
        verbose: Print detailed progress
        output_file: Optional JSON file to save results

    Returns:
        Dictionary with best_score, best_config, and all_attempts
    """
    # Default models
    if model_names is None:
        model_names = ["gpt-4o-mini", "gpt-4o-mini", "gpt-4o-mini"]

    # Default task
    if task is None:
        task = """Optimize the Che-Tercieux queue model to maximize social welfare (W).
        Balance provider profit and agent utility by tuning arrival rates, service rates,
        entry rules, and exit rules. Consider queue discipline and information structure."""

    # Initialize
    if verbose:
        click.echo(f"Starting evolution with {num_iterations} iterations")
        click.echo(f"Using models: {model_names}")
        click.echo(f"Simulation time per iteration: {simulation_time:.0f}")
        click.echo()

    ensemble = Ensemble(model_names, task)
    database = ensemble.database

    best_score = float("-inf")
    best_config = None

    for iteration in range(1, num_iterations + 1):
        if verbose:
            click.echo("=" * 60)
            click.echo(f"ITERATION {iteration}/{num_iterations}")
            click.echo("=" * 60)

        try:
            # 1. Run explore-refine-act pipeline
            if verbose:
                click.echo("Running agent pipeline (Explore -> Refine -> Act)...")

            result = ensemble.pipeline()
            config = result.get("final_config", {})

            if verbose:
                click.echo(f"  Confidence: {result.get('confidence_level', 'unknown')}")
                reasoning = result.get("reasoning", "N/A")
                click.echo(f"  Reasoning: {reasoning[:100]}...")

            # 2. Convert config to model
            if verbose:
                click.echo("Converting config to CheTercieuxQueueModel...")

            model = convert_config_to_model(
                config, V_surplus, C_waiting_cost, R_provider_profit, alpha_weight
            )

            # 3. Run simulation
            if verbose:
                click.echo(f"Running simulation ({simulation_time:.0f} time units)...")

            simulator = QueueSimulator(model)
            sim_results = simulator.run_simulation(max_time=simulation_time)

            # 4. Evaluate performance
            if verbose:
                click.echo("Evaluating welfare score...")

            welfare_score = evaluate_designer_performance(model, sim_results)

            # 5. Generate observations
            observations = _generate_observations(config, sim_results, welfare_score)

            # 6. Store in database
            database.add_attempt(
                Attempt(attempt=config, score=welfare_score, observations=observations)
            )

            # 7. Track best
            if welfare_score > best_score:
                best_score = welfare_score
                best_config = config
                if verbose:
                    click.secho(
                        f"\nNEW BEST SCORE: {welfare_score:.4f}", fg="green", bold=True
                    )
            else:
                if verbose:
                    click.echo(f"\nScore: {welfare_score:.4f} (Best: {best_score:.4f})")

            # 8. Print summary
            if verbose:
                click.echo("\nIteration Summary:")
                click.echo(f"  Welfare Score (W): {welfare_score:.4f}")
                click.echo(
                    f"  Expected Queue Length E[k]: {sim_results.expected_queue_length_E_k:.2f}"
                )
                click.echo(
                    f"  Expected Service Flow E[mu_k]: {sim_results.expected_service_flow_E_mu_k:.2f}"
                )
                click.echo(f"  Agents Served: {sim_results.num_served}")
                click.echo(
                    f"  V_surplus: {model.V_surplus:.2f}, C_cost: {model.C_waiting_cost:.2f}"
                )
                click.echo(
                    f"  R_profit: {model.R_provider_profit:.2f}, alpha: {model.alpha_weight:.2f}"
                )
                click.echo()

        except Exception as e:
            click.echo(f"Error in iteration {iteration}: {e}", err=True)
            if verbose:
                import traceback

                traceback.print_exc()
            continue

    # Prepare results
    results = {
        "best_score": best_score,
        "best_config": best_config,
        "total_attempts": len(database.get_attempts()),
        "all_scores": [attempt.score for attempt in database.get_attempts()],
    }

    # Final summary
    if verbose:
        click.echo("=" * 60)
        click.echo("EVOLUTION COMPLETE")
        click.echo("=" * 60)
        click.echo(f"\nBest Welfare Score: {best_score:.4f}")
        click.echo(f"Total Attempts: {results['total_attempts']}")

        if best_config:
            click.echo("\nBest Configuration:")
            for key, value in best_config.items():
                if key != "reasoning":
                    click.echo(f"  {key}: {value}")

    # Save to file if requested
    if output_file:
        output_data = {
            "best_score": best_score,
            "best_config": best_config,
            "all_attempts": [
                {
                    "config": attempt.attempt,
                    "score": attempt.score,
                    "observations": attempt.observations,
                }
                for attempt in database.get_attempts()
            ],
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        if verbose:
            click.echo(f"\nResults saved to: {output_file}")

    return results


def _generate_observations(
    config: dict, results: SimulationResults, welfare_score: float
) -> str:
    """Generate human-readable observations about the simulation results."""

    E_k = results.expected_queue_length_E_k
    E_mu = results.expected_service_flow_E_mu_k

    observations = []
    observations.append(f"Welfare score: {welfare_score:.4f}")
    observations.append(f"Expected queue length E[k]: {E_k:.2f}")
    observations.append(f"Expected service flow E[mu_k]: {E_mu:.2f}")
    observations.append(f"Agents served: {results.num_served}")

    # Add insights
    if E_k > 10:
        observations.append(
            "High queue length - consider increasing service rate or adding exit rules"
        )
    elif E_k < 1:
        observations.append("Low queue length - system is underutilized")

    if welfare_score > 10:
        observations.append("Strong welfare performance")
    elif welfare_score < 0:
        observations.append("Negative welfare - costs exceed benefits")

    return " | ".join(observations)


@click.command()
@click.option(
    "-n",
    "--iterations",
    type=int,
    default=10,
    show_default=True,
    help="Number of optimization iterations",
)
@click.option(
    "-t",
    "--sim-time",
    type=float,
    default=10000.0,
    show_default=True,
    help="Simulation time per iteration",
)
@click.option(
    "-m",
    "--models",
    type=str,
    multiple=True,
    help="Model names (provide 3: explore, refine, act). Example: -m gpt-4o -m gpt-4o-mini -m gpt-4o-mini",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output JSON file for results",
)
@click.option(
    "-q", "--quiet", is_flag=True, help="Minimal output (only show final results)"
)
@click.option(
    "--task",
    type=str,
    default=None,
    help="Custom task description for the optimization",
)
@click.option(
    "--v-surplus",
    type=float,
    default=10.0,
    show_default=True,
    help="Net surplus from service (V > 0)",
)
@click.option(
    "--c-cost",
    type=float,
    default=1.0,
    show_default=True,
    help="Per-period waiting cost (C > 0)",
)
@click.option(
    "--r-profit",
    type=float,
    default=5.0,
    show_default=True,
    help="Profit per served agent (R > 0)",
)
@click.option(
    "--alpha",
    type=float,
    default=0.5,
    show_default=True,
    help="Weight on agents' welfare (alpha in [0, 1])",
)
def main(
    iterations,
    sim_time,
    models,
    output,
    quiet,
    task,
    v_surplus,
    c_cost,
    r_profit,
    alpha,
):
    """
    Evolutionary optimizer for Che-Tercieux queue models.

    Uses LLM-based agents to explore, refine, and optimize queue system
    parameters for maximum social welfare.

    Examples:

      \b
      # Run 10 iterations with defaults
      python main.py

      \b
      # Run 20 iterations with longer simulations
      python main.py --iterations 20 --sim-time 50000

      \b
      # Save results to file
      python main.py --output results.json

      \b
      # Use specific models
      python main.py -m gpt-4o -m gpt-4o-mini -m gpt-4o-mini

      \b
      # Quiet mode
      python main.py --quiet --iterations 5
    """
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        click.echo("Error: OPENAI_API_KEY environment variable not set", err=True)
        sys.exit(1)

    # Validate models
    model_names = None
    if models:
        if len(models) != 3:
            click.echo(
                "Error: Must provide exactly 3 model names (explore, refine, act)",
                err=True,
            )
            sys.exit(1)
        model_names = list(models)

    # Run evolution
    try:
        results = run_evolution(
            num_iterations=iterations,
            simulation_time=sim_time,
            model_names=model_names,
            task=task,
            verbose=not quiet,
            output_file=output,
            V_surplus=v_surplus,
            C_waiting_cost=c_cost,
            R_provider_profit=r_profit,
            alpha_weight=alpha,
        )

        # Always print final score
        click.echo(f"\nFinal Best Score: {results['best_score']:.4f}")

        sys.exit(0)

    except KeyboardInterrupt:
        click.echo("\n\nOptimization interrupted by user", err=True)
        sys.exit(130)
    except Exception as e:
        click.echo(f"\nFatal error: {e}", err=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
