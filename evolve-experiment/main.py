"""
CLI tool for queuing system hyperparameter optimization (Alpha Evolve style)

This demonstrates how the explore-refine-act pipeline works:
1. EXPLORE: Analyze past hyperparameter configurations and propose candidates
2. REFINE: Optimize the strategy and select the best configuration
3. ACT: Make the final decision and commit to hyperparameters

The system learns from simulation feedback and iteratively improves hyperparameters.
"""

import os
import click
from ensemble import Ensemble
from database import Attempt
from evaluator import Evaluator
from openai import OpenAI
import json


def get_available_models():
    """Fetch available OpenAI models"""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        models = client.models.list()
        # Filter to only chat models (gpt models)
        model_ids = [
            model.id for model in models if model.id.startswith(("gpt-", "o1-", "o3-"))
        ]
        # Sort and put common ones first
        priority_models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
        ]
        sorted_models = []
        for pm in priority_models:
            if pm in model_ids:
                sorted_models.append(pm)
        # Add remaining models
        for m in sorted(model_ids):
            if m not in sorted_models:
                sorted_models.append(m)
        return sorted_models
    except Exception as e:
        click.echo(f"Error fetching models: {e}", err=True)
        # Fallback to common models
        return ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]


@click.command()
@click.option(
    "--arrival-rate",
    "-a",
    type=float,
    help="Arrival rate (lambda) for the queue system. If not provided, you'll be prompted.",
)
@click.option(
    "--max-iterations",
    "-m",
    type=int,
    help="Maximum number of optimization iterations. If not provided, you'll be prompted.",
)
@click.option(
    "--simulation-duration",
    "-d",
    type=float,
    help="Simulation duration for each evaluation. If not provided, you'll be prompted.",
)
@click.option(
    "--explore-model",
    type=str,
    help="Model to use for the EXPLORE stage. If not provided, you'll be prompted.",
)
@click.option(
    "--refine-model",
    type=str,
    help="Model to use for the REFINE stage. If not provided, you'll be prompted.",
)
@click.option(
    "--act-model",
    type=str,
    help="Model to use for the ACT stage. If not provided, you'll be prompted.",
)
def main(
    arrival_rate,
    max_iterations,
    simulation_duration,
    explore_model,
    refine_model,
    act_model,
):
    """Queue System Hyperparameter Optimizer - Alpha Evolve Style CLI Tool

    An AI system that uses a three-stage pipeline (Explore-Refine-Act)
    to iteratively optimize queuing system hyperparameters based on simulation feedback.
    """
    click.echo()

    click.echo("=" * 70)
    click.echo("🚦 QUEUE SYSTEM HYPERPARAMETER OPTIMIZER - Alpha Evolve Style")
    click.echo("=" * 70)
    click.echo()

    # Get available models
    click.echo("📡 Fetching available models from OpenAI...")
    available_models = get_available_models()
    click.echo(f"✓ Found {len(available_models)} available models")
    click.echo()

    # Interactive prompts if parameters not provided
    if arrival_rate is None:
        arrival_rate = click.prompt(
            "📥 Enter arrival rate (lambda) - agents per time unit",
            type=float,
            default=0.8,
        )

    if max_iterations is None:
        max_iterations = click.prompt(
            "🔄 Enter maximum number of optimization iterations",
            type=click.IntRange(1, 50),
            default=10,
        )

    if simulation_duration is None:
        simulation_duration = click.prompt(
            "⏱️  Enter simulation duration per evaluation",
            type=float,
            default=100.0,
        )

    click.echo()
    click.echo("🤖 Now select models for each stage of the pipeline:")
    click.echo("   (You can use the same model for all stages or mix different models)")
    click.echo()

    if explore_model is None:
        explore_model = click.prompt(
            "   1️⃣  EXPLORE stage model (analyzes patterns)",
            type=click.Choice(available_models, case_sensitive=True),
            default=available_models[0] if available_models else "gpt-4o-mini",
            show_choices=True,
        )

    if refine_model is None:
        refine_model = click.prompt(
            "   2️⃣  REFINE stage model (optimizes strategy)",
            type=click.Choice(available_models, case_sensitive=True),
            default=explore_model,  # Default to same as explore
            show_choices=True,
        )

    if act_model is None:
        act_model = click.prompt(
            "   3️⃣  ACT stage model (makes final decision)",
            type=click.Choice(available_models, case_sensitive=True),
            default=refine_model,  # Default to same as refine
            show_choices=True,
        )

    model_names = [explore_model, refine_model, act_model]
    task_description = (
        f"Optimize hyperparameters for a multi-server queue system with arrival rate {arrival_rate}. "
        f"Hyperparameters: num_servers (1-10), service_rate (0.1-5.0), queue_discipline (FIFO/LIFO/PRIORITY). "
        f"Goal: Minimize efficiency score (wait time, queue length, balanced utilization)."
    )

    # Initialize evaluator
    evaluator = Evaluator(
        target_arrival_rate=arrival_rate,
        simulation_duration=simulation_duration,
        seed=42,
    )

    # Display configuration
    click.echo()
    click.echo("=" * 70)
    click.echo("⚙️  CONFIGURATION")
    click.echo("=" * 70)
    click.echo(f"📥 Arrival rate (lambda): {arrival_rate}")
    click.echo(f"🔄 Max iterations: {max_iterations}")
    click.echo(f"⏱️  Simulation duration: {simulation_duration}")
    click.echo(f"🔍 Explore model: {model_names[0]}")
    click.echo(f"🔧 Refine model: {model_names[1]}")
    click.echo(f"🎯 Act model: {model_names[2]}")
    click.echo("=" * 70)
    click.echo()

    # Confirm to start
    if not click.confirm("Ready to start optimization?", default=True):
        click.echo("Cancelled.")
        return

    click.echo()

    # Initialize the ensemble
    ensemble = Ensemble(model_names=model_names, task=task_description)

    best_score = float("inf")
    best_config = None

    # Optimization loop
    for iteration_num in range(1, max_iterations + 1):
        click.echo(f"\n{'='*70}")
        click.echo(f"🔄 ITERATION #{iteration_num}/{max_iterations}")
        click.echo(f"{'='*70}")

        # Run the pipeline to get hyperparameters
        click.echo(
            "🔍 Phase 1: EXPLORE - Analyzing past configurations and generating candidates..."
        )
        click.echo("🔧 Phase 2: REFINE - Optimizing strategy...")
        click.echo("🎯 Phase 3: ACT - Making final decision...")
        click.echo()

        try:
            action = ensemble.pipeline()
            hyperparameters = action["hyperparameters"]

            click.echo("📊 AI Decision:")
            click.echo(f"  • Servers: {hyperparameters['num_servers']}")
            click.echo(f"  • Service rate: {hyperparameters['service_rate']:.2f}")
            click.echo(f"  • Queue discipline: {hyperparameters['queue_discipline']}")
            click.echo(f"  • Confidence: {action.get('confidence_level', 'unknown')}")
            click.echo(f"  • Reasoning: {action.get('reasoning', 'N/A')[:100]}...")

            # Evaluate the configuration
            click.echo()
            click.echo("🧪 Running simulation...")
            evaluation = evaluator.evaluate(hyperparameters)
            efficiency_score = evaluation["efficiency_score"]
            metrics = evaluation["metrics"]
            observations = evaluation["observations"]

            click.echo()
            click.echo("📈 Simulation Results:")
            click.echo(
                f"  • Efficiency score: {efficiency_score:.2f} (lower is better)"
            )
            click.echo(f"  • Average wait time: {metrics['average_wait_time']:.2f}")
            click.echo(
                f"  • Average queue length: {metrics['average_queue_length']:.2f}"
            )
            click.echo(f"  • Server utilization: {metrics['server_utilization']:.2%}")
            click.echo(f"  • Agents served: {metrics['agents_served']}")
            click.echo(f"  • 💬 Feedback: {observations}")

            # Track best configuration
            if efficiency_score < best_score:
                best_score = efficiency_score
                best_config = hyperparameters
                click.secho(
                    f"  ✨ New best configuration! (score: {best_score:.2f})",
                    fg="green",
                    bold=True,
                )

            # Store attempt in database for next iteration
            attempt = Attempt(
                attempt=hyperparameters,
                score=efficiency_score,
                observations=observations,
            )
            ensemble.database.add_attempt(attempt)

        except Exception as e:
            click.echo()
            click.secho(f"❌ Error during pipeline execution: {e}", fg="red", err=True)
            click.echo("This might be due to API issues or JSON parsing errors.")
            import traceback

            click.echo(traceback.format_exc())
            break

    # Show summary
    click.echo()
    click.echo("=" * 70)
    click.echo("📈 OPTIMIZATION SUMMARY")
    click.echo("=" * 70)
    click.echo(f"Total iterations: {len(ensemble.database.get_attempts())}")
    click.echo()

    if best_config:
        click.echo("🏆 Best Configuration Found:")
        click.echo(f"  • Servers: {best_config['num_servers']}")
        click.echo(f"  • Service rate: {best_config['service_rate']:.2f}")
        click.echo(f"  • Queue discipline: {best_config['queue_discipline']}")
        click.echo(f"  • Efficiency score: {best_score:.2f}")
        click.echo()

    click.echo("All configurations tested:")
    for i, att in enumerate(ensemble.database.get_attempts(), 1):
        config = att.attempt
        click.echo(
            f"  {i}. Servers={config['num_servers']}, "
            f"μ={config['service_rate']:.2f}, "
            f"Discipline={config['queue_discipline']} → "
            f"Score: {att.score:.2f}"
        )
    click.echo()
    click.echo("=" * 70)


if __name__ == "__main__":
    main()
