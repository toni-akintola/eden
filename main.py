"""
CLI tool for the number-guessing AI system (Alpha Evolve style)

This demonstrates how the explore-refine-act pipeline works:
1. EXPLORE: Analyze past attempts and propose candidate guesses
2. REFINE: Optimize the strategy and select the best guess
3. ACT: Make the final decision and commit to a guess

The system learns from feedback and iteratively improves its guesses.
"""

import os
import click
from ensemble import Ensemble
from database import Attempt
from openai import OpenAI


def get_feedback(guess: int, secret: int) -> tuple[str, int]:
    """
    Returns feedback for a guess
    Returns: (feedback_text, distance_score)
    """
    distance = abs(secret - guess)

    if guess == secret:
        return "🎉 CORRECT! You found the secret number!", 0
    elif guess < secret:
        if distance <= 5:
            return "Too low, but very close!", distance
        elif distance <= 15:
            return "Too low, getting warmer", distance
        else:
            return "Too low", distance
    else:  # guess > secret
        if distance <= 5:
            return "Too high, but very close!", distance
        elif distance <= 15:
            return "Too high, getting warmer", distance
        else:
            return "Too high", distance


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
    "--secret-number",
    "-s",
    type=int,
    help="The secret number to guess (1-100). If not provided, you'll be prompted.",
)
@click.option(
    "--max-attempts",
    "-m",
    type=int,
    help="Maximum number of attempts allowed. If not provided, you'll be prompted.",
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
def main(secret_number, max_attempts, explore_model, refine_model, act_model):
    """Number Guessing AI - Alpha Evolve Style CLI Tool

    An AI system that uses a three-stage pipeline (Explore-Refine-Act)
    to iteratively guess a secret number based on feedback.
    """
    click.echo()
    click.echo("=" * 70)
    click.echo("🎯 NUMBER GUESSING AI - Alpha Evolve Style")
    click.echo("=" * 70)
    click.echo()

    # Get available models
    click.echo("📡 Fetching available models from OpenAI...")
    available_models = get_available_models()
    click.echo(f"✓ Found {len(available_models)} available models")
    click.echo()

    # Interactive prompts if parameters not provided
    if secret_number is None:
        secret_number = click.prompt(
            "🔢 Enter the secret number (1-100)",
            type=click.IntRange(1, 100),
            default=42,
        )

    if max_attempts is None:
        max_attempts = click.prompt(
            "🎲 Enter maximum number of attempts",
            type=click.IntRange(1, 50),
            default=10,
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
        "Guess a secret number between 1 and 100. You'll get feedback on each guess."
    )

    # Display configuration
    click.echo()
    click.echo("=" * 70)
    click.echo("⚙️  CONFIGURATION")
    click.echo("=" * 70)
    click.echo(f"🎯 Secret number: {secret_number} (hidden from AI)")
    click.echo(f"🎲 Max attempts: {max_attempts}")
    click.echo(f"🔍 Explore model: {model_names[0]}")
    click.echo(f"🔧 Refine model: {model_names[1]}")
    click.echo(f"🎯 Act model: {model_names[2]}")
    click.echo("=" * 70)
    click.echo()

    # Confirm to start
    if not click.confirm("Ready to start?", default=True):
        click.echo("Cancelled.")
        return

    click.echo()

    # Initialize the ensemble
    ensemble = Ensemble(model_names=model_names, task=task_description)

    # Game loop
    for attempt_num in range(1, max_attempts + 1):
        click.echo(f"\n{'='*70}")
        click.echo(f"🎮 ATTEMPT #{attempt_num}/{max_attempts}")
        click.echo(f"{'='*70}")

        # Run the pipeline to get a guess
        click.echo(
            "🔍 Phase 1: EXPLORE - Analyzing patterns and generating candidates..."
        )
        click.echo("🔧 Phase 2: REFINE - Optimizing strategy...")
        click.echo("🎯 Phase 3: ACT - Making final decision...")
        click.echo()

        try:
            action = ensemble.pipeline()
            guess = action["final_guess"]

            click.echo("📊 AI Decision:")
            click.echo(f"  • Guess: {guess}")
            click.echo(f"  • Confidence: {action.get('confidence_level', 'unknown')}")
            click.echo(f"  • Reasoning: {action.get('reasoning', 'N/A')}")

            # Get feedback
            feedback_text, distance = get_feedback(guess, secret_number)
            click.echo()
            click.echo(f"💬 Feedback: {feedback_text}")
            click.echo(f"  • Distance from target: {distance}")

            # Store attempt in database for next iteration
            attempt = Attempt(attempt=guess, score=distance, observations=feedback_text)
            ensemble.database.add_attempt(attempt)

            # Check if won
            if guess == secret_number:
                click.echo()
                click.secho(
                    f"🎊 SUCCESS! The AI found the secret number in {attempt_num} attempts!",
                    fg="green",
                    bold=True,
                )
                break

        except Exception as e:
            click.echo()
            click.secho(f"❌ Error during pipeline execution: {e}", fg="red", err=True)
            click.echo("This might be due to API issues or JSON parsing errors.")
            break

    else:
        click.echo()
        click.secho(
            f"😔 Game Over! The AI didn't find the number in {max_attempts} attempts.",
            fg="yellow",
        )
        click.echo(f"The secret number was: {secret_number}")

    # Show summary
    click.echo()
    click.echo("=" * 70)
    click.echo("📈 GAME SUMMARY")
    click.echo("=" * 70)
    click.echo(f"Total attempts: {len(ensemble.database.get_attempts())}")
    click.echo()
    click.echo("All guesses:")
    for i, att in enumerate(ensemble.database.get_attempts(), 1):
        click.echo(f"  {i}. Guess: {att.attempt} → {att.observations}")
    click.echo()
    click.echo("=" * 70)


if __name__ == "__main__":
    main()
