"""
Demo script for the number-guessing AI system (Alpha Evolve style)

This demonstrates how the explore-refine-act pipeline works:
1. EXPLORE: Analyze past attempts and propose candidate guesses
2. REFINE: Optimize the strategy and select the best guess
3. ACT: Make the final decision and commit to a guess

The system learns from feedback and iteratively improves its guesses.
"""

import random
from ensemble import Ensemble
from database import Attempt

# Configuration
SECRET_NUMBER = random.randint(1, 100)  # The number to guess
MAX_ATTEMPTS = 10
TASK_DESCRIPTION = (
    "Guess a secret number between 1 and 100. You'll get feedback on each guess."
)

# You can use different models for each stage, or the same model
# Example: ["gpt-4", "gpt-4", "gpt-4"] or ["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo"]
MODEL_NAMES = ["gpt-4o-mini", "gpt-4o-mini", "gpt-4o-mini"]


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


def main():
    print("=" * 60)
    print("NUMBER GUESSING AI - Alpha Evolve Style")
    print("=" * 60)
    print(f"Secret number: {SECRET_NUMBER} (hidden from AI)")
    print(f"Task: {TASK_DESCRIPTION}")
    print(
        f"Models: Explore={MODEL_NAMES[0]}, Refine={MODEL_NAMES[1]}, Act={MODEL_NAMES[2]}"
    )
    print("=" * 60)
    print()

    # Initialize the ensemble
    ensemble = Ensemble(model_names=MODEL_NAMES, task=TASK_DESCRIPTION)

    # Game loop
    for attempt_num in range(1, MAX_ATTEMPTS + 1):
        print(f"\n{'='*60}")
        print(f"ATTEMPT #{attempt_num}")
        print(f"{'='*60}")

        # Run the pipeline to get a guess
        print("🔍 Phase 1: EXPLORE - Analyzing patterns and generating candidates...")
        print("🔧 Phase 2: REFINE - Optimizing strategy...")
        print("🎯 Phase 3: ACT - Making final decision...")

        try:
            action = ensemble.pipeline()
            guess = action["final_guess"]

            print(f"\n📊 AI Decision:")
            print(f"  - Guess: {guess}")
            print(f"  - Confidence: {action.get('confidence_level', 'unknown')}")
            print(f"  - Reasoning: {action.get('reasoning', 'N/A')}")

            # Get feedback
            feedback_text, distance = get_feedback(guess, SECRET_NUMBER)
            print(f"\n💬 Feedback: {feedback_text}")
            print(f"  - Distance from target: {distance}")

            # Store attempt in database for next iteration
            attempt = Attempt(attempt=guess, score=distance, observations=feedback_text)
            ensemble.database.add_attempt(attempt)

            # Check if won
            if guess == SECRET_NUMBER:
                print(
                    f"\n🎊 SUCCESS! The AI found the secret number in {attempt_num} attempts!"
                )
                break

        except Exception as e:
            print(f"\n❌ Error during pipeline execution: {e}")
            print("This might be due to API issues or JSON parsing errors.")
            break

    else:
        print(
            f"\n😔 Game Over! The AI didn't find the number in {MAX_ATTEMPTS} attempts."
        )
        print(f"The secret number was: {SECRET_NUMBER}")

    # Show summary
    print(f"\n{'='*60}")
    print("GAME SUMMARY")
    print(f"{'='*60}")
    print(f"Total attempts: {len(ensemble.database.get_attempts())}")
    print("\nAll guesses:")
    for i, att in enumerate(ensemble.database.get_attempts(), 1):
        print(f"  {i}. Guess: {att.attempt} → {att.observations}")


if __name__ == "__main__":
    main()
