"""
Test script to preview the prompts without making API calls.
This is useful for debugging and understanding what the AI sees.
"""

from constants import build_explore_prompt, build_refine_prompt, build_act_prompt
from database import Database, Attempt


def test_explore_prompt():
    """Test the explore prompt with sample data"""
    print("=" * 80)
    print("EXPLORE PROMPT TEST")
    print("=" * 80)

    # Create a database with some sample attempts
    db = Database()
    db.add_attempt(Attempt(attempt=50, score=30, observations="Too low"))
    db.add_attempt(
        Attempt(attempt=75, score=5, observations="Too high, but very close!")
    )
    db.add_attempt(
        Attempt(attempt=68, score=2, observations="Too low, but very close!")
    )

    task = "Guess a secret number between 1 and 100"
    prompt = build_explore_prompt(task, db)

    print(prompt)
    print("\n" + "=" * 80)
    print()


def test_refine_prompt():
    """Test the refine prompt with sample exploration"""
    print("=" * 80)
    print("REFINE PROMPT TEST")
    print("=" * 80)

    exploration = {
        "analysis": "Based on 3 attempts, the number is between 68 and 75",
        "current_range": {"min": 68, "max": 75},
        "pattern_insights": "The number is very close to our recent guesses",
        "candidate_guesses": [
            {"value": 70, "reasoning": "Middle of narrowed range"},
            {"value": 69, "reasoning": "Slightly lower, given feedback pattern"},
            {"value": 71, "reasoning": "Slightly higher, alternative hypothesis"},
        ],
        "confidence": "high",
    }

    prompt = build_refine_prompt(exploration)
    print(prompt)
    print("\n" + "=" * 80)
    print()


def test_act_prompt():
    """Test the act prompt with sample refinement"""
    print("=" * 80)
    print("ACT PROMPT TEST")
    print("=" * 80)

    refinement = {
        "evaluation": "The exploration correctly identified a narrow range",
        "optimal_strategy": "Binary search with slight bias toward 70",
        "recommended_guess": 70,
        "reasoning": "70 is the optimal binary search midpoint between 68-75, and feedback suggests we're very close",
        "expected_outcome": "Either correct or within 1-2 of the target",
        "backup_guess": 69,
    }

    prompt = build_act_prompt(refinement)
    print(prompt)
    print("\n" + "=" * 80)
    print()


def main():
    print("\n🔍 Testing all three prompt stages...\n")

    test_explore_prompt()
    test_refine_prompt()
    test_act_prompt()

    print("✅ All prompts generated successfully!")
    print("\nThese are the prompts that will be sent to the AI models.")
    print("You can review them to understand how the system works.")


if __name__ == "__main__":
    main()
