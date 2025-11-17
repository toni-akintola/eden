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
    db.add_attempt(
        Attempt(
            attempt={"num_servers": 1, "service_rate": 0.5, "queue_discipline": "FIFO"},
            score=45.2,
            observations="High wait times - system is struggling | Long queues (avg: 8.3) - consider more servers or faster service | Very high server utilization - servers are overloaded | Throughput: 78 agents served | ⚠️ System is unstable (arrival rate >= service capacity) - queues will grow indefinitely!",
        )
    )
    db.add_attempt(
        Attempt(
            attempt={"num_servers": 3, "service_rate": 1.0, "queue_discipline": "FIFO"},
            score=12.5,
            observations="Good wait times - acceptable performance | Queue stays short - good capacity | High server utilization - efficient use of resources | Throughput: 82 agents served",
        )
    )
    db.add_attempt(
        Attempt(
            attempt={"num_servers": 2, "service_rate": 0.8, "queue_discipline": "FIFO"},
            score=8.3,
            observations="Excellent wait times - agents are served quickly! | Queue stays short - good capacity | High server utilization - efficient use of resources | Throughput: 80 agents served",
        )
    )

    task = "Optimize hyperparameters for a multi-server queue system with arrival rate 0.8. Hyperparameters: num_servers (1-10), service_rate (0.1-5.0), queue_discipline (FIFO/LIFO/PRIORITY). Goal: Minimize efficiency score (wait time, queue length, balanced utilization)."
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
        "analysis": "Based on 3 configurations tested, 2 servers with service_rate 0.8 performed best. Single server was unstable. Three servers had good performance but may be over-provisioned.",
        "pattern_insights": "The optimal configuration balances capacity (num_servers * service_rate > arrival_rate) with utilization. Service rate around 0.8-1.0 per server seems optimal.",
        "candidate_configs": [
            {
                "num_servers": 2,
                "service_rate": 0.9,
                "queue_discipline": "FIFO",
                "reasoning": "Slightly higher service rate than best so far, should maintain stability while improving efficiency",
            },
            {
                "num_servers": 2,
                "service_rate": 0.75,
                "queue_discipline": "PRIORITY",
                "reasoning": "Try priority queue discipline to see if it improves wait times for high-priority agents",
            },
        ],
        "confidence": "high",
        "key_learnings": "Two servers appears optimal. Service rate should be around 0.8-1.0. Need to test if queue discipline matters.",
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
        "evaluation": "The exploration correctly identified that 2 servers with service_rate around 0.8-0.9 is optimal. Priority queue discipline is worth testing.",
        "optimal_strategy": "Balance capacity and utilization - ensure stability (arrival_rate < num_servers * service_rate) while minimizing wait times",
        "recommended_config": {
            "num_servers": 2,
            "service_rate": 0.85,
            "queue_discipline": "FIFO",
            "reasoning": "2 servers with 0.85 service rate provides good balance: rho = 0.8/(2*0.85) = 0.47, which is stable and efficient",
        },
        "reasoning": "This configuration should maintain low wait times while keeping utilization around 80%",
        "expected_outcome": "Efficiency score around 7-9, with wait times under 2.0 and queue length under 1.0",
        "alternative_config": {
            "num_servers": 2,
            "service_rate": 0.9,
            "queue_discipline": "PRIORITY",
            "reasoning": "Alternative with priority queue to test if discipline affects performance",
        },
    }

    prompt = build_act_prompt(refinement)
    print(prompt)
    print("\n" + "=" * 80)
    print()


def main():
    print("\n🔍 Testing all three prompt stages for queuing system optimization...\n")

    test_explore_prompt()
    test_refine_prompt()
    test_act_prompt()

    print("✅ All prompts generated successfully!")
    print("\nThese are the prompts that will be sent to the AI models.")
    print("You can review them to understand how the system works.")


if __name__ == "__main__":
    main()
