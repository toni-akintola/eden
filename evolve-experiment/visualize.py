"""Visualization tools for evolutionary optimization."""

import matplotlib.pyplot as plt
import numpy as np
from typing import List
from database import Organism


def plot_fitness_progression(history: List[dict], output_file: str = None):
    """
    Plot fitness progression over evolution steps.

    Args:
        history: List of dicts with 'step', 'avg_child_fitness', 'best_child_fitness', 'best_fitness'
        output_file: Optional file to save plot
    """
    if not history:
        return

    steps = [h["step"] for h in history]
    avg_child_fitness = [h.get("avg_child_fitness", 0.0) for h in history]
    best_child_fitness = [h.get("best_child_fitness") for h in history]
    best_fitness = [h.get("best_fitness") for h in history]

    plt.figure(figsize=(10, 6))
    if any(f is not None for f in avg_child_fitness):
        plt.plot(
            steps,
            avg_child_fitness,
            "o-",
            alpha=0.5,
            label="Avg Child Fitness",
            markersize=4,
        )
    if any(f is not None for f in best_child_fitness):
        plt.plot(
            steps,
            best_child_fitness,
            "^-",
            alpha=0.6,
            label="Best Child Fitness",
            markersize=4,
        )
    if any(f is not None for f in best_fitness):
        plt.plot(
            steps,
            best_fitness,
            "s-",
            linewidth=2,
            label="Overall Best Fitness",
            markersize=6,
        )

    # Highlight improvements
    if any(f is not None for f in best_fitness):
        improvements = [
            i
            for i in range(1, len(steps))
            if best_fitness[i] is not None
            and best_fitness[i - 1] is not None
            and best_fitness[i] > best_fitness[i - 1]
        ]
        if improvements:
            plt.scatter(
                [steps[i] for i in improvements],
                [best_fitness[i] for i in improvements],
                color="green",
                s=100,
                zorder=5,
                label="Improvements",
            )

    plt.xlabel("Evolution Step")
    plt.ylabel("Fitness Score")
    plt.title("Fitness Progression Over Evolution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def plot_population_stats(database, output_file: str = None):
    """
    Plot population statistics: fitness distribution and generation distribution.

    Args:
        database: Database instance with organisms
        output_file: Optional file to save plot
    """
    organisms = database.all()
    if not organisms:
        return

    evaluated = [o for o in organisms if o.fitness is not None]
    if not evaluated:
        return

    fitnesses = [o.fitness for o in evaluated]
    generations = [o.generation for o in evaluated]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Fitness distribution
    ax1.hist(fitnesses, bins=min(20, len(fitnesses)), edgecolor="black", alpha=0.7)
    ax1.axvline(
        np.mean(fitnesses),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(fitnesses):.2f}",
    )
    ax1.axvline(
        max(fitnesses),
        color="green",
        linestyle="--",
        label=f"Best: {max(fitnesses):.2f}",
    )
    ax1.set_xlabel("Fitness Score")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Fitness Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Generation distribution
    gen_counts = {}
    for g in generations:
        gen_counts[g] = gen_counts.get(g, 0) + 1

    gens = sorted(gen_counts.keys())
    counts = [gen_counts[g] for g in gens]
    ax2.bar(gens, counts, edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Number of Organisms")
    ax2.set_title("Population by Generation")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def visualize_functions(
    organism: Organism,
    arrival_rate: float = None,
    service_rate: float = None,
    k_range: range = range(0, 20),
    output_file: str = None,
):
    """
    Visualize the actual function behavior by plotting them over queue lengths.

    Args:
        organism: Organism to visualize
        arrival_rate: Fixed exogenous arrival rate (lambda)
        service_rate: Fixed exogenous service rate (mu)
        k_range: Range of queue lengths to plot
        output_file: Optional file to save plot
    """
    from utils import parse_code_to_function

    try:
        entry_fn = parse_code_to_function(organism.entry_rule_code)
        exit_fn = parse_code_to_function(organism.exit_rule_code)
    except Exception as e:
        print(f"Error parsing functions: {e}")
        return

    k_vals = list(k_range)
    entry_vals = [entry_fn(k) for k in k_vals]

    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Arrival rate (fixed exogenous, shown as constant)
    if arrival_rate is not None:
        axes[0, 0].axhline(
            y=arrival_rate, color="b", linewidth=2, label=f"λ = {arrival_rate}"
        )
        axes[0, 0].set_xlabel("Queue Length (k)")
        axes[0, 0].set_ylabel("Arrival Rate")
        axes[0, 0].set_title(f"Arrival Rate (Fixed Exogenous)\nλ = {arrival_rate}")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, "Arrival rate not provided", ha="center", va="center")
        axes[0, 0].set_title("Arrival Rate")

    # Service rate (fixed exogenous, shown as constant)
    if service_rate is not None:
        axes[0, 1].axhline(
            y=service_rate, color="g", linewidth=2, label=f"μ = {service_rate}"
        )
        axes[0, 1].set_xlabel("Queue Length (k)")
        axes[0, 1].set_ylabel("Service Rate")
        axes[0, 1].set_title(f"Service Rate (Fixed Exogenous)\nμ = {service_rate}")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, "Service rate not provided", ha="center", va="center")
        axes[0, 1].set_title("Service Rate")

    # Entry rule
    axes[1, 0].plot(k_vals, entry_vals, "r-o", markersize=4)
    axes[1, 0].set_xlabel("Queue Length (k)")
    axes[1, 0].set_ylabel("Entry Probability")
    axes[1, 0].set_title(f"Entry Rule\n{organism.entry_rule_code[:60]}...")
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].grid(True, alpha=0.3)

    # Exit rule (sample at k=10, varying l)
    l_vals = list(range(1, 11))
    exit_rates = [exit_fn(10, l)[0] for l in l_vals]
    exit_probs = [exit_fn(10, l)[1] for l in l_vals]

    ax_twin = axes[1, 1].twinx()
    axes[1, 1].plot(l_vals, exit_rates, "m-o", markersize=4, label="Rate")
    ax_twin.plot(l_vals, exit_probs, "c-s", markersize=4, label="Probability")
    axes[1, 1].set_xlabel("Position (l) at k=10")
    axes[1, 1].set_ylabel("Exit Rate", color="m")
    ax_twin.set_ylabel("Exit Probability", color="c")
    axes[1, 1].set_title(f"Exit Rule (at k=10)\n{organism.exit_rule_code[:60]}...")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(
        f"Organism Functions (Fitness: {organism.fitness:.4f if organism.fitness else 'N/A'}, Gen: {organism.generation}, Discipline: {organism.queue_discipline})",
        fontsize=14,
        y=0.995,
    )
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def create_summary_report(database, history: List[dict], output_file: str = None):
    """Create a comprehensive summary report."""
    best = database.get_best()
    if not best:
        print("No organisms evaluated yet")
        return

    report = f"""
EVOLUTION SUMMARY REPORT
{'='*60}
Total Steps: {len(history)}
Population Size: {database.size()}
Best Fitness: {best.fitness:.4f if best.fitness else 'N/A'}
Best Generation: {best.generation}

BEST ORGANISM:
  Entry Rule: {best.entry_rule_code}
  Exit Rule: {best.exit_rule_code}
  Queue Discipline: {best.queue_discipline}

FITNESS STATISTICS:
  Mean Avg Child: {np.mean([h.get('avg_child_fitness', 0.0) for h in history]):.4f}
  Mean Best Child: {np.mean([h.get('best_child_fitness', 0.0) or 0.0 for h in history]):.4f}
  Final Best: {max([h.get('best_fitness', 0.0) or 0.0 for h in history]):.4f}
  
IMPROVEMENTS: {len([i for i in range(1, len(history)) if history[i].get('best_fitness') and history[i-1].get('best_fitness') and history[i]['best_fitness'] > history[i-1]['best_fitness']])} steps
"""

    print(report)

    if output_file:
        with open(output_file, "w") as f:
            f.write(report)
        print(f"Report saved to {output_file}")
