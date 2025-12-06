#!/usr/bin/env python3
"""
Analyze evolutionary search results from Langfuse CSV export.
Generates visualizations and summary statistics.
"""

import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Set seaborn style for prettier plots
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.size"] = 10

# Increase CSV field size limit for large JSON fields
csv.field_size_limit(sys.maxsize)


@dataclass
class OrganismRecord:
    """Parsed organism from trace data."""

    id: str
    entry_rule_code: str
    exit_rule_code: str
    queue_discipline: str
    information_rule: str
    generation: int
    fitness: Optional[float]
    parent_id: Optional[str]
    mutation_reasoning: Optional[str]
    num_served: Optional[int] = None
    expected_queue_length: Optional[float] = None
    expected_service_flow: Optional[float] = None


def clean_json_string(s: str) -> str:
    """Clean up escaped JSON strings from CSV."""
    if not s:
        return s
    # Remove outer quotes if present
    s = s.strip()
    if s.startswith('"""') and s.endswith('"""'):
        s = s[3:-3]
    elif s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    # Unescape
    s = s.replace('\\"', '"')
    s = s.replace("\\n", "\n")
    return s


def parse_organism_from_json(data: dict) -> Optional[OrganismRecord]:
    """Parse an organism record from JSON data."""
    try:
        sim_results = data.get("cached_simulation_results", {})
        mutation = data.get("mutation_record", {})

        return OrganismRecord(
            id=data.get("id", ""),
            entry_rule_code=data.get("entry_rule_code", ""),
            exit_rule_code=data.get("exit_rule_code", ""),
            queue_discipline=data.get("queue_discipline", ""),
            information_rule=data.get("information_rule", ""),
            generation=data.get("generation", 0),
            fitness=data.get("fitness"),
            parent_id=data.get("parent_id"),
            mutation_reasoning=mutation.get("mutation_reasoning") if mutation else None,
            num_served=sim_results.get("num_served") if sim_results else None,
            expected_queue_length=(
                sim_results.get("expected_queue_length_E_k") if sim_results else None
            ),
            expected_service_flow=(
                sim_results.get("expected_service_flow_E_mu_k") if sim_results else None
            ),
        )
    except Exception as e:
        return None


def extract_organisms_from_csv(csv_path: str) -> List[OrganismRecord]:
    """Extract all organisms from the CSV file."""
    organisms = []
    seen_ids = set()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Try to parse output field (contains organism data)
            output_str = row.get("output", "")
            if output_str:
                output_str = clean_json_string(output_str)
                try:
                    data = json.loads(output_str)
                    if isinstance(data, dict) and "entry_rule_code" in data:
                        org = parse_organism_from_json(data)
                        if org and org.id not in seen_ids:
                            organisms.append(org)
                            seen_ids.add(org.id)
                except:
                    pass

            # Also try input field for additional organisms
            input_str = row.get("input", "")
            if input_str:
                input_str = clean_json_string(input_str)
                try:
                    data = json.loads(input_str)
                    # Check for nested organisms in kwargs
                    if isinstance(data, dict):
                        kwargs = data.get("kwargs", {})
                        for key in ["parent", "parent1", "parent2", "organism"]:
                            if key in kwargs:
                                org_data = kwargs[key]
                                if (
                                    isinstance(org_data, dict)
                                    and "entry_rule_code" in org_data
                                ):
                                    org = parse_organism_from_json(org_data)
                                    if org and org.id not in seen_ids:
                                        organisms.append(org)
                                        seen_ids.add(org.id)
                except:
                    pass

    return organisms


def analyze_top_performers(organisms: List[OrganismRecord], top_n: int = 10) -> None:
    """Analyze and print top performing organisms."""
    evaluated = [o for o in organisms if o.fitness is not None]
    if not evaluated:
        print("No evaluated organisms found.")
        return

    evaluated.sort(key=lambda x: x.fitness, reverse=True)
    top = evaluated[:top_n]

    print(f"\n{'='*80}")
    print(f"TOP {top_n} PERFORMING ORGANISMS")
    print(f"{'='*80}")

    for i, org in enumerate(top, 1):
        print(f"\n#{i} - Fitness: {org.fitness:.4f} (Gen {org.generation})")
        print(f"  ID: {org.id}")
        print(f"  Entry: {org.entry_rule_code}")
        print(f"  Exit: {org.exit_rule_code}")
        print(f"  Discipline: {org.queue_discipline}, Info: {org.information_rule}")
        if org.num_served:
            print(
                f"  Served: {org.num_served}, E[k]: {org.expected_queue_length:.3f}, E[μ]: {org.expected_service_flow:.3f}"
            )


def analyze_strategy_patterns(organisms: List[OrganismRecord]) -> Dict[str, Any]:
    """Analyze common patterns in successful strategies."""
    evaluated = [o for o in organisms if o.fitness is not None]
    if not evaluated:
        return {}

    # Group by discipline and info rule
    discipline_fitness = defaultdict(list)
    info_rule_fitness = defaultdict(list)

    for org in evaluated:
        discipline_fitness[org.queue_discipline].append(org.fitness)
        info_rule_fitness[org.information_rule].append(org.fitness)

    print(f"\n{'='*80}")
    print("STRATEGY PATTERN ANALYSIS")
    print(f"{'='*80}")

    print("\nQueue Discipline Performance:")
    for disc, fitnesses in sorted(discipline_fitness.items()):
        print(
            f"  {disc}: n={len(fitnesses)}, mean={np.mean(fitnesses):.4f}, max={max(fitnesses):.4f}"
        )

    print("\nInformation Rule Performance:")
    for rule, fitnesses in sorted(info_rule_fitness.items()):
        print(
            f"  {rule}: n={len(fitnesses)}, mean={np.mean(fitnesses):.4f}, max={max(fitnesses):.4f}"
        )

    # Analyze entry rule patterns in top performers
    top_10_pct = evaluated[: max(1, len(evaluated) // 10)]

    print("\nEntry Rule Patterns in Top 10%:")
    entry_patterns = defaultdict(int)
    for org in top_10_pct:
        # Extract pattern type
        code = org.entry_rule_code
        if "if k <" in code:
            # Count thresholds
            thresholds = re.findall(r"if k < (\d+)", code)
            pattern = f"threshold_based ({len(thresholds)} levels)"
        elif "lambda k:" in code and "/" in code:
            pattern = "inverse_based"
        elif "lambda k:" in code and re.search(r"\d+\.\d+\s*[+-]", code):
            pattern = "linear"
        else:
            pattern = "constant_or_other"
        entry_patterns[pattern] += 1

    for pattern, count in sorted(entry_patterns.items(), key=lambda x: -x[1]):
        print(f"  {pattern}: {count} organisms ({100*count/len(top_10_pct):.1f}%)")

    return {
        "discipline_fitness": dict(discipline_fitness),
        "info_rule_fitness": dict(info_rule_fitness),
    }


def print_summary_statistics(organisms: List[OrganismRecord]) -> None:
    """Print summary statistics."""
    evaluated = [o for o in organisms if o.fitness is not None]
    if not evaluated:
        print("No evaluated organisms found.")
        return

    fitnesses = [o.fitness for o in evaluated]
    generations = [o.generation for o in evaluated]

    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total organisms: {len(organisms)}")
    print(f"Evaluated organisms: {len(evaluated)}")
    print(f"Generation range: {min(generations)} - {max(generations)}")
    print(f"\nFitness Statistics:")
    print(f"  Min: {min(fitnesses):.4f}")
    print(f"  Max: {max(fitnesses):.4f}")
    print(f"  Mean: {np.mean(fitnesses):.4f}")
    print(f"  Median: {np.median(fitnesses):.4f}")
    print(f"  Std Dev: {np.std(fitnesses):.4f}")

    # Percentiles
    print(f"\nFitness Percentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  {p}th: {np.percentile(fitnesses, p):.4f}")


def plot_fitness_by_generation(
    organisms: List[OrganismRecord], output_file: str = None
) -> None:
    """Plot fitness progression by generation."""
    evaluated = [o for o in organisms if o.fitness is not None]
    if not evaluated:
        return

    # Group by generation
    gen_fitness = defaultdict(list)
    for org in evaluated:
        gen_fitness[org.generation].append(org.fitness)

    generations = sorted(gen_fitness.keys())
    avg_fitness = [np.mean(gen_fitness[g]) for g in generations]
    max_fitness = [max(gen_fitness[g]) for g in generations]
    min_fitness = [min(gen_fitness[g]) for g in generations]

    # Running best
    running_best = []
    best_so_far = float("-inf")
    for g in generations:
        best_so_far = max(best_so_far, max(gen_fitness[g]))
        running_best.append(best_so_far)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Fill area for range
    ax.fill_between(
        generations,
        min_fitness,
        max_fitness,
        alpha=0.15,
        color="gray",
        label="Range (min-max)",
    )

    # Plot mean fitness with seaborn styling
    sns.lineplot(
        x=generations,
        y=avg_fitness,
        ax=ax,
        linewidth=2,
        label="Mean Fitness",
        color=sns.color_palette()[0],
    )

    # Plot running best
    sns.lineplot(
        x=generations,
        y=running_best,
        ax=ax,
        linewidth=2.5,
        label="Best So Far",
        color=sns.color_palette()[2],
    )

    # Mark improvements
    improvements = [
        i for i in range(1, len(running_best)) if running_best[i] > running_best[i - 1]
    ]
    if improvements:
        ax.scatter(
            [generations[i] for i in improvements],
            [running_best[i] for i in improvements],
            color=sns.color_palette()[2],
            s=120,
            zorder=5,
            label="Improvements",
            marker="^",
            edgecolors="white",
            linewidths=1.5,
        )

    # Add theoretical optimum line at 14
    ax.axhline(
        y=14.0,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label="Theoretical Optimum (14.0)",
        zorder=3,
    )

    # Add annotation near the line
    if generations:
        ax.annotate(
            "Theoretical Optimum = 14.0",
            xy=(generations[-1], 14.0),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color="red",
            bbox=dict(
                boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="red"
            ),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        )

    ax.set_xlabel("Generation", fontsize=12, fontweight="bold")
    ax.set_ylabel("Fitness Score", fontsize=12, fontweight="bold")
    ax.set_title(
        "Fitness Progression Over Generations", fontsize=14, fontweight="bold", pad=15
    )
    ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Saved: {output_file}")
    else:
        plt.show()
    plt.close()


def plot_fitness_distribution(
    organisms: List[OrganismRecord], output_file: str = None
) -> None:
    """Plot fitness distribution histogram."""
    evaluated = [o for o in organisms if o.fitness is not None]
    if not evaluated:
        return

    fitnesses = [o.fitness for o in evaluated]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram with seaborn
    sns.histplot(
        fitnesses,
        bins=30,
        ax=ax1,
        kde=True,
        color=sns.color_palette()[0],
        edgecolor="white",
        linewidth=1.2,
        alpha=0.7,
    )
    ax1.axvline(
        np.mean(fitnesses),
        color="red",
        linestyle="--",
        linewidth=2.5,
        label=f"Mean: {np.mean(fitnesses):.2f}",
    )
    ax1.axvline(
        max(fitnesses),
        color="green",
        linestyle="--",
        linewidth=2.5,
        label=f"Best: {max(fitnesses):.2f}",
    )
    ax1.set_xlabel("Fitness Score", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax1.set_title("Fitness Distribution", fontsize=13, fontweight="bold", pad=10)
    ax1.legend(frameon=True, fancybox=True, shadow=True)

    # Generation distribution with seaborn
    generations = [o.generation for o in evaluated]
    gen_counts = defaultdict(int)
    for g in generations:
        gen_counts[g] += 1

    gens = sorted(gen_counts.keys())
    counts = [gen_counts[g] for g in gens]

    sns.barplot(
        x=gens,
        y=counts,
        ax=ax2,
        color=sns.color_palette()[1],
        edgecolor="white",
        linewidth=1.5,
        alpha=0.8,
    )
    ax2.set_xlabel("Generation", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Number of Organisms", fontsize=11, fontweight="bold")
    ax2.set_title("Population by Generation", fontsize=13, fontweight="bold", pad=10)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Saved: {output_file}")
    else:
        plt.show()
    plt.close()


def plot_strategy_comparison(
    organisms: List[OrganismRecord], output_file: str = None
) -> None:
    """Plot comparison of strategies."""
    evaluated = [o for o in organisms if o.fitness is not None]
    if not evaluated:
        return

    # Prepare data for seaborn
    plot_data = []
    for org in evaluated:
        key = f"{org.queue_discipline} + {org.information_rule}"
        plot_data.append({"Strategy": key, "Fitness": org.fitness})

    df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Use seaborn boxplot
    bp = sns.boxplot(
        data=df,
        x="Strategy",
        y="Fitness",
        ax=ax,
        palette="RdYlGn",
        width=0.6,
        linewidth=1.5,
        fliersize=4,
    )

    # Rotate x-axis labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right", fontsize=9)
    ax.set_xlabel("Strategy Combination", fontsize=12, fontweight="bold")
    ax.set_ylabel("Fitness Score", fontsize=12, fontweight="bold")
    ax.set_title(
        "Fitness by Strategy Combination\n(Queue Discipline + Information Rule)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # Add count annotations
    for i, strategy in enumerate(df["Strategy"].unique()):
        strategy_data = df[df["Strategy"] == strategy]["Fitness"]
        max_val = strategy_data.max()
        count = len(strategy_data)
        ax.annotate(
            f"n={count}",
            xy=(i, max_val),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="gray"
            ),
        )

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Saved: {output_file}")
    else:
        plt.show()
    plt.close()


def create_summary_report(
    organisms: List[OrganismRecord], output_file: str = None
) -> str:
    """Create a comprehensive text report."""
    evaluated = [o for o in organisms if o.fitness is not None]
    if not evaluated:
        return "No evaluated organisms found."

    evaluated.sort(key=lambda x: x.fitness, reverse=True)
    best = evaluated[0]

    fitnesses = [o.fitness for o in evaluated]
    generations = [o.generation for o in evaluated]

    report = f"""
================================================================================
                    EVOLUTIONARY SEARCH ANALYSIS REPORT
================================================================================

OVERVIEW
--------
Total organisms analyzed: {len(organisms)}
Evaluated organisms: {len(evaluated)}
Generation range: {min(generations)} - {max(generations)}

FITNESS STATISTICS
------------------
Best Fitness: {max(fitnesses):.4f}
Mean Fitness: {np.mean(fitnesses):.4f}
Median Fitness: {np.median(fitnesses):.4f}
Std Dev: {np.std(fitnesses):.4f}
Min Fitness: {min(fitnesses):.4f}

Percentiles:
  25th: {np.percentile(fitnesses, 25):.4f}
  50th: {np.percentile(fitnesses, 50):.4f}
  75th: {np.percentile(fitnesses, 75):.4f}
  90th: {np.percentile(fitnesses, 90):.4f}
  95th: {np.percentile(fitnesses, 95):.4f}
  99th: {np.percentile(fitnesses, 99):.4f}

BEST ORGANISM (Fitness: {best.fitness:.4f})
------------------------------------------
ID: {best.id}
Generation: {best.generation}
Queue Discipline: {best.queue_discipline}
Information Rule: {best.information_rule}

Entry Rule:
  {best.entry_rule_code}

Exit Rule:
  {best.exit_rule_code}

Simulation Results:
  Agents Served: {best.num_served if best.num_served else 'N/A'}
  Expected Queue Length E[k]: {f"{best.expected_queue_length:.4f}" if best.expected_queue_length else 'N/A'}
  Expected Service Flow E[μ_k]: {f"{best.expected_service_flow:.4f}" if best.expected_service_flow else 'N/A'}

STRATEGY ANALYSIS
-----------------
"""

    # Discipline stats
    discipline_fitness = defaultdict(list)
    info_rule_fitness = defaultdict(list)
    for org in evaluated:
        discipline_fitness[org.queue_discipline].append(org.fitness)
        info_rule_fitness[org.information_rule].append(org.fitness)

    report += "\nQueue Discipline Performance:\n"
    for disc, fits in sorted(discipline_fitness.items(), key=lambda x: -np.mean(x[1])):
        report += (
            f"  {disc}: n={len(fits)}, mean={np.mean(fits):.4f}, max={max(fits):.4f}\n"
        )

    report += "\nInformation Rule Performance:\n"
    for rule, fits in sorted(info_rule_fitness.items(), key=lambda x: -np.mean(x[1])):
        report += (
            f"  {rule}: n={len(fits)}, mean={np.mean(fits):.4f}, max={max(fits):.4f}\n"
        )

    report += f"""
TOP 5 ORGANISMS
---------------
"""
    for i, org in enumerate(evaluated[:5], 1):
        report += f"\n#{i} - Fitness: {org.fitness:.4f} (Gen {org.generation})\n"
        report += f"  Entry: {org.entry_rule_code}\n"
        report += f"  Exit: {org.exit_rule_code}\n"
        report += f"  {org.queue_discipline} / {org.information_rule}\n"

    print(report)

    if output_file:
        with open(output_file, "w") as f:
            f.write(report)
        print(f"\nSaved report to: {output_file}")

    return report


def main():
    # Default path
    csv_path = (
        sys.argv[1] if len(sys.argv) > 1 else "../notes/evolutionary-search-run.csv"
    )
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    print(f"Loading data from: {csv_path}")
    organisms = extract_organisms_from_csv(csv_path)
    print(f"Extracted {len(organisms)} unique organisms")

    if not organisms:
        print("No organisms found in CSV!")
        return

    # Print analyses
    print_summary_statistics(organisms)
    analyze_top_performers(organisms, top_n=10)
    analyze_strategy_patterns(organisms)

    # Generate plots
    print(f"\nGenerating visualizations...")
    plot_fitness_by_generation(organisms, f"{output_dir}/fitness_progression.png")
    plot_fitness_distribution(organisms, f"{output_dir}/fitness_distribution.png")
    plot_strategy_comparison(organisms, f"{output_dir}/strategy_comparison.png")

    # Generate report
    create_summary_report(organisms, f"{output_dir}/evolution_report.txt")

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
