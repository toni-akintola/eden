# EDEN: Economic Design Engine

**LLM-Driven Evolutionary Search for Optimal Economic Mechanism Design**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

EDEN (Economic Design Engine) is a system that uses large language models (LLMs) to guide evolutionary search for discovering optimal economic mechanisms. This repository implements EDEN's application to the queue design problem formalized by Che and Tercieux (2023), achieving near-optimal welfare scores through LLM-driven mutations.

## Overview

EDEN combines:
- **Evolutionary Algorithms**: Population-based search with selection pressure, crossover, and elitism
- **LLM-Guided Mutation**: GPT-5.1 generates intelligent mutations by reasoning about economic incentives
- **Discrete-Event Simulation**: Accurate evaluation of queue mechanisms via DES
- **Adaptive Control**: Temperature-based mutation strength that balances exploration and exploitation

**Key Results:**
- Achieves welfare scores within 0.07% of theoretical optimum (14.0103 vs 14.0)
- Independently rediscovers theoretical insights (information opacity, patient queue management)
- Converges in ~26 generations (424 evaluations) vs. millions required by RL approaches

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Results](#results)
- [Research Paper](#research-paper)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip
- OpenAI API key

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd eden
   ```

2. **Install dependencies:**
   
   Using `uv` (recommended):
   ```bash
   uv sync
   ```
   
   Or using `pip`:
```bash
pip install -r requirements.txt
```

3. **Set environment variables:**
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Quick Start

Run a short evolutionary search (10 steps, 4 workers):

```bash
cd evolve-experiment
uv run python main.py --steps 10 --workers 4 --visualize
```

This will:
- Initialize a population with a random seed organism
- Evolve for 10 generations using LLM-guided mutations
- Generate visualization plots showing fitness progression
- Output results to `evolution_results.json`

## Usage

### Basic Evolution Run

```bash
cd evolve-experiment
uv run python main.py \
    --steps 50 \
    --sim-time 100000 \
    --workers 8 \
    --model gpt-5.1 \
    --visualize
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--steps` | 10 | Number of evolution steps |
| `--sim-time` | 10000 | Simulation timesteps per evaluation |
| `--workers` | 1 | Parallel workers for evaluation |
| `--model` | gpt-5.1 | LLM model for mutations |
| `--selection-pressure` | 0.1 | Fraction of children to keep (top 10%) |
| `--elite-size` | 10 | Number of elite organisms to preserve |
| `--visualize` | False | Generate visualization plots |

### Full Parameter List

```bash
uv run python main.py --help
```

### Analyzing Results

If you have a CSV export from Langfuse:

```bash
cd notes
uv run python analyze_run.py --csv-path evolutionary-search-run.csv
```

This generates:
- `fitness_progression.png`: Fitness over generations
- `fitness_distribution.png`: Distribution of fitness scores
- `strategy_comparison.png`: Performance by strategy combination
- `evolution_report.txt`: Detailed text summary

## Project Structure

```
eden/
├── evolve-experiment/          # Main EDEN implementation
│   ├── main.py                 # CLI entry point and evolution loop
│   ├── database.py             # Population management with fitness-weighted sampling
│   ├── mutator.py              # LLM-based mutation with adaptive control
│   ├── queue_simulator.py      # Discrete-event simulation engine
│   ├── evaluator.py            # Welfare function computation
│   ├── evolve_types.py         # Type definitions for Che-Tercieux model
│   ├── utils.py                # Utility functions (parsing, prompts)
│   └── visualize.py            # Plotting and visualization
│
├── notes/                      # Analysis and results
│   ├── analyze_run.py          # Script to analyze CSV exports
│   ├── evolutionary-search-run.csv  # Example run data
│   ├── fitness_progression.png      # Generated visualizations
│   ├── fitness_distribution.png
│   ├── strategy_comparison.png
│   └── evolution_report.txt
│
├── main.tex                    # Research paper (LaTeX)
├── pyproject.toml              # Project dependencies
├── README.md                   # This file
├── ARCHITECTURE.md             # Detailed architecture documentation
└── CONTRIBUTING.md             # Contribution guidelines
```

## How It Works

### 1. Organism Representation

Each organism is a complete queue mechanism:
- **Queue Discipline**: FCFS, LIFO, or SIRO
- **Information Rule**: Full, Coarse, or No Information
- **Entry Rule**: Function `x(k) → [0,1]` mapping queue length to entry probability
- **Exit Rule**: Function `(k,ℓ) → (y, z)` mapping queue state to exit rate/probability

### 2. Fitness Evaluation

1. Run discrete-event simulation for 100,000 timesteps
2. Compute empirical stationary distribution
3. Calculate welfare: `W = (1-α)·R·E[μ_k] + α·(E[μ_k]·V - E[k]·C)`

### 3. LLM-Guided Mutation

GPT-5.1 receives:
- Parent organism's configuration and fitness
- High-performing "inspiration" organisms
- Current mutation strength (small/medium/large/radical)
- Economic context about queue design

The LLM generates a child organism by reasoning about which modifications might improve welfare.

### 4. Evolutionary Operators

- **Selection Pressure**: Only top 10% of children survive
- **Elitism**: Best organisms preserved across generations
- **Crossover**: Uniform crossover combining traits from two parents
- **Adaptive Mutation**: Temperature-based control increases exploration when stuck

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed technical documentation.

## Results

### Convergence Performance

- **Best Fitness**: 14.0103 (theoretical optimum: 14.0)
- **Generations to Convergence**: ~22
- **Total Evaluations**: 424
- **Sample Efficiency**: ~4 orders of magnitude better than RL approaches

### Key Discoveries

1. **Information Opacity**: All top organisms use NO_INFORMATION, confirming theoretical predictions
2. **Patient Queue Management**: Evolved exit rules implement "patience" by removing agents from congested queues
3. **Queue Discipline Near-Equivalence**: FCFS and SIRO achieve similar performance under no-information conditions

### Visualizations

See `notes/` directory for example visualizations:
- Fitness progression over generations
- Distribution of fitness scores
- Strategy performance comparisons

## Research Paper

This repository accompanies the research paper:

> **The Gardens of EDEN: Optimal Economic Mechanism Design via LLM-Driven Evolutionary Search**

The paper is available as `main.tex` (LaTeX source). Key contributions:
- Demonstrates LLM-driven evolutionary search for mechanism design
- Validates Che-Tercieux theoretical results computationally
- Shows sample efficiency advantages over RL approaches

## Development

### Running Tests

```bash
# Add tests when implemented
pytest tests/
```

### Code Quality

```bash
# Format code
ruff format evolve-experiment/

# Lint code
ruff check evolve-experiment/
```

### Adding New Features

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code style guidelines
- Pull request process
- Development setup
- Testing requirements

## References

- **Che, Y.-K., & Tercieux, O. (2023).** Optimal Queue Design. *arXiv preprint arXiv:2307.07746*.
- **Zheng, S., et al. (2022).** The AI Economist: Taxation policy design via two-level deep multiagent reinforcement learning. *Science Advances*, 8(18), eabk2607.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Che and Tercieux for the theoretical queue design framework
- OpenAI for GPT-5.1 API access
- Langfuse for observability and tracing

---

**Note**: This is research code. For production use, consider adding:
- Comprehensive test suite
- Security hardening (replace `eval()` usage)
- Performance optimizations (multiprocessing, GPU acceleration)
- Enhanced error handling

See the codebase quality assessment in the research paper for detailed recommendations.
