# EDEN Evolution Experiment

This directory contains the core implementation of EDEN's evolutionary search system.

## Quick Start

```bash
# Run a short evolution (10 steps, 4 workers)
uv run python main.py --steps 10 --workers 4 --visualize

# Run a full evolution (50 steps, 8 workers)
uv run python main.py --steps 50 --workers 8 --model gpt-5.1 --visualize
```

## Files

- **`main.py`**: CLI entry point and evolution loop orchestrator
- **`database.py`**: Population management with fitness-weighted sampling
- **`mutator.py`**: LLM-based mutation with adaptive control
- **`queue_simulator.py`**: Discrete-event simulation engine
- **`evaluator.py`**: Welfare function computation
- **`evolve_types.py`**: Type definitions for Che-Tercieux model
- **`utils.py`**: Utility functions (parsing, prompts)
- **`visualize.py`**: Plotting and visualization

## Key Components

### Evolution Loop (`main.py`)

The main evolution loop:
1. Initializes population with seed organism
2. For each step:
   - Samples parent and inspirations
   - Generates child via mutation or crossover
   - Evaluates child fitness via DES
   - Applies selection pressure
   - Updates adaptive mutation temperature

### Population Database (`database.py`)

Thread-safe organism storage with:
- Fitness-weighted sampling
- Top-k tracking for inspirations
- Population pruning to limit memory

### LLM Mutator (`mutator.py`)

Uses GPT-5.1 to generate intelligent mutations:
- Adaptive mutation strength (small/medium/large/radical)
- Incorporates parent, inspirations, and lineage history
- Temperature-based exploration control

### Queue Simulator (`queue_simulator.py`)

Discrete-event simulation implementing Che-Tercieux model:
- Agent belief updating (Bayesian)
- Information rule handling
- Stationary distribution computation

## Configuration

Key parameters (set via CLI or defaults):
- `--steps`: Number of evolution steps
- `--sim-time`: Simulation timesteps per evaluation (default: 10,000)
- `--workers`: Parallel workers (default: 1)
- `--selection-pressure`: Fraction of children to keep (default: 0.1)
- `--elite-size`: Number of elite organisms (default: 10)

See `python main.py --help` for full parameter list.

## Output

- Console output: Progress updates, best fitness, statistics
- JSON file (if `--output` specified): Complete results
- Visualizations (if `--visualize`): Fitness progression, population stats

