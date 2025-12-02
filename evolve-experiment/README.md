# Queue Model Evolution Optimizer

An evolutionary optimization system for the Che-Tercieux queue model using LLM-based agents to explore, refine, and act on configuration space.

## Overview

This system uses a multi-agent approach (Explore-Refine-Act) powered by OpenAI's language models to optimize queue system parameters for maximum social welfare.

## Installation

```bash
# Install dependencies
pip install openai pydantic click

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

```bash
# Run with default settings (10 iterations)
python main.py

# Or make it executable and run directly
chmod +x main.py
./main.py
```

### CLI Options

```bash
# Run 20 iterations with longer simulations
python main.py --iterations 20 --sim-time 50000

# Save results to JSON file
python main.py --output results.json

# Quiet mode (minimal output)
python main.py --quiet

# Use specific models for each phase
python main.py -m gpt-4o -m gpt-4o-mini -m gpt-4o-mini

# Combine options
python main.py -n 15 -t 20000 -o output.json --quiet
```

### Command-Line Arguments

- `-n, --iterations N`: Number of optimization iterations (default: 10)
- `-t, --sim-time T`: Simulation time per iteration (default: 10000.0)
- `-m, --models`: Model names (provide 3 times for explore, refine, act phases)
- `-o, --output FILE`: Save results to JSON file
- `-q, --quiet`: Minimal output (only show final results)
- `--task TEXT`: Custom task description for optimization
- `--help`: Show help message and exit

## How It Works

1. **Explore Phase**: Agent analyzes past configurations and generates candidate parameters
2. **Refine Phase**: Agent evaluates candidates and selects optimal configuration
3. **Act Phase**: Agent makes final decision and commits to parameter values
4. **Simulation**: Configuration is converted to queue model and simulated
5. **Evaluation**: Welfare score is calculated and stored in database
6. **Iteration**: Process repeats, with agents learning from past attempts

## Model Parameters

The system optimizes:

- `V_surplus`: Net surplus from service (V > 0)
- `C_waiting_cost`: Per-period waiting cost (C > 0)
- `R_provider_profit`: Profit per served agent (R > 0)
- `alpha_weight`: Weight on agents' welfare [0, 1]
- `arrival_rate_fn`: State-dependent arrival rate
- `service_rate_fn`: State-dependent service rate
- `entry_rule_fn`: Entry probability by queue length
- `exit_rule_fn`: Exit rules (rate and probability)
- `queue_discipline`: FCFS, LIFO, or SIRO
- `information_rule`: Information structure for agents

## Output

The optimization outputs:
- Best welfare score achieved
- Best configuration parameters
- All attempts with scores and observations
- Optional JSON file with complete results

## Example Output

```
Starting evolution with 10 iterations
Using models: ['gpt-4o-mini', 'gpt-4o-mini', 'gpt-4o-mini']
Simulation time per iteration: 10000.0

============================================================
ITERATION 1/10
============================================================
Running agent pipeline (Explore -> Refine -> Act)...
  Confidence: medium
  Reasoning: Balancing provider profit with agent utility...
Converting config to CheTercieuxQueueModel...
Running simulation (10000.0 time units)...
Evaluating welfare score...

NEW BEST SCORE: 14.2345

Iteration Summary:
  Welfare Score (W): 14.2345
  Expected Queue Length E[k]: 2.15
  Expected Service Flow E[mu_k]: 2.00
  Agents Served: 2003
  V_surplus: 10.00, C_cost: 1.00
  R_profit: 5.00, alpha: 0.50

...

============================================================
EVOLUTION COMPLETE
============================================================

Best Welfare Score: 15.3421
Total Attempts: 10
```

## Architecture

### Components

- `main.py`: CLI interface and evolution loop
- `ensemble.py`: Multi-agent system (Explore-Refine-Act)
- `queue_simulator.py`: Discrete event simulation
- `evaluator.py`: Welfare score calculation
- `utils.py`: Helper functions and prompt builders
- `database.py`: Attempt storage and retrieval
- `evolve_types.py`: Type definitions

### Agent Pipeline

```
Database (past attempts)
    ↓
Explore Agent → candidate configs
    ↓
Refine Agent → recommended config
    ↓
Act Agent → final config
    ↓
Simulator → results
    ↓
Evaluator → welfare score
    ↓
Database (store attempt)
```

## License

MIT

