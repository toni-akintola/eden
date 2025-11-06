# Eden - Alpha Evolve Style AI System

An AI system that uses a three-stage pipeline (Explore-Refine-Act) to iteratively solve problems. Currently implements a number-guessing game as a demonstration.

## Overview

This system mimics the Alpha Evolve approach with three distinct stages:

1. **🔍 EXPLORE** - Analyzes past attempts, identifies patterns, and generates multiple candidate solutions
2. **🔧 REFINE** - Evaluates candidates and optimizes the strategy into a single best approach  
3. **🎯 ACT** - Makes the final decision and commits to an action

Each stage can use a different AI model, allowing you to leverage specialized strengths.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

### Interactive Mode (Recommended)

Simply run the script and follow the prompts:

```bash
python main.py
```

You'll be prompted to:
- Set a secret number (1-100)
- Choose maximum attempts
- Select AI models for each stage

### Command Line Mode

Pass all parameters via command line flags:

```bash
python main.py --secret-number 42 --max-attempts 10 \
  --explore-model gpt-4o-mini \
  --refine-model gpt-4o-mini \
  --act-model gpt-4o-mini
```

### Available Options

```
Options:
  -s, --secret-number INTEGER  The secret number to guess (1-100)
  -m, --max-attempts INTEGER   Maximum number of attempts allowed
  --explore-model TEXT         Model to use for EXPLORE stage
  --refine-model TEXT          Model to use for REFINE stage
  --act-model TEXT             Model to use for ACT stage
  --help                       Show this message and exit
```

### Examples

**Quick test with gpt-4o-mini:**
```bash
python main.py -s 73 -m 15 \
  --explore-model gpt-4o-mini \
  --refine-model gpt-4o-mini \
  --act-model gpt-4o-mini
```

**Mix different models:**
```bash
python main.py -s 50 -m 10 \
  --explore-model gpt-4o \
  --refine-model gpt-4o-mini \
  --act-model gpt-4o-mini
```

**Interactive mode (no flags):**
```bash
python main.py
```

## Testing Prompts

To preview the prompts without making API calls:

```bash
python test_prompts.py
```

This shows you exactly what each AI model sees at each stage.

## Project Structure

```
eden/
├── main.py           # CLI entry point
├── ensemble.py       # Three-stage pipeline implementation
├── constants.py      # Prompt builders for each stage
├── database.py       # Attempt storage and retrieval
├── evaluator.py      # Evaluation utilities
├── test_prompts.py   # Test prompts without API calls
└── requirements.txt  # Python dependencies
```

## How It Works

1. **Initial State**: The AI has no information about the secret number
2. **Explore**: Analyzes previous attempts (if any) and proposes 2-3 candidate guesses with reasoning
3. **Refine**: Evaluates the candidates and selects the optimal guess using strategic thinking (e.g., binary search)
4. **Act**: Makes the final decision with a sanity check
5. **Feedback**: Receives feedback ("too high", "too low", etc.) and stores it in the database
6. **Repeat**: Goes back to step 2, now with historical context to learn from

## Extending to Complex Tasks

The current implementation demonstrates number guessing, but the architecture is designed for more complex tasks:

- **Code optimization**: Explore different approaches, refine to best algorithm, act by implementing it
- **Problem solving**: Explore solution space, refine to optimal strategy, act by executing steps
- **Creative tasks**: Explore multiple ideas, refine to best concept, act by producing final output

See the prompt design in `constants.py` to understand how to adapt prompts for different tasks.

## Future Enhancements

- [ ] Meta-learning layer that adapts strategy based on success rates
- [ ] Ensemble voting where multiple models vote on decisions
- [ ] Reflection stage after attempts to improve future performance
- [ ] Strategy library that models can reference
- [ ] Confidence calibration for better uncertainty estimates

## License

MIT

