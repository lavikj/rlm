# NVDLA Question-Answering Dataset & Evaluation

Tools for building and evaluating RLM's question-answering performance on NVDLA documentation.

## What It Does

1. **Build Dataset** (`build_qa_dataset.py`): Extracts snippets from NVDLA markdown files and generates questions using an LLM (~180 QA pairs)
2. **Evaluate RLM** (`evaluate_rlm_qa.py`): Uses RLM to answer questions with full documentation context, evaluates quality
3. **Analyze Results** (`analyze_results.py`): Comprehensive performance analysis and statistics

## Prerequisites

```bash
# Ensure RLM is installed (from project root)
uv pip install -e .

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

## Quick Start

```bash
# Navigate to the qa_evaluation directory
cd qa_evaluation

# First, run smoke test to verify setup
./run_qa_evaluation.sh test

# Then run full pipeline
./run_qa_evaluation.sh full
```

**Other commands:** `build`, `evaluate`, `analyze`, `clean`

## Manual Step-by-Step

If you prefer to run each step individually:

```bash
# Navigate to the qa_evaluation directory
cd qa_evaluation

# Step 1: Build dataset (~180 QA pairs)
python build_qa_dataset.py

# Step 2: Evaluate RLM
python evaluate_rlm_qa.py

# Step 3: Analyze results
python analyze_results.py
```

## Configuration

Edit the scripts to customize:

**Dataset size** (`build_qa_dataset.py`):
```python
max_snippets_per_file=10,  # More = larger dataset
min_paragraph_length=150   # Lower = more snippets
```

**Test subset** (`evaluate_rlm_qa.py`):
```python
MAX_QUESTIONS = 10  # Limit for quick testing
```

**LLM backends** (`evaluate_rlm_qa.py`):
```python
# Use Anthropic
backend = "anthropic"
backend_kwargs = {
    "api_key": os.getenv("ANTHROPIC_API_KEY"),
    "model_name": "claude-3-5-sonnet-20241022"
}

# Use Gemini
backend = "gemini"
backend_kwargs = {
    "api_key": os.getenv("GEMINI_API_KEY"),
    "model_name": "gemini-2.0-flash-exp"
}
```

## Cost Estimation

**Full pipeline (~180 QA pairs):** $12-25
- Dataset creation: $2-5
- RLM answers: $5-10
- Evaluation: $5-10

**Smoke test (3 questions):** < $1

## Use Cases

- Benchmark RLM's long-context QA performance
- Compare different LLM backends (GPT-4, Claude, Gemini)
- Test different configurations (temperature, max_tokens)
- Track performance over time
- Identify which documentation areas need improvement

