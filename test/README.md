# AI-Pedia Test Suite Overview

This directory contains the **evaluation and performance scripts** used by the project.

## Directory Structure

```
test/
├── README.md           # This file
├── evaluation_pipeline/   # Phase 1: Ablation study (keyword extraction, retrieval, ranking)
├── llm_baseline_eval/      # Phase 2: current live LLM-with-browsing baseline comparison
└── performance/           # Performance benchmarks
```

## Current Evaluation Design

The dissertation uses **two active evaluation phases**:

### Phase 1: Ablation Study (`evaluation_pipeline/`)
Tests whether each component of AI-Pedia contributes:
- Simple frequency baseline vs TF-IDF+MMR for keyword extraction
- Raw search vs ranked results for recommendation quality

### Phase 2: Current Live LLM Baseline (`llm_baseline_eval/`)
External comparison with GPT-4o using browsing capability:
- Baseline: GPT-4o with hosted web search (no structured pipeline)
- Compares URL health, domain diversity, authority coverage, and modality balance
- Strict live mode records `run_mode=live_openai` and fails rather than silently falling back

## Data Locations

- Test corpora: `data/test_corpora/`
- Generated outputs: Each test folder has a `results/` subdirectory

## How to Run

### Current Main Evaluation (Live LLM Baseline)
```bash
cd /Users/macbook/Desktop/AI-Pedia/Project/Code
export OPENAI_API_KEY="your-api-key"
export LLM_BASELINE_REQUIRE_LIVE=1
python test/llm_baseline_eval/evaluator.py \
    --corpora-root data/test_corpora \
    --output test/llm_baseline_eval/results
```

### Ablation Study
```bash
python test/evaluation_pipeline/evaluator.py
```

### Performance Benchmark
```bash
python test/performance/run_performance_tests.py
```

## Key Files

| Directory | Main Script | Purpose |
|----------|-------------|---------|
| `llm_baseline_eval/` | `evaluator.py` | Current live GPT-4o baseline comparison |
| `evaluation_pipeline/` | `evaluator.py` | Ablation study |
| `performance/` | `run_performance_tests.py` | Performance benchmarks |

## LLM Configuration

AI-Pedia uses LLM for:
- **Summary generation**: `backend/core/ai_summarizer.py` uses `gpt-4o-mini`
- **Baseline evaluation**: `test/llm_baseline_eval/llm_baseline.py` uses `gpt-4o`

Both read API key from `OPENAI_API_KEY` environment variable.
