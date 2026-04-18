# AI-Pedia Test Suite Overview

This directory contains the **evaluation and performance scripts** used by the project.

## Directory Structure

```
test/
├── README.md           # This file
├── evaluation_pipeline/   # Phase 1: Ablation study (keyword extraction, retrieval, ranking)
├── llm_baseline_eval/      # Phase 2: LLM-with-browsing baseline comparison
├── holdout_eval/           # Phase 3: Holdout validation (current main evaluation)
└── performance/           # Performance benchmarks
```

## Current Evaluation Design

The dissertation uses **three evaluation phases**:

### Phase 1: Ablation Study (`evaluation_pipeline/`)
Tests whether each component of AI-Pedia contributes:
- Simple frequency baseline vs TF-IDF+MMR for keyword extraction
- Raw search vs ranked results for recommendation quality

### Phase 2: LLM Baseline (`llm_baseline_eval/`)
External comparison with GPT-4o using browsing capability:
- Baseline: GPT-4o with web search (no structured pipeline)
- Compares URL health, domain diversity

### Phase 3: Holdout Validation (`holdout_eval/`) ← **CURRENT MAIN EVALUATION**
Holdout validation inspired by ML train/validation split:
- Input Set (~20 docs): Documents used to generate recommendations
- Validation Set (~50 docs): Independent documents for evaluation
- Baseline: Free-form LLM with web search
- AI-Pedia: Structured pipeline (TF-IDF+MMR → multi-source → CBF)

**Metrics**: Simpson's Diversity Index, Validation Set Relevance, Coverage Score

## Data Locations

- Evaluation corpora: `data/holdout_corpora/`
- Test corpora: `data/test_corpora/`
- Generated outputs: Each test folder has a `results/` subdirectory

## How to Run

### Current Main Evaluation (Holdout Validation)
```bash
cd /Users/macbook/Desktop/AI-Pedia/Project
export OPENAI_API_KEY="your-api-key"  # Optional, fallback results available
python test/holdout_eval/holdout_evaluator.py \
    --corpora-root data/holdout_corpora \
    --output test/holdout_eval/results
```

### Ablation Study
```bash
python test/evaluation_pipeline/evaluator.py --use-focused-corpora
```

### Performance Benchmark
```bash
python test/performance/run_performance_tests.py
```

## Key Files

| Directory | Main Script | Purpose |
|----------|-------------|---------|
| `holdout_eval/` | `holdout_evaluator.py` | Main holdout validation evaluation |
| `evaluation_pipeline/` | `evaluator.py` | Ablation study |
| `llm_baseline_eval/` | `llm_baseline.py` | LLM baseline comparison |
| `performance/` | `run_performance_tests.py` | Performance benchmarks |

## LLM Configuration

AI-Pedia uses LLM for:
- **Summary generation**: `backend/core/ai_summarizer.py` uses `gpt-4o-mini`
- **Baseline evaluation**: `test/holdout_eval/holdout_evaluator.py` uses `gpt-4o`

Both read API key from `OPENAI_API_KEY` environment variable.
