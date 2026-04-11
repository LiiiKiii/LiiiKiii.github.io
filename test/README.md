# AI-Pedia Test Suite Overview

This directory contains the **current reproducible evaluation and performance scripts** used by the project.

## What lives in `test/`

- `evaluation_pipeline/`  
  Main quantitative evaluation code for keyword extraction, retrieval quality, and ranking quality.

- `performance/`  
  Lightweight local performance benchmarks for the deterministic stages of the pipeline.

No legacy evaluation-planning folders are kept here anymore. The goal is for `test/` to reflect the **current runnable evaluation setup**, not older abandoned designs.

## Important path clarification

The **evaluation scripts are in `test/`**, but the **input corpora are not stored inside `test/`**.

Current layout:

- Evaluation code: `test/evaluation_pipeline/`
- Performance code: `test/performance/`
- Focused evaluation corpora: `data/test_corpora/`
- Legacy single-corpus input: `data/test_corpus/`
- Generated evaluation outputs: `test/evaluation_pipeline/results/`
- Generated performance outputs: `test/performance/results/`

## Current evaluation design

The dissertation's current evaluation uses **three focused learner-scenario corpora**:

- `data/test_corpora/foundations_ml/`
- `data/test_corpora/nlp_transformers/`
- `data/test_corpora/vision_representation/`

Each corpus contains 10 AI/ML-themed text documents. The evaluation script can run either:

1. **single-corpus mode** for ad hoc testing, or
2. **focused multi-corpus mode** for the current dissertation evaluation.

## How to run the current evaluation

From the project root:

```bash
python test/evaluation_pipeline/evaluator.py --use-focused-corpora --reuse-cache
```

This regenerates the aggregate results and per-corpus outputs used for the paper.

## How to run the performance benchmark

From the project root:

```bash
python test/performance/run_performance_tests.py
```

## Main generated files

### Evaluation outputs

- `test/evaluation_pipeline/results/evaluation_results.json`
- `test/evaluation_pipeline/results/evaluation_tables.tex`
- `test/evaluation_pipeline/results/keyword_metrics.csv`
- `test/evaluation_pipeline/results/resource_metrics.csv`
- `test/evaluation_pipeline/results/corpus_overview.csv`
- `test/evaluation_pipeline/results/<corpus_name>/...`

### Performance outputs

- `test/performance/results/performance_results.json`
- `test/performance/results/performance_summary.csv`
- `test/performance/results/performance_table.tex`

## Why this matters

If you need to explain the repository structure in the dissertation or viva, the most accurate summary is:

> The testing and evaluation scripts live in `test/`, the evaluation corpora live in `data/test_corpora/`, and the generated quantitative outputs are written back to `test/.../results/` for direct inclusion in the dissertation.
