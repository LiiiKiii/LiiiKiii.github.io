# AI-Pedia Evaluation Pipeline

This directory contains the **current quantitative evaluation pipeline** for AI-Pedia.

It is the code used to generate the project's reproducible keyword, retrieval, ranking, table, and chart outputs.

## Scope

The evaluation measures three parts of the pipeline:

1. **Keyword extraction**
2. **Multi-source resource retrieval**
3. **CBF-based ranking and recommendation shaping**

The current dissertation evaluation is based on **focused multi-corpus testing**, rather than a single mixed-topic bundle.

## Input data locations

The scripts live here, but the corpora are stored outside this folder:

- Focused corpora root: `data/test_corpora/`
- Legacy single-corpus input: `data/test_corpus/`

Current focused corpora:

- `data/test_corpora/foundations_ml/`
- `data/test_corpora/nlp_transformers/`
- `data/test_corpora/vision_representation/`

## Key files

- `config.py`  
  Paths and configuration constants for evaluation.

- `metrics.py`  
  Metric definitions for keyword quality, AI relevance, authority, URL validity, and related comparisons.

- `evaluator.py`  
  Main runner. Supports both single-corpus and focused multi-corpus evaluation.

- `results/`  
  Regenerated outputs for the dissertation, including aggregate summaries and per-corpus folders.

## Evaluation modes

### 1. Focused multi-corpus mode (current dissertation setup)

Run from the project root:

```bash
python test/evaluation_pipeline/evaluator.py --use-focused-corpora --reuse-cache
```

This runs evaluation across all corpora in `data/test_corpora/` and writes:

- aggregate JSON summary
- CSV metric tables
- LaTeX tables
- paper-ready figures
- per-corpus outputs under `results/<corpus_name>/`

### 2. Single-corpus mode

```bash
python test/evaluation_pipeline/evaluator.py --corpus data/test_corpus
```

This mode is still useful for quick debugging or smaller pilot runs.

## Main outputs

Aggregate outputs:

- `results/evaluation_results.json`
- `results/evaluation_tables.tex`
- `results/keyword_metrics.csv`
- `results/resource_metrics.csv`
- `results/corpus_overview.csv`

Per-corpus outputs:

- `results/<corpus_name>/evaluation_results.json`
- `results/<corpus_name>/raw_search_results.json`

## Core metrics

### Keyword-stage metrics

- **Coverage**: how many documents are touched by the extracted keywords
- **AI relevance**: how strongly extracted terms align with AI/ML concepts
- **Diversity**: how non-redundant the keyword set is

### Resource-stage metrics

- **AI relevance**: proportion of retrieved or ranked resources judged AI-related
- **Authority score**: proportion of resources from stronger sources
- **Valid URLs**: proportion of URLs that pass validation checks
- **Noise reduction**: candidate-pool compression from raw retrieval to final recommendation set
- **Cross-platform / modality diversity**: whether text, video, and code are preserved

## Current interpretation

The current evaluation is designed to show that:

- phrase-based keyword extraction produces more useful topic signals than a naive baseline,
- retrieval produces broad multimodal candidate pools,
- ranking improves the quality of the final recommendation set rather than merely reshuffling links.

## Reproducibility note

The intended workflow is:

1. Run the evaluation script.
2. Regenerate outputs in `test/evaluation_pipeline/results/`.
3. Use the generated tables and figures in the paper.

This avoids hand-copied metrics and keeps the dissertation aligned with the actual code outputs.
