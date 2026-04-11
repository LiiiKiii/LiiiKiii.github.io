# AI-Pedia Evaluation Method

This document describes the **current reproducible evaluation setup** for AI-Pedia.

## 1. Evaluation Goal

The project evaluates whether AI-Pedia improves the path from learner documents to external reinforcement resources by examining three stages of the pipeline:

1. **Keyword extraction quality**
2. **Resource retrieval quality**
3. **Ranking quality of the final recommendations**

The current dissertation evaluation is a **focused multi-corpus pilot**, designed to test the system across several coherent learner scenarios rather than one mixed-topic corpus.

## 2. Repository Layout for Evaluation

The evaluation assets are split across code, input data, and generated outputs:

- Evaluation code: `test/evaluation_pipeline/`
- Performance benchmarking code: `test/performance/`
- Focused evaluation corpora: `data/test_corpora/`
- Legacy single-corpus input: `data/test_corpus/`
- Generated evaluation outputs: `test/evaluation_pipeline/results/`
- Generated performance outputs: `test/performance/results/`

## 3. Current Focused Corpora

The current setup uses three focused corpora:

- `foundations_ml`
- `nlp_transformers`
- `vision_representation`

Each corpus contains ten AI/ML-related text documents. The goal is not to simulate the full diversity of the whole field in one bundle, but to test the recommender in several coherent topic clusters that better match realistic learner scenarios.

## 4. Main Scripts

### 4.1 Quantitative evaluation

- `test/evaluation_pipeline/evaluator.py`
- `test/evaluation_pipeline/metrics.py`
- `test/evaluation_pipeline/config.py`

### 4.2 Local deterministic performance benchmarking

- `test/performance/run_performance_tests.py`

## 5. How to Run the Current Evaluation

From the project root:

```bash
python test/evaluation_pipeline/evaluator.py --use-focused-corpora --reuse-cache
```

This regenerates the main evaluation outputs used by the dissertation.

For local deterministic performance benchmarking:

```bash
python test/performance/run_performance_tests.py
```

## 6. Main Evaluation Outputs

### 6.1 Aggregate outputs

- `test/evaluation_pipeline/results/evaluation_results.json`
- `test/evaluation_pipeline/results/evaluation_tables.tex`
- `test/evaluation_pipeline/results/keyword_metrics.csv`
- `test/evaluation_pipeline/results/resource_metrics.csv`
- `test/evaluation_pipeline/results/corpus_overview.csv`

### 6.2 Per-corpus outputs

- `test/evaluation_pipeline/results/<corpus_name>/evaluation_results.json`
- `test/evaluation_pipeline/results/<corpus_name>/raw_search_results.json`

### 6.3 Performance outputs

- `test/performance/results/performance_results.json`
- `test/performance/results/performance_summary.csv`
- `test/performance/results/performance_table.tex`

## 7. Evaluation Questions

The current evaluation addresses three practical questions:

### RQ1. Does the keyword extraction stage produce useful topic signals?

This is assessed using:

- **Coverage**
- **AI relevance**
- **Keyword diversity**

### RQ2. Does multi-source retrieval produce useful candidate pools?

This is assessed using:

- **AI relevance of retrieved resources**
- **Authority score**
- **Valid URL rate**
- **Modality/platform diversity**

### RQ3. Does ranking improve the final recommendation set?

This is assessed by comparing raw retrieval, unranked top-K, and ranked top-K outputs using:

- **AI relevance**
- **Authority score**
- **Noise reduction**
- **Valid URL rate**

## 8. Metric Interpretation

### 8.1 Coverage

Coverage measures how many documents are touched by the extracted keywords. In the current focused-corpus setup, perfect coverage is not treated as the only sign of quality, because more specific multi-word phrases may intentionally trade literal coverage for better retrieval-oriented topic precision.

### 8.2 AI relevance

AI relevance measures whether extracted keywords or recommended resources remain genuinely within the intended AI/ML domain.

### 8.3 Authority score

Authority score estimates how many returned resources come from stronger or more trustworthy sources.

### 8.4 Noise reduction

Noise reduction captures how aggressively the pipeline compresses a large raw candidate pool into a smaller final recommendation set.

## 9. Reproducibility Principle

The evaluation workflow is designed so that the paper does not depend on manually copied spreadsheet numbers.

Instead, the intended process is:

1. run the evaluation scripts,
2. regenerate JSON, CSV, LaTeX, and figure outputs,
3. include those outputs in the dissertation.

This keeps the paper synchronized with the actual code behaviour.

## 10. Scope and Limitations

The current results should be understood as a **reproducible pilot evaluation**, not as a full user study. The system has been tested across multiple focused corpora and deterministic performance checks, but future work should still include human annotation, longitudinal study design, and broader deployment-oriented robustness checks.
