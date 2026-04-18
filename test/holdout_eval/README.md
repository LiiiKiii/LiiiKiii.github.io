# Holdout Validation Evaluation

This is the **current main evaluation** for AI-Pedia, implementing a holdout validation approach inspired by ML train/validation splits.

## Core Idea

- **Input Set**: Documents used by both systems to generate recommendations
- **Validation Set**: Independent documents used to evaluate recommendation quality
- **Comparison**: Baseline LLM vs AI-Pedia structured pipeline

## Methods Compared

### Baseline: Free-form LLM with Web Search
GPT-4o with browsing capability but **no structured pipeline**:
- ❌ No TF-IDF keyword extraction
- ❌ No source whitelist
- ❌ No modality balancing
- ❌ No content-based ranking

### AI-Pedia: Structured Pipeline
Full system with structured retrieval:
- ✅ TF-IDF + MMR keyword extraction
- ✅ Multi-source retrieval (Wikipedia, arXiv, Scholar, YouTube, GitHub)
- ✅ 5-5-5 modality balancing
- ✅ Content-Based Filtering (CBF) ranking

## Evaluation Metrics

1. **Simpson's Diversity Index**: Measures domain/type diversity
2. **Validation Set Relevance**: Cosine similarity to validation documents
3. **Coverage Score**: Proportion of validation docs covered

## Running

```bash
cd /Users/macbook/Desktop/AI-Pedia/Project
export OPENAI_API_KEY="your-api-key"  # Optional
python test/holdout_eval/holdout_evaluator.py \
    --corpora-root data/holdout_corpora \
    --output test/holdout_eval/results
```

## Output Files

- `aggregate_results.json`: Combined results across all corpora
- `*_results.json`: Per-corpus results
- `holdout_evaluation_tables.tex`: LaTeX table for paper
- `holdout_evaluation_summary.csv`: CSV summary
- `holdout_diversity.png`: Diversity comparison chart
- `holdout_relevance.png`: Relevance comparison chart
- `holdout_coverage.png`: Coverage comparison chart

## Data Format

```
data/holdout_corpora/
├── nlp_transformers/
│   ├── input/     # 10-20 .txt files
│   └── validation/  # 10-50 .txt files
├── vision_representation/
│   ├── input/
│   └── validation/
└── foundations_ml/
    ├── input/
    └── validation/
```
