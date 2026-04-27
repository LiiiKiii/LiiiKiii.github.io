# LLM-with-Browsing Baseline Evaluation (v3)

## Purpose

This module implements an LLM-with-browsing baseline for comparison with the AI-Pedia pipeline. It is designed to break the circular-evaluation problem identified in the original within-system evaluation while keeping result provenance explicit.

## The Problem with Self-Evidence

The original evaluation compared `Simple Frequency Baseline` vs `AI-Pedia TF-IDF+MMR` using metrics (keyword AI-relevance, resource AI-relevance) that rely on the same AI-keyword dictionaries used by the pipeline itself. This creates a risk of circularity: the pipeline generates keywords that match the evaluation criteria, making the comparison self-reinforcing rather than genuinely comparative.

## The Solution: LLM-with-Browsing Baseline (v3)

**Key change from v2**: the baseline now has two transparent execution modes:

- `live_openai`: uses the OpenAI Responses API with the hosted `web_search` tool when `OPENAI_API_KEY` is configured.
- `synthetic_dry_run`: uses deterministic local fixtures when no API key is available or a live call fails. These rows are useful for evaluator development and dissertation figure layout, but they are not live LLM outputs.

The baseline still lacks the structural retrieval components that define AI-Pedia. This keeps the comparison focused: both modes model free-form search behaviour, but only AI-Pedia uses structured keyword extraction, a source whitelist, and modality balancing.

### Baseline constraints

- **No structured keyword extraction**: the LLM freely decides what to search for.
- **No source whitelisting**: any domain is acceptable.
- **No modality balancing**: text/video/code are not enforced.
- **No de-duplication**: results are returned as found.
- **No systematic URL freshness check**: live mode may label links uncertain; dry-run fixtures include plausible uncertain/broken cases.

### AI-Pedia's advantages

- Structured TF-IDF+MMR keyword extraction.
- Source whitelist (arXiv, Wikipedia, YouTube, GitHub, and implementation/tutorial sources).
- 5-5-5 modality balancing.
- Live retrieval from verified source APIs where available.

## Files

| File | Description |
|------|-------------|
| `llm_baseline.py` | Baseline implementation: live OpenAI web search or labelled dry-run fixture |
| `evaluator.py` | Side-by-side comparison runner |
| `results/` | Auto-generated: JSON, CSV, LaTeX tables |

## Usage

```bash
# Full comparison. Uses live OpenAI mode when OPENAI_API_KEY is set.
export OPENAI_API_KEY=your_key_here
python evaluator.py --corpora-root ../data/test_corpora --output results/

# Strict live comparison. Fails instead of falling back to dry-run fixtures.
export OPENAI_API_KEY=your_key_here
export LLM_BASELINE_REQUIRE_LIVE=1
python evaluator.py --corpora-root ../data/test_corpora --output results/

# Reuse cached results (no API calls).
python evaluator.py --skip-llm --corpora-root ../data/test_corpora --output results/
```

## Current Live Results

| Metric | LLM-with-Browsing | AI-Pedia | Winner |
|--------|------------------|----------|--------|
| Live URLs | **100%** | **100%** | Tie |
| Broken URLs | 0% | 0% | Tie |
| Domain diversity | **0.867** | 0.862 | Tie |
| Authoritative sources | 37.8% | **66.7%** | AI-Pedia |
| Resource-type diversity | 0.476 | **0.667** | AI-Pedia |
| Code resources | 3 total | **15 total** | AI-Pedia |
| Video resources | 11 total | **15 total** | AI-Pedia |
| Balanced 5-5-5 | No | Yes | AI-Pedia |

These numbers come from the current `results/llm_comparison_results.json`, which records `llm_run_modes=["live_openai"]`. The deterministic dry-run mode remains available for evaluator development and figure-layout testing, but it should not be reported as live LLM evidence.

## Why This Design Is Defensible

1. **External metrics**: URL health, authority domain labels, modality counts are model-independent facts
2. **Fair comparison**: both systems have browsing; only AI-Pedia has structure
3. **No circularity**: baseline doesn't share components with AI-Pedia
4. **Reproducibility**: live outputs are saved as JSON/CSV/LaTeX artefacts, while deterministic dry-run data keeps the evaluator testable when API access is unavailable.
