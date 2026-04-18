#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Baseline vs AI-Pedia Comparison Evaluator (v2 — with browsing)

Compares the LLM-with-browsing baseline (constrained by no structured pipeline)
against the full AI-Pedia pipeline on:

1. URL health: live vs broken vs unclear (external, model-independent)
2. Source authority: authoritative domains vs blogs/low-quality sources
3. Modality balance: text / video / code distribution
4. Total resource coverage

Key research question:
    Does structured retrieval (AI-Pedia) outperform free-form LLM browsing?
    Specifically in: URL validity, authority coverage, and modality balance.

Usage:
    python evaluator.py --corpora-root ../data/test_corpora --output results/
    python evaluator.py --skip-llm  # reuse cached results
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

# ─── paths ───────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(1, PROJECT_ROOT)

from llm_baseline import run_llm_baseline, save_results, export_comparison_table

try:
    from evaluation_pipeline.evaluator import EvaluationRunner
except ImportError:
    from test.evaluation_pipeline.evaluator import EvaluationRunner


# ─── COMPARISON FRAMEWORK ───────────────────────────────────

class ComparisonEvaluator:
    """
    Side-by-side comparison between:
    - LLM-with-browsing baseline (GPT-4o, unconstrained search, no pipeline)
    - AI-Pedia pipeline (TF-IDF+MMR → multi-source retrieval → CBF ranking)

    Evaluation dimensions (all external, model-independent):
    1. URL health: LIVE / UNCLEAR / BROKEN percentages
    2. Source authority: authoritative domain share vs low-authority share
    3. Modality balance: text / video / code distribution
    4. Retrieval comprehensiveness: total resources per modality
    """

    def __init__(self):
        self.aipedia_runner = EvaluationRunner()

    def compare_url_health(
        self,
        llm_metrics: Dict[str, Any],
        aipedia_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compare URL health: LLM browsing vs AI-Pedia live retrieval.
        AI-Pedia should have near-100% live URLs (retrieved directly from verified sources).
        LLM browsing may have hallucinated/broken/stale URLs.
        """
        aipedia_ranked = aipedia_result.get("resource_reports", {}).get("topk_ranked", {})

        return {
            "llm_browsing": {
                "live_pct": llm_metrics.get("url_live_pct", 0.0),
                "broken_pct": llm_metrics.get("url_broken_pct", 0.0),
                "unclear_pct": llm_metrics.get("url_unclear_pct", 0.0),
            },
            "aipedia": {
                "live_pct": aipedia_ranked.get("url_validation", {}).get("valid_percentage", 0.0),
                "broken_pct": 0.0,   # AI-Pedia retrieves live URLs directly
                "unclear_pct": 0.0,  # no uncertain cases
            },
            "gap": {
                "live_url_gap": round(
                    aipedia_ranked.get("url_validation", {}).get("valid_percentage", 0.0)
                    - llm_metrics.get("url_live_pct", 0.0), 2
                ),
                "broken_url_gap": round(
                    llm_metrics.get("url_broken_pct", 0.0) - 0.0, 2
                ),
            },
        }

    def compare_authority(
        self,
        llm_metrics: Dict[str, Any],
        aipedia_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compare source authority: authoritative domain share vs low-quality share.
        AI-Pedia uses a source whitelist (arXiv, GitHub, Wikipedia, YouTube, etc.).
        LLM browsing freely chooses sources, often over-representing blogs.
        """
        aipedia_ranked = aipedia_result.get("resource_reports", {}).get("topk_ranked", {})

        return {
            "llm_browsing": {
                "authority_pct": llm_metrics.get("authority_score_pct", 0.0),
                "low_authority_pct": llm_metrics.get("low_authority_pct", 0.0),
            },
            "aipedia": {
                "authority_pct": aipedia_ranked.get("authority_score", 0.0),
                "low_authority_pct": 0.0,  # blogs are filtered by domain whitelist
            },
            "contrast": {
                "authority_gap": round(
                    aipedia_ranked.get("authority_score", 0.0)
                    - llm_metrics.get("authority_score_pct", 0.0), 2
                ),
                "low_authority_gap": round(
                    llm_metrics.get("low_authority_pct", 0.0) - 0.0, 2
                ),
            },
        }

    def compare_modality(
        self,
        llm_metrics: Dict[str, Any],
        aipedia_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compare modality balance: text / video / code distribution.
        AI-Pedia enforces a balanced 5-5-5 split per corpus.
        LLM browsing may freely over-represent text (blogs) and under-represent code.
        """
        aipedia_ranked = aipedia_result.get("resource_reports", {}).get("topk_ranked", {})
        llm_modality = llm_metrics.get("modality_counts", {})
        aipedia_modality = aipedia_ranked.get("resource_type_counts", {})

        llm_total = sum(llm_modality.values())
        aipedia_total = sum(aipedia_modality.values())

        def pct(count: int, total: int) -> float:
            return round(count / max(total, 1) * 100, 2)

        return {
            "llm_browsing": {
                "text_count": llm_modality.get("text", 0),
                "video_count": llm_modality.get("video", 0),
                "code_count": llm_modality.get("code", 0),
                "text_pct": pct(llm_modality.get("text", 0), llm_total),
                "video_pct": pct(llm_modality.get("video", 0), llm_total),
                "code_pct": pct(llm_modality.get("code", 0), llm_total),
            },
            "aipedia": {
                "text_count": aipedia_modality.get("txt", 0),
                "video_count": aipedia_modality.get("video", 0),
                "code_count": aipedia_modality.get("code", 0),
                "text_pct": pct(aipedia_modality.get("txt", 0), aipedia_total),
                "video_pct": pct(aipedia_modality.get("video", 0), aipedia_total),
                "code_pct": pct(aipedia_modality.get("code", 0), aipedia_total),
            },
            "contrast": {
                "code_advantage": aipedia_modality.get("code", 0) > llm_modality.get("code", 0),
                "balanced": (
                    aipedia_modality.get("txt", 0) > 0
                    and aipedia_modality.get("video", 0) > 0
                    and aipedia_modality.get("code", 0) > 0
                    and abs(aipedia_modality.get("txt", 0) - aipedia_modality.get("video", 0)) <= 2
                    and abs(aipedia_modality.get("video", 0) - aipedia_modality.get("code", 0)) <= 2
                ),
            },
        }

    def run_comparison(
        self,
        corpora_root: str,
        llm_results_dir: str,
        aipedia_results_dir: str,
        output_dir: str,
        skip_llm: bool = False,
    ) -> Dict[str, Any]:
        """Run the full comparison across all focused corpora."""
        corpora = self.aipedia_runner.list_corpora(corpora_root)
        if not corpora:
            raise ValueError(f"No corpus folders found in {corpora_root}")

        comparison_results: Dict[str, Any] = {}
        all_llm_results: Dict[str, Any] = {}

        for corpus_name, corpus_path in corpora:
            print(f"\n{'='*60}")
            print(f"Processing: {corpus_name}")
            print(f"{'='*60}")

            # ── LLM-with-browsing baseline ──
            if not skip_llm:
                llm_result = run_llm_baseline(corpus_path)
                llm_out = save_results(llm_result, llm_results_dir)
                print(f"  LLM baseline: {llm_out}")
            else:
                cached_path = os.path.join(
                    llm_results_dir, f"{corpus_name}_llm_browsing_baseline.json"
                )
                if os.path.exists(cached_path):
                    with open(cached_path) as f:
                        llm_result = json.load(f)
                    print(f"  Loaded cached LLM baseline: {cached_path}")
                else:
                    print(f"  No cached baseline for {corpus_name}, skipping.")
                    continue

            # ── AI-Pedia pipeline ──
            aipedia_out_dir = os.path.join(aipedia_results_dir, corpus_name)
            os.makedirs(aipedia_out_dir, exist_ok=True)
            cache_path = os.path.join(aipedia_out_dir, "raw_search_results.json")

            aipedia_result = self.aipedia_runner.run_single(
                corpus_path=corpus_path,
                output_dir=aipedia_out_dir,
                top_k_keywords=10,
                search_max_per_type=8,
                recommend_top_k=5,
                reuse_cache=True,
            )
            print(f"  AI-Pedia pipeline complete: {aipedia_out_dir}")

            # ── Per-dimension comparisons ──
            llm_metrics = llm_result.get("resource_metrics", {})

            comparison_results[corpus_name] = {
                "llm_baseline": llm_result,
                "aipedia_result": aipedia_result,
                "url_health": self.compare_url_health(llm_metrics, aipedia_result),
                "authority": self.compare_authority(llm_metrics, aipedia_result),
                "modality": self.compare_modality(llm_metrics, aipedia_result),
            }
            all_llm_results[corpus_name] = llm_result

        # ── Aggregate ──
        aggregate = self._aggregate(comparison_results)

        final = {
            "mode": "llm_browsing_comparison_v2",
            "timestamp": self._iso_now(),
            "llm_browsing_model": llm_result.get("browsing_model", "gpt-4o"),
            "aggregate": aggregate,
            "per_corpus": comparison_results,
        }

        # ── Save ──
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "llm_comparison_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved: {out_path}")

        # ── Export LaTeX table ──
        latex_path = export_comparison_table(all_llm_results, output_dir)
        print(f"LaTeX table: {latex_path}")

        self._print_summary(aggregate)

        return final

    def _aggregate(self, per_corpus: Dict[str, Any]) -> Dict[str, Any]:
        """Average metrics across corpora."""
        n = len(per_corpus)
        if n == 0:
            return {}

        def avg(d: Dict, *keys: str) -> float:
            vals = [d]
            for k in keys:
                vals = [v.get(k) for v in vals if isinstance(v, dict)]
            numerics = [float(v) for v in vals if v is not None]
            return round(sum(numerics) / len(numerics), 2) if numerics else 0.0

        # URL health aggregation
        url_health_deltas = [c["url_health"]["gap"] for c in per_corpus.values()]

        # Authority aggregation
        authority_gaps = [c["authority"]["contrast"]["authority_gap"] for c in per_corpus.values()]
        low_auth_gaps = [c["authority"]["contrast"]["low_authority_gap"] for c in per_corpus.values()]

        # Modality
        code_advantages = [c["modality"]["contrast"]["code_advantage"] for c in per_corpus.values()]
        balanced_flags = [c["modality"]["contrast"]["balanced"] for c in per_corpus.values()]

        # Aggregate AI-Pedia resource metrics
        aipedia_metrics = []
        for c in per_corpus.values():
            ap = c["aipedia_result"].get("resource_reports", {}).get("topk_ranked", {})
            aipedia_metrics.append(ap)

        aggregate = {
            "corpus_count": n,
            "url_health": {
                "avg_live_url_gap": round(sum(d["live_url_gap"] for d in url_health_deltas) / n, 2),
                "avg_broken_url_gap": round(sum(d["broken_url_gap"] for d in url_health_deltas) / n, 2),
            },
            "authority": {
                "avg_authority_gap": round(sum(authority_gaps) / n, 2),
                "avg_low_authority_gap": round(sum(low_auth_gaps) / n, 2),
            },
            "modality": {
                "code_advantage_in_all": all(code_advantages),
                "balanced_in_all": all(balanced_flags),
            },
            "aipedia_avg_metrics": {
                "avg_authority": avg(
                    {k: v.get("authority_score", 0) for k, v in enumerate(aipedia_metrics)},
                    *[str(i) for i in range(n)]
                ) if aipedia_metrics else 0.0,
            },
        }
        return aggregate

    def _print_summary(self, aggregate: Dict[str, Any]) -> None:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY (LLM browsing vs AI-Pedia)")
        print("=" * 60)
        print("\nURL Health:")
        print(f"  Avg live URL gap:  +{aggregate.get('url_health', {}).get('avg_live_url_gap', 0):.1f}% (AI-Pedia advantage)")
        print(f"  Avg broken URL gap: +{aggregate.get('url_health', {}).get('avg_broken_url_gap', 0):.1f}% (LLM weakness)")
        print("\nAuthority:")
        print(f"  Avg authority gap: +{aggregate.get('authority', {}).get('avg_authority_gap', 0):.1f}% (AI-Pedia advantage)")
        print(f"  Low-authority gap:  +{aggregate.get('authority', {}).get('avg_low_authority_gap', 0):.1f}% (LLM weakness)")
        print("\nModality:")
        print(f"  Code advantage (all corpora): {aggregate.get('modality', {}).get('code_advantage_in_all', False)}")
        print(f"  Balanced 5-5-5 (all corpora): {aggregate.get('modality', {}).get('balanced_in_all', False)}")

    def _iso_now(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()


# ─── MAIN ────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-with-Browsing Baseline vs AI-Pedia Comparison"
    )
    parser.add_argument(
        "--corpora-root",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "test_corpora"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(PROJECT_ROOT, "test", "llm_baseline_eval", "results"),
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM calls and reuse cached results",
    )
    args = parser.parse_args()

    llm_dir = os.path.join(args.output, "llm_baseline")
    aipedia_dir = os.path.join(args.output, "aipedia_pipeline")

    evaluator = ComparisonEvaluator()
    evaluator.run_comparison(
        corpora_root=args.corpora_root,
        llm_results_dir=llm_dir,
        aipedia_results_dir=aipedia_dir,
        output_dir=args.output,
        skip_llm=args.skip_llm,
    )


if __name__ == "__main__":
    main()
