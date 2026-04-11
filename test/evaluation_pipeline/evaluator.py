#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restricted Baseline vs Full AI-Pedia Pipeline Comparison

This module implements the core evaluation methodology:
1. Restricted Baseline: Simple keyword matching without advanced processing
2. Full Pipeline: TF-IDF + MMR + AI Filtering + CBF Ranking
3. Side-by-side comparison using the same foundation model
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend.core.keyword_extractor import extract_keywords_from_folder
from backend.core.resource_searcher import search_all_sources
from backend.core.recommender import recommend_resources
from test.evaluation_pipeline.config import EvalConfig, load_config
from test.evaluation_pipeline.metrics import EvaluationMetrics


class RestrictedBaseline:
    """
    Restricted baseline that simulates a naive resource discovery approach.
    Uses simple keyword extraction without TF-IDF weighting or MMR diversity.
    """

    def __init__(self):
        self.config = load_config()
        self.metrics = EvaluationMetrics()

    def extract_simple_keywords(self, documents: List[str], top_k: int = 10) -> List[str]:
        """
        Extract keywords using simple frequency counting (no TF-IDF, no MMR).
        This simulates a naive student approach.
        """
        from collections import Counter
        import re

        # Simple word frequency
        word_counts = Counter()
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "being", "have", "has", "had", "do", "does", "did", "will",
                     "would", "could", "should", "may", "might", "must", "shall",
                     "can", "need", "dare", "ought", "used", "to", "of", "in", "for",
                     "on", "with", "at", "by", "from", "as", "into", "through",
                     "during", "before", "after", "above", "below", "between",
                     "and", "but", "or", "nor", "so", "yet", "both", "either",
                     "neither", "not", "only", "just", "also", "very", "too",
                     "quite", "rather", "this", "that", "these", "those", "it"}

        for doc in documents:
            # Simple tokenization
            words = re.findall(r'\b[a-zA-Z]{3,}\b', doc.lower())
            words = [w for w in words if w not in stop_words]
            word_counts.update(words)

        # Return top-k most frequent words (no diversity consideration)
        return [word for word, count in word_counts.most_common(top_k)]

    def search_naive(self, keywords: List[str], max_results: int = 20) -> List[Dict]:
        """
        Perform naive search without any filtering or ranking.
        Just return raw search results in arbitrary order.
        """
        # Simple search - just get results from Wikipedia
        results = []
        from backend.core.resource_searcher import search_wikipedia

        for keyword in keywords[:5]:  # Limit to first 5 keywords
            try:
                wiki_results = search_wikipedia(keyword, max_results // 5)
                results.extend(wiki_results)
            except Exception as e:
                print(f"Warning: Search failed for keyword '{keyword}': {e}")

        return results[:max_results]  # Return first max_results


class FullAIPediaPipeline:
    """
    Full AI-Pedia pipeline with all advanced features:
    - TF-IDF + MMR keyword extraction
    - Multi-source search
    - AI-domain filtering
    - CBF ranking
    """

    def __init__(self):
        self.config = load_config()
        self.metrics = EvaluationMetrics()

    def process(self, corpus_path: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Process corpus through full AI-Pedia pipeline.
        Returns recommended resources.
        """
        # Step 1: Extract keywords with TF-IDF + MMR
        print("📝 Step 1: Extracting keywords with TF-IDF + MMR...")
        keywords = extract_keywords_from_folder(corpus_path, top_k=top_k)

        # Step 2: Multi-source search
        print("🔍 Step 2: Searching multiple sources...")
        resources = search_all_sources(keywords)

        # Step 3: AI-domain filtering (built into search)
        print("🧹 Step 3: AI-domain filtering...")
        # Filtering is already applied in search_all_sources

        # Step 4: CBF ranking
        print("📊 Step 4: Computing CBF similarity and ranking...")
        ranked_resources = recommend_resources(corpus_path, resources, top_k=top_k)

        return {
            "keywords": keywords,
            "resources": ranked_resources,
            "resource_count": len(ranked_resources)
        }


class ComparisonEvaluator:
    """
    Compare restricted baseline vs full AI-Pedia pipeline.
    Uses the same foundation model to isolate pipeline contribution.
    """

    def __init__(self):
        self.restricted = RestrictedBaseline()
        self.full_pipeline = FullAIPediaPipeline()
        self.metrics = EvaluationMetrics()

    def run_evaluation(self, corpus_path: str, output_dir: str = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation comparing both approaches.
        """
        print("=" * 60)
        print("AI-Pedia Evaluation: Restricted vs Full Pipeline")
        print("=" * 60)

        # Read corpus documents
        import glob
        txt_files = glob.glob(os.path.join(corpus_path, "*.txt"))
        documents = []
        for f in txt_files:
            with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                documents.append(file.read())

        if not documents:
            print("⚠️  No documents found in corpus!")
            return {}

        print(f"📂 Loaded {len(documents)} documents from corpus")

        # ========== RESTRICTED BASELINE ==========
        print("\n" + "=" * 60)
        print("Running RESTRICTED Baseline (Naive Search)...")
        print("=" * 60)

        # Simple keyword extraction
        restricted_keywords = self.restricted.extract_simple_keywords(documents)
        print(f"📌 Extracted keywords: {restricted_keywords}")

        # Naive search
        baseline_resources = self.restricted.search_naive(restricted_keywords)
        print(f"📥 Found {len(baseline_resources)} resources")

        # Evaluate baseline
        baseline_report = self.metrics.generate_report(
            baseline_resources,
            initial_count=len(baseline_resources)
        )

        print("\n📊 Restricted Baseline Results:")
        for key, value in baseline_report.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

        # ========== FULL PIPELINE ==========
        print("\n" + "=" * 60)
        print("Running FULL AI-Pedia Pipeline...")
        print("=" * 60)

        try:
            pipeline_result = self.full_pipeline.process(corpus_path)
            pipeline_resources = pipeline_result["resources"]
            pipeline_keywords = pipeline_result["keywords"]

            print(f"📌 Extracted keywords: {pipeline_keywords}")
            print(f"📥 Found {len(pipeline_resources)} resources")

            # Evaluate full pipeline
            pipeline_report = self.metrics.generate_report(
                pipeline_resources,
                initial_count=len(baseline_resources)  # Use baseline count for comparison
            )

            print("\n📊 Full Pipeline Results:")
            for key, value in pipeline_report.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")

        except Exception as e:
            print(f"⚠️  Full pipeline evaluation failed: {e}")
            pipeline_report = {}
            pipeline_resources = []

        # ========== COMPARISON ==========
        print("\n" + "=" * 60)
        print("COMPARISON: Restricted vs Full Pipeline")
        print("=" * 60)

        comparison = {
            "baseline": baseline_report,
            "full_pipeline": pipeline_report,
            "improvements": {}
        }

        # Calculate improvements
        if pipeline_report and baseline_report:
            improvements = {}

            # AI Relevance improvement
            if "ai_relevance" in baseline_report and "ai_relevance" in pipeline_report:
                improvements["ai_relevance_delta"] = (
                    pipeline_report["ai_relevance"] - baseline_report["ai_relevance"]
                )

            # Noise reduction
            if "noise_reduction" in pipeline_report:
                improvements["noise_reduction"] = pipeline_report["noise_reduction"]

            # Diversity improvement
            if "cross_platform_diversity" in baseline_report and "cross_platform_diversity" in pipeline_report:
                improvements["diversity_delta"] = (
                    pipeline_report["cross_platform_diversity"] - baseline_report["cross_platform_diversity"]
                )

            comparison["improvements"] = improvements

            print("\n🚀 Key Improvements:")
            for key, value in improvements.items():
                print(f"  {key}: {value:+.1f}")

        # Save results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "evaluation_results.json")
            with open(output_file, 'w') as f:
                json.dump(comparison, f, indent=2)
            print(f"\n💾 Results saved to: {output_file}")

        return comparison


def main():
    """Main evaluation entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AI-Pedia Evaluation: Restricted vs Full Pipeline"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path to test corpus directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Use default corpus if not provided
    corpus_path = args.corpus or os.path.join(
        os.path.dirname(__file__),
        "..", "..", "data", "test_corpus"
    )

    output_dir = args.output or os.path.join(
        os.path.dirname(__file__),
        "results"
    )

    # Run evaluation
    evaluator = ComparisonEvaluator()
    results = evaluator.run_evaluation(corpus_path, output_dir)

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()