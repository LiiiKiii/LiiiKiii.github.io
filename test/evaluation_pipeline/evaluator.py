#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-Pedia evaluation runner.

This script focuses on a defensible, code-backed comparison that the current
project can actually execute:
1. simple frequency-based keyword baseline vs TF-IDF+MMR keyword extraction
2. raw multi-source search results vs top-k unranked selection vs CBF-ranked output
3. JSON + CSV + LaTeX tables + chart outputs for direct use in the paper
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List

import matplotlib.pyplot as plt

# Add local evaluation dir and project root to import path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(1, PROJECT_ROOT)

from backend.core.keyword_extractor import extract_keywords_from_folder
from backend.core.recommender import recommend_best_resources
from backend.core.resource_searcher import search_all_resources
from config import EvalConfig, load_config
from metrics import EvaluationMetrics


class EvaluationRunner:
    """Run the practical evaluation workflow for AI-Pedia."""

    def __init__(self):
        self.config = load_config()
        self.metrics = EvaluationMetrics(EvalConfig.AI_RELEVANCE_KEYWORDS)

    def load_documents(self, corpus_path: str) -> List[str]:
        documents: List[str] = []
        for root, _, files in os.walk(corpus_path):
            for name in sorted(files):
                if not name.lower().endswith(".txt") or name.startswith("._") or name.startswith(".DS_Store"):
                    continue
                file_path = os.path.join(root, name)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
                    content = handle.read().strip()
                    if content:
                        documents.append(content)
        return documents

    def extract_simple_keywords(self, documents: List[str], top_k: int = 10) -> List[str]:
        import re

        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between",
            "and", "but", "or", "nor", "so", "yet", "both", "either",
            "neither", "not", "only", "just", "also", "very", "too",
            "quite", "rather", "this", "that", "these", "those", "it",
        }

        word_counts = Counter()
        for doc in documents:
            words = re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", doc.lower())
            words = [word for word in words if word not in stop_words]
            word_counts.update(words)

        return [word for word, _ in word_counts.most_common(top_k)]

    def take_first_k_per_type(self, all_resources: Dict[str, List[Dict[str, Any]]], top_k: int) -> Dict[str, List[Dict[str, Any]]]:
        return {resource_type: list(items[:top_k]) for resource_type, items in all_resources.items()}

    def save_json(self, path: str, payload: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def load_json(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def save_csv(self, path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def save_text(self, path: str, content: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)

    def export_tables(self, results: Dict[str, Any], output_dir: str) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        exported: Dict[str, str] = {}

        keyword_simple = results["keyword_reports"]["simple_frequency"]
        keyword_full = results["keyword_reports"]["tfidf_mmr"]
        raw = results["resource_reports"]["raw_search"]
        unranked = results["resource_reports"]["topk_unranked"]
        ranked = results["resource_reports"]["topk_ranked"]

        keyword_rows = [
            {
                "method": "Simple frequency baseline",
                "keyword_count": keyword_simple["keyword_count"],
                "coverage_pct": keyword_simple["coverage"],
                "ai_relevance_pct": keyword_simple["ai_relevance"],
                "diversity": keyword_simple["diversity"],
            },
            {
                "method": "TF-IDF + MMR",
                "keyword_count": keyword_full["keyword_count"],
                "coverage_pct": keyword_full["coverage"],
                "ai_relevance_pct": keyword_full["ai_relevance"],
                "diversity": keyword_full["diversity"],
            },
        ]
        keyword_csv = os.path.join(output_dir, "keyword_metrics.csv")
        self.save_csv(
            keyword_csv,
            ["method", "keyword_count", "coverage_pct", "ai_relevance_pct", "diversity"],
            keyword_rows,
        )
        exported["keyword_csv"] = keyword_csv

        resource_rows = [
            {
                "stage": "Raw search",
                "total_resources": raw["total_resources"],
                "ai_relevance_pct": raw["ai_relevance"],
                "authority_score_pct": raw["authority_score"],
                "valid_url_pct": raw["url_validation"]["valid_percentage"],
                "noise_reduction_pct": raw.get("noise_reduction", 0.0),
                "cross_platform_diversity": raw["cross_platform_diversity"],
            },
            {
                "stage": "Top-k unranked",
                "total_resources": unranked["total_resources"],
                "ai_relevance_pct": unranked["ai_relevance"],
                "authority_score_pct": unranked["authority_score"],
                "valid_url_pct": unranked["url_validation"]["valid_percentage"],
                "noise_reduction_pct": unranked.get("noise_reduction", 0.0),
                "cross_platform_diversity": unranked["cross_platform_diversity"],
            },
            {
                "stage": "Top-k ranked",
                "total_resources": ranked["total_resources"],
                "ai_relevance_pct": ranked["ai_relevance"],
                "authority_score_pct": ranked["authority_score"],
                "valid_url_pct": ranked["url_validation"]["valid_percentage"],
                "noise_reduction_pct": ranked.get("noise_reduction", 0.0),
                "cross_platform_diversity": ranked["cross_platform_diversity"],
            },
        ]
        resource_csv = os.path.join(output_dir, "resource_metrics.csv")
        self.save_csv(
            resource_csv,
            [
                "stage",
                "total_resources",
                "ai_relevance_pct",
                "authority_score_pct",
                "valid_url_pct",
                "noise_reduction_pct",
                "cross_platform_diversity",
            ],
            resource_rows,
        )
        exported["resource_csv"] = resource_csv

        latex_path = os.path.join(output_dir, "evaluation_tables.tex")
        latex_content = """%% Auto-generated by test/evaluation_pipeline/evaluator.py
\\begin{table}[!t]
\\renewcommand{\\arraystretch}{1.2}
\\caption{Keyword Extraction on the Pilot Corpus}
\\label{tab:keyword_quality_auto}
\\centering
\\footnotesize
\\begin{tabular}{@{}lccc@{}}
\\hline\\hline
Method & Keywords & Coverage (\\%%) & AI rel. (\\%%)\\\\
\\hline
Simple frequency baseline & %(simple_keyword_count)s & %(simple_coverage).2f & %(simple_ai).2f\\\\
TF-IDF + MMR & %(full_keyword_count)s & %(full_coverage).2f & %(full_ai).2f\\\\
\\hline\\hline
\\end{tabular}
\\end{table}

\\begin{table}[!t]
\\renewcommand{\\arraystretch}{1.2}
\\caption{Resource Quality Across Retrieval and Ranking Stages}
\\label{tab:pipeline_comparison_auto}
\\centering
\\footnotesize
\\begin{tabular}{@{}lccc@{}}
\\hline\\hline
Metric & Raw search & Top-$K$ unranked & Top-$K$ ranked\\\\
\\hline
Total resources & %(raw_total)s & %(unranked_total)s & %(ranked_total)s\\\\
AI relevance (\\%%) & %(raw_ai).2f & %(unranked_ai).2f & %(ranked_ai).2f\\\\
Authority score (\\%%) & %(raw_auth).2f & %(unranked_auth).2f & %(ranked_auth).2f\\\\
Valid URLs (\\%%) & %(raw_url).2f & %(unranked_url).2f & %(ranked_url).2f\\\\
Noise reduction (\\%%) & %(raw_noise).2f & %(unranked_noise).2f & %(ranked_noise).2f\\\\
Cross-platform diversity & %(raw_div)s & %(unranked_div)s & %(ranked_div)s\\\\
\\hline\\hline
\\end{tabular}
\\end{table}
""" % {
            "simple_keyword_count": keyword_simple["keyword_count"],
            "simple_coverage": keyword_simple["coverage"],
            "simple_ai": keyword_simple["ai_relevance"],
            "full_keyword_count": keyword_full["keyword_count"],
            "full_coverage": keyword_full["coverage"],
            "full_ai": keyword_full["ai_relevance"],
            "raw_total": raw["total_resources"],
            "unranked_total": unranked["total_resources"],
            "ranked_total": ranked["total_resources"],
            "raw_ai": raw["ai_relevance"],
            "unranked_ai": unranked["ai_relevance"],
            "ranked_ai": ranked["ai_relevance"],
            "raw_auth": raw["authority_score"],
            "unranked_auth": unranked["authority_score"],
            "ranked_auth": ranked["authority_score"],
            "raw_url": raw["url_validation"]["valid_percentage"],
            "unranked_url": unranked["url_validation"]["valid_percentage"],
            "ranked_url": ranked["url_validation"]["valid_percentage"],
            "raw_noise": raw.get("noise_reduction", 0.0),
            "unranked_noise": unranked.get("noise_reduction", 0.0),
            "ranked_noise": ranked.get("noise_reduction", 0.0),
            "raw_div": raw["cross_platform_diversity"],
            "unranked_div": unranked["cross_platform_diversity"],
            "ranked_div": ranked["cross_platform_diversity"],
        }
        self.save_text(latex_path, latex_content)
        exported["latex_tables"] = latex_path

        return exported

    def generate_plots(self, results: Dict[str, Any], plots_dir: str) -> Dict[str, str]:
        os.makedirs(plots_dir, exist_ok=True)
        generated: Dict[str, str] = {}

        # Plot 1: pipeline comparison (raw vs unranked top-k vs ranked top-k)
        raw = results["resource_reports"]["raw_search"]
        unranked = results["resource_reports"]["topk_unranked"]
        ranked = results["resource_reports"]["topk_ranked"]

        labels = ["Raw search", "Top-k unranked", "CBF ranked"]
        metrics = ["ai_relevance", "authority_score", "valid_url_pct"]
        values = {
            "Raw search": [raw["ai_relevance"], raw["authority_score"], raw["url_validation"]["valid_percentage"]],
            "Top-k unranked": [unranked["ai_relevance"], unranked["authority_score"], unranked["url_validation"]["valid_percentage"]],
            "CBF ranked": [ranked["ai_relevance"], ranked["authority_score"], ranked["url_validation"]["valid_percentage"]],
        }

        x = range(len(metrics))
        width = 0.24
        plt.figure(figsize=(9, 5))
        for idx, label in enumerate(labels):
            offset = [pos + (idx - 1) * width for pos in x]
            plt.bar(offset, values[label], width=width, label=label)
        plt.xticks(list(x), ["AI relevance", "Authority", "Valid URLs"], rotation=0)
        plt.ylabel("Percentage")
        plt.ylim(0, 100)
        plt.title("AI-Pedia pipeline quality comparison")
        plt.legend()
        plt.tight_layout()
        pipeline_plot = os.path.join(plots_dir, "evaluation_pipeline_comparison.png")
        plt.savefig(pipeline_plot, dpi=220)
        plt.close()
        generated["pipeline_comparison"] = pipeline_plot

        # Plot 2: keyword quality comparison (simple baseline vs TF-IDF+MMR)
        keyword_simple = results["keyword_reports"]["simple_frequency"]
        keyword_full = results["keyword_reports"]["tfidf_mmr"]

        plt.figure(figsize=(8, 5))
        keyword_metrics = ["coverage", "ai_relevance", "diversity"]
        keyword_labels = ["Coverage", "AI relevance", "Diversity"]
        simple_values = [keyword_simple["coverage"], keyword_simple["ai_relevance"], keyword_simple["diversity"] * 100]
        full_values = [keyword_full["coverage"], keyword_full["ai_relevance"], keyword_full["diversity"] * 100]
        x2 = range(len(keyword_metrics))
        plt.bar([pos - 0.18 for pos in x2], simple_values, width=0.36, label="Simple frequency")
        plt.bar([pos + 0.18 for pos in x2], full_values, width=0.36, label="TF-IDF + MMR")
        plt.xticks(list(x2), keyword_labels)
        plt.ylabel("Percentage / normalized score")
        plt.ylim(0, 100)
        plt.title("Keyword extraction quality comparison")
        plt.legend()
        plt.tight_layout()
        keyword_plot = os.path.join(plots_dir, "evaluation_precision_comparison.png")
        plt.savefig(keyword_plot, dpi=220)
        plt.close()
        generated["keyword_comparison"] = keyword_plot

        # Plot 3: resource type distribution before and after ranking
        raw_counts = results["resource_type_counts"]["raw_search"]
        ranked_counts = results["resource_type_counts"]["topk_ranked"]
        type_labels = ["txt", "video", "code"]
        raw_values = [raw_counts.get(label, 0) for label in type_labels]
        ranked_values = [ranked_counts.get(label, 0) for label in type_labels]

        plt.figure(figsize=(8, 5))
        x3 = range(len(type_labels))
        plt.bar([pos - 0.18 for pos in x3], raw_values, width=0.36, label="Raw search")
        plt.bar([pos + 0.18 for pos in x3], ranked_values, width=0.36, label="Final ranked output")
        plt.xticks(list(x3), ["Text", "Video", "Code"])
        plt.ylabel("Resource count")
        plt.title("Resource type distribution before and after ranking")
        plt.legend()
        plt.tight_layout()
        distribution_plot = os.path.join(plots_dir, "resource_type_distribution.png")
        plt.savefig(distribution_plot, dpi=220)
        plt.close()
        generated["resource_type_distribution"] = distribution_plot

        return generated

    def run(
        self,
        corpus_path: str,
        output_dir: str,
        top_k_keywords: int = 10,
        search_max_per_type: int = 8,
        recommend_top_k: int = 5,
        reuse_cache: bool = False,
    ) -> Dict[str, Any]:
        documents = self.load_documents(corpus_path)
        if len(documents) < 2:
            raise ValueError("Evaluation needs at least 2 TXT documents in the corpus.")

        os.makedirs(output_dir, exist_ok=True)
        cache_path = os.path.join(output_dir, "raw_search_results.json")

        simple_keywords = self.extract_simple_keywords(documents, top_k=top_k_keywords)
        tfidf_keywords = extract_keywords_from_folder(corpus_path, top_k=top_k_keywords)

        keyword_reports = {
            "simple_frequency": self.metrics.generate_keyword_report(documents, simple_keywords),
            "tfidf_mmr": self.metrics.generate_keyword_report(documents, tfidf_keywords),
        }

        if reuse_cache and os.path.exists(cache_path):
            raw_resources = self.load_json(cache_path)
        else:
            raw_resources = search_all_resources(tfidf_keywords, max_per_type=search_max_per_type)
            self.save_json(cache_path, raw_resources)

        raw_flat = self.metrics.flatten_resources(raw_resources)
        topk_unranked = self.take_first_k_per_type(raw_resources, recommend_top_k)
        topk_unranked_flat = self.metrics.flatten_resources(topk_unranked)
        ranked_resources = recommend_best_resources(corpus_path, raw_resources, top_k_per_type=recommend_top_k)
        ranked_flat = self.metrics.flatten_resources(ranked_resources)

        resource_reports = {
            "raw_search": self.metrics.generate_resource_report(raw_flat, original_corpus=documents),
            "topk_unranked": self.metrics.generate_resource_report(
                topk_unranked_flat,
                initial_count=len(raw_flat),
                original_corpus=documents,
            ),
            "topk_ranked": self.metrics.generate_resource_report(
                ranked_flat,
                initial_count=len(raw_flat),
                original_corpus=documents,
            ),
        }

        improvements = {
            "keyword_coverage_delta": round(
                keyword_reports["tfidf_mmr"]["coverage"] - keyword_reports["simple_frequency"]["coverage"],
                2,
            ),
            "keyword_ai_relevance_delta": round(
                keyword_reports["tfidf_mmr"]["ai_relevance"] - keyword_reports["simple_frequency"]["ai_relevance"],
                2,
            ),
            "keyword_diversity_delta": round(
                keyword_reports["tfidf_mmr"]["diversity"] - keyword_reports["simple_frequency"]["diversity"],
                4,
            ),
            "ranking_ai_relevance_delta": round(
                resource_reports["topk_ranked"]["ai_relevance"] - resource_reports["topk_unranked"]["ai_relevance"],
                2,
            ),
            "ranking_authority_delta": round(
                resource_reports["topk_ranked"]["authority_score"] - resource_reports["topk_unranked"]["authority_score"],
                2,
            ),
            "ranking_noise_reduction": resource_reports["topk_ranked"].get("noise_reduction", 0.0),
        }

        results = {
            "corpus": {
                "path": corpus_path,
                "document_count": len(documents),
                "word_count_total": sum(len(doc.split()) for doc in documents),
            },
            "parameters": {
                "top_k_keywords": top_k_keywords,
                "search_max_per_type": search_max_per_type,
                "recommend_top_k": recommend_top_k,
            },
            "keyword_reports": keyword_reports,
            "resource_reports": resource_reports,
            "resource_type_counts": {
                "raw_search": dict(Counter(resource.get("resource_type", "unknown") for resource in raw_flat)),
                "topk_unranked": dict(Counter(resource.get("resource_type", "unknown") for resource in topk_unranked_flat)),
                "topk_ranked": dict(Counter(resource.get("resource_type", "unknown") for resource in ranked_flat)),
            },
            "improvements": improvements,
        }

        results_path = os.path.join(output_dir, "evaluation_results.json")
        self.save_json(results_path, results)
        return results


def default_paper_figures_dir() -> str:
    candidate = os.path.join(
        PROJECT_ROOT,
        "Paper",
        "L3-CS Project Paper Template (LaTeX)",
        "figures",
    )
    return candidate if os.path.isdir(candidate) else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-Pedia evaluation runner")
    parser.add_argument("--corpus", type=str, default=EvalConfig.TEST_CORPUS_PATH)
    parser.add_argument("--output", type=str, default=EvalConfig.OUTPUT_DIR)
    parser.add_argument("--plots-dir", type=str, default=None)
    parser.add_argument("--tables-dir", type=str, default=None)
    parser.add_argument("--top-k-keywords", type=int, default=10)
    parser.add_argument("--search-max-per-type", type=int, default=8)
    parser.add_argument("--recommend-top-k", type=int, default=5)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()

    runner = EvaluationRunner()
    results = runner.run(
        corpus_path=args.corpus,
        output_dir=args.output,
        top_k_keywords=args.top_k_keywords,
        search_max_per_type=args.search_max_per_type,
        recommend_top_k=args.recommend_top_k,
        reuse_cache=args.reuse_cache,
    )

    plots_dir = args.plots_dir or default_paper_figures_dir() or args.output
    tables_dir = args.tables_dir or args.output
    generated = runner.generate_plots(results, plots_dir)
    exported_tables = runner.export_tables(results, tables_dir)

    print("=" * 60)
    print("AI-Pedia evaluation complete")
    print("=" * 60)
    print(json.dumps(results["improvements"], indent=2, ensure_ascii=False))
    print("Generated plots:")
    for name, path in generated.items():
        print(f"- {name}: {path}")
    print("Exported tables:")
    for name, path in exported_tables.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
